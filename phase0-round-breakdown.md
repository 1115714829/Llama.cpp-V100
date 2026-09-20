# phase0-round-breakdown.md — Phase 0：基线 + 每轮成本分解（2026-09-20）

> 目标：按 1cat-vLLM 的方法在 V100 上压 llama.cpp 的**每轮延迟**。
> 主参考 = 1cat-vLLM（一个完整 B1 DFlash2 轮 = **18.465–18.603 ms**，MBPP28 18.567 ms / AL 4.686 / **251.6 tok/s**，
> TP4 + FULL target+draft CUDA Graphs + NVFP4 + E4M3 KV + BF16 lm_head）。
> 本文件记录 llama.cpp 侧的基线与其**每轮成分**（实测，不是推断）。

## 1. 模型与环境（Phase 0.1）

| 项 | 值 |
|---|---|
| target | `Qwen3.8-27B-Q8_0.gguf`，**29,047,086,048 B**（ModelScope `unsloth/Qwen3.8-27B-GGUF`） |
| 校验 | `sha256 = a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348` **完全一致** |
| draft | `/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`（1.14 GB，官方 DFlash2） |
| 卡 | GPU 0/1/2（同一 NUMA node），`--split-mode tensor --tensor-split 1,1,1` |
| 其他 | `--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0`，ctx 8192，官方采样 temp 1.0 / top_p 0.95 / top_k 20 |

**为什么用 Q8_0 而不是 FP8**：用户的生产 1cat 服务跑 FP8；llama.cpp 无 FP8 CUDA 路径。Q8_0 ≈ 8.5 bpw
（FP8 E4M3 ≈ 8.0 bpw）⇒ **每 token 多读约 6% 字节**，数值上 Q8_0 通常不差于 FP8。**这是口径近似，报告里必须标注**。
（另注：真正与 1cat 的 206–251 tok/s 对标的行是 **NVFP4**；FP8 256K 无 MTP 他们只有 41.11。）

## 2. 基线（Phase 0.3）

`Q8_0 target + DFlash2 n=7`，3 卡，3 个 prompt，seed 42，n_predict 512：

| request | tg | ms/token | AL (mean len) | 接受率 |
|---|---|---|---|---|
| 1 | **44.44** | 22.50 | 4.11 | 0.447 |
| 2 | **57.12** | 17.51 | 5.21 | 0.605 |
| 3 | **65.43** | 15.28 | 6.08 | 0.726 |

**硬门通过**：AL = 4.11 / 5.21 / 6.08 落在 4–6 ⇒ **Q8_0 target 与基座 DFlash2 draft 兼容**（不像 TurboFCFusion 那样掉到 1.2–1.4），基线有效。

**与 1cat 的同一标尺对比**：我们最好 65.43 tok/s，1cat 206–263 tok/s（短上下文，4 卡）。差距 ~3–4×。

## 3. 量具（Phase 0.2）—— 两个可用通道，一个教训

新增（env-gated，未设时零开销）：
- `LLAMA_ROUND_TIMING`：在 `src/llama-context.cpp::process_ubatch` 累计 **build_graph / sched_alloc_graph /
  set_inputs / graph_compute** 四段耗时 + `reuse`/`rebuild` 计数，经 `perf_get_data()` 输出。
- `LLAMA_SPEC_TIMING`：在 `common/speculative.cpp` 的 DFlash2 `draft()` 里累计 **draft_decode（draft 前向）/
  selector（CPU 侧 selector）/ walk（候选走查）**，每 16 轮打印一次。

**教训（重要，省了大量时间）**：
1. **`llama-bench` 完全不打印 libllama 的 INFO 日志**（实测 0 条）⇒ **不能用 llama-bench 验证"库内部日志/埋点是否生效"**。
   server 会打印（带 `slot print_timing:` 前缀）。
2. **libllama 的埋点用 `LLAMA_LOG_INFO` 可能看不到**（原因未查明，试了 3 次都不出），
   但 **`fprintf(stderr, ...)` 一定可见** ⇒ 埋点先用 fprintf 验证"函数是否真的被执行"，再切回日志。
3. **这台机器是网络启动的无盘系统**：`/` 是 NFS ⇒ 下载、加载模型、**甚至编译**都吃同一张网卡；
   下载与模型加载基本打满网卡。**不要在下载/加载的同时编译或跑测**。
4. 29 GB 模型的首轮加载在 page cache 命中后只要 ~15 s；`sha256sum` 也是 29 GB 读取，别和跑测并行。

## 4. ★每轮成本分解（Phase 0 的核心产出）

配置：Q8_0 + DFlash2 `n=7`，3 卡，512 tokens ⇒ **55.49 tok/s / 18.02 ms per token / AL 5.21**
⇒ **每轮 ≈ 18.02 × 5.21 ≈ 92.6 ms**（另一算法：9209.16 ms ÷ (512/5.21=98.3 轮) = 93.7 ms）。

| 成分 | ms/轮 | 占比 | 证据 |
|---|---|---|---|
| **target verify（M=8）** | **29.87** | **32%** | `[RT] perf: rounds=101 … enqueue_us=3017277`（M=8 时 `graph_compute` 是同步的，可信） |
| **CPU 侧 selector** | **23.44** | **25%** | `draft: spec timing: n=96 \| draft_decode=15.01 selector=23.44 walk=0.12 ms/round` |
| **draft 前向** | **15.01** | **16%** | 同上 |
| target 主机侧图工作（build+alloc+setin） | **~2.10** | **2%** | `build_us=12390 alloc_us=197555 setin_us=3384` ÷ 101 |
| 候选走查 | 0.12 | 0.1% | 同上 |
| 其余（CPU 采样 / 接受判定 / server 开销） | ~22 | 24% | 92.6 − 上述之和 |

**由此得出的三条结论：**

### 4.1 ★「多形状 decode graph 缓存」不是杠杆（原案 Phase 1.4 撤销）
主机侧三段合计 **2.10 ms/轮（2%）**，其中 alloc 1.94 ms 是大头（TP3 多卡，单卡时仅 0.18 ms）。
`reuse=94 / rebuild=7`（93% 复用率）。
⇒ **即使把主机侧图工作降到 0，收益上限也只有 ~2%。** 原计划把它当"预期最大收益"是**错的**（已由实测纠正）。
（注：其他 fork 的 "TP alloc 38 ms → 4-6 ms" 是**他们的配置**；我们这里是 1.94 ms，不是 38 ms。）

### 4.2 ★M=8 的 target forward 明显在带宽下限之上（Tensor Core 杠杆成立）
同一配置换臂（3 卡，同 prompt/seed，512 tokens）：

| 臂 | M | ms/token | AL | **ms/轮** |
|---|---|---|---|---|
| `--spec-type none` | 1 | 32.35 | 1.00 | **32.35** |
| `dflash n=3` | 4 | 19.49 | 3.45 | **67.2** |
| `dflash n=7` | 8 | 17.77 | 5.21 | **92.6** |

每多一行验证的**边际成本 ≈ 6.4–11.6 ms**（不是 0）⇒ **M=8 不是纯带宽受限**。
纯带宽理想下 M=8 的每行成本应 = 32.35/8 ≈ **4.0 ms**，实测 **11.6 ms** ⇒ **约 2.9× 于带宽下限**。
⇒ 这段"超出带宽下限"的计算量正是 **Tensor Core（HMMA）可攻击的部分**；1cat 的 m5/small-N HMMA 算子也正对着这里。

### 4.3 ★★我们自己的 CPU selector 值 23.4 ms/轮（最大单项，且是我们可控的代码）
`build_dflash2_selector_cpu()` 是我按 PR #27858 移植进来的，**只在 `--split-mode tensor` 下启用**
（`is_dflash2_cpu`）。它每轮对 8 个 block 位置各做一次 **全词表 top-k（n_vocab = 248,320，见 §7）** + gate 矩阵乘
（rank × n_embd），**全部在 POWER9 主机 CPU 单线程上跑** ⇒ 23.44 ms/轮。

**这条揭示了 Phase 1a 移植的隐藏代价**：修好崩溃是对的，但"在 tensor 切分下把 selector 放到 CPU"这个设计
本身每轮要付 23.4 ms。**要么优化它（并行/向量化/换更优的 top-k 选择），要么避免它（改用非 tensor 切分）。**

### 4.4 draft 前向 15.01 ms/轮也偏高
1.14 GB 的 draft 前向只用 15 ms ⇒ 有效带宽仅 **~76 GB/s**（峰值 ~900 GB/s 的单卡，~2700 GB/s 三卡合计）。
draft 比 target 小 25×，却占 target M=8 的一半时间 ⇒ **draft 前向是 dispatch/小 batch 受限**，不是带宽受限。
可能与 8-token 小 batch 的 kernel 效率、draft 是否只落在一张卡上有关。

## 5. 对原方案的修正（实测驱动）

| 原计划 | 实测后 | 处置 |
|---|---|---|
| Phase 1.4 多形状 decode graph 缓存（"预期最大收益"） | 主机侧共 2.1 ms/轮（2%） | **撤销/降级**（数据证明无价值） |
| Phase 1.5 verify 形状稳定（为图缓存服务） | 同上，收益上限 ~2% | **降级**（仅当 1.4 复活才需要） |
| Phase 2.8 M≈8 GEMM 走 Tensor Core | M=8 每行 11.6 ms vs 带宽下限 4.0 ms ⇒ **有 ~2.9× 空间** | **保留，且证据支持** |
| （新增）CPU selector 优化 | **23.4 ms/轮，最大单项，代码是我们自己的** | **新增为高优先** |
| （新增）改用非 tensor 切分以避开 CPU selector + CPU 采样 | 待测（layer 模式早先测过 24% 慢，但那是 target 不兼容时的数字） | **新增为待测** |
| （新增）draft 前向 15 ms 偏高（~76 GB/s） | 待查（小 batch kernel 效率 / 设备放置） | **新增为待测** |

## 6. 复现

- 量具源码改动：`src/llama-context.{h,cpp}`（`LLAMA_ROUND_TIMING`，fprintf 探针 + 计数）、
  `common/speculative.cpp`（`LLAMA_SPEC_TIMING`）。
- 脚本：`/root/p33-verify-q8.sh`（sha256）→ `p34/p39/p44`（构建+埋点验证）→ `p35-base-q8.sh`（基线）
  → `p47-mcurve.sh`（M 曲线）→ `p46/p48-*.sh`（每轮分解）。
- 日志：`/tmp/base-q8.log`、`/tmp/p47.log`、`/tmp/p46-server.log`、`/tmp/p48-server.log`。
- 二进制快照：`/root/libdir-rt`（= b11053 + C4 + C5 + PR#27858 + 本轮埋点）。

## 7. 模型卡权威参数与架构（2026-09-20 补，用户提醒"要参考模型卡建议参数"）

来源：ModelScope `unsloth/Qwen3.8-27B-GGUF` 的 `README.md`（其 `base_model: Qwen/Qwen3.8-27B`，即官方 **Qwen3.8-27B** 模型卡的
"Best Practices" 段落）。**这是采样的权威依据**。

**推荐采样参数（官方 Best Practices）**
| 模式 | temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty |
|---|---|---|---|---|---|---|
| **Thinking（默认开启）** | **1.0** | **0.95** | **20** | 0.0 | 0.0 | 1.0 |
| Instruct（非 thinking） | 0.7 | 0.80 | 20 | 0.0 | **1.5** | 1.0 |

- 官方还有 "Adequate Output Length" 建议：推理段最多 262,144 token、最终回答最多 131,072 token（1M 上下文内）。
- **我们 Phase 0 基线的采样（temp 1.0 / top_p 0.95 / top_k 20 / min_p 0.0）正好等于官方 thinking 档**——
  属于"歪打正着"，**不是**随意选的。⚠️ 但**非 thinking 档必须加 `presence_penalty=1.5`**，别漏。
- **重要对比缺口（待补）**：1cat 的生产单元在 chat template 里设了
  `{"enable_thinking": true, "preserve_thinking": true, "reasoning_effort": "xhigh"}`；
  我们的请求**没有**传这些 prompt 模板参数（`reasoning_effort` 影响推理长度 ⇒ 直接影响接受率）。
  **llama.cpp 的 server 支持在请求体里传 `chat_template_kwargs`
  （`tools/server/server-common.cpp:1330-1353`，且对 `enable_thinking`/`reasoning_effort` 有专门处理）**
  ⇒ 后续所有对标必须补上同样三个 kwarg，否则 AL/吞吐不可比。

**架构（官方数据，用于指导 kernel 工作）**
- Hidden 5120；**64 层**，布局 **16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))** ⇒ **48 层 GDN + 16 层注意力**（与早先实测一致）。
- GDN：V 48 头 / QK 16 头，head_dim 128。**Gated Attention：Q 24 头 / KV 4 头，head_dim = 256，RoPE dim 64**。
  ⇒ **长上下文 decode 的 attention 是 D=256** ⇒ 正是 1cat 的 `sm70_79t/v37`（D256）与我们 `fattn-vec` 面对的形态。
- FFN intermediate 17,408。
- **Token embedding = 248,320（padded）**，LM output 同样 248,320 ⇒ **CPU selector 的 top-k 是在 248,320 上做的**
  （比先前笔误的 151,936 大得多，23.4 ms 更易解释）。
- **MTP：trained with multiple steps** ⇒ **官方基座模型自带 MTP 头** ⇒ 我们可以在**同一个官方 target**上做
  `dflash` vs `mtp` 的干净 A/B（此前只能用不兼容的 TurboFCFusion 比 MTP）。
- 上下文：原生 262,144，可用 YaRN 扩到 1,000,000。

**另一个查证结果**：`llama.cpp/common` 里**没有任何读取 GGUF `general.sampling.*` 的代码**（grep 无匹配）
⇒ **llama.cpp 不使用量化时嵌进 GGUF 的推荐采样参数**，必须由外部（模型卡 / 请求参数）提供。
（本机快照里也没有 gguf dump 工具，故无法直接列 GGUF KV 表。）

## 8. ★对标口径修正 + `layer` vs `tensor` 的决定性结果（2026-09-20）

**口径修正**：请求体补上 1cat 生产单元用的 chat template kwargs
`{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}`（`tools/server/server-common.cpp:1330-1353` 支持）。
**结果：与不带 kwargs 逐位相同**（acceptance 0.60469、413/683 tokens 完全一致）⇒ Qwen 模板本来就默认 thinking，
**我们 Phase 0 基线已经在正确合约上**，无需回改。

**三臂对比（同一 prompt、seed 42、512 tokens、官方 thinking 采样、Q8_0 target、3 卡同 NUMA）**

| 臂 | tg | ms/token | AL | 接受率 | CPU selector | draft 前向 | target 每轮 |
|---|---|---|---|---|---|---|---|
| `tensor + dflash n=7` | 55.90 | 17.89 | 5.21 | 0.605 | **23.44 ms** | 14.79 ms | 29.4 ms |
| `tensor + mtp n=4` | 68.25 | 14.65 | 3.57 | 0.644 | — | — | 24.6 ms |
| **`layer + dflash n=7`** | **72.58** | **13.78** | 5.16 | 0.597 | **0.00 ms** | **4.30 ms** | **4.6 ms** |

**结论（待 3 prompt 复测确认）**：**`--split-mode layer` + DFlash2 = 72.58 t/s，比 `tensor` 快 +30%，也比 MTP4 快**。
机制由上表的量具直接解释：layer 模式下 selector 走**图内 GPU 路径**（CPU 侧 0.00 ms），draft 前向 14.79→4.30 ms，
target 每轮 29.4→4.6 ms —— 同时消掉了 **23.4 ms 的 CPU selector** 与 **tensor 并行逐层 all-reduce 的开销**。
⇒ **若成立，则"优化 CPU selector"这条不再必要（被绕过），最优解是改回 `--split-mode layer`。**
⚠️ 与 2026-09-20 早先"保持 `--split-mode tensor`"的结论**相反**（那次是 MTP + 重复文本 prompt + Q2_K_XL 的工况）⇒ 必须复测。

## 9. ★显存"残留"的真相与清理方法（2026-09-20，用户提示）

现象：`nvidia-smi` 显示 GPU 1–5 各占 2.4–3.1 GB，但 **`No running processes found`**。
诊断（三步，全部实测）：
1. `fuser /dev/nvidia*` ⇒ **只有 `nvidia-persistenced`(PID 2244) 持有设备节点**，没有用户进程、没有僵尸。
2. `nvidia-smi -q` ⇒ **`Reserved: 1024 MiB` 与 `Used: 3063 MiB` 是分开统计的**（干净 GPU0 同样显示 Reserved 1024）。
3. `free -g` ⇒ **`buff/cache = 70 GB`** —— 本机是 **PPC64LE，把 V100 HBM2 映射进系统内存**（见 §0.6），
   所以内核**页缓存**落在了显存里，被驱动记成 FB `Used`。

**清理（安全，无需 GPU reset）**：`sync; echo 3 > /proc/sys/vm/drop_caches`
⇒ `buff/cache` 70 GB → **1 GB**，free 78 GB → **148 GB**，**GPU 1–5 全部回到 0–1 MiB**。

**⇒ 已是硬性测量前置步骤**：每轮跑测前先 `drop_caches`。否则 3 GB/卡的页缓存会挤占模型可用的显存
（可能影响能否装满 GPU、乃至 256K 时 OOM），也可能污染"多卡/单卡"的对比。
**不要**把它误判成"泄漏的 CUDA 上下文"、也不要为此 reset GPU。
