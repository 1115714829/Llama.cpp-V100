# FINAL-REPORT.md — llama.cpp V100(SM70) 专项优化：交付报告

> 日期 2026-09-20。工作区 `F:\vllm+llama.cpp`。真机 AC922（6× Tesla V100-SXM2-16GB，ppc64le，CUDA 12.4）。
> **最终目标：让 llama.cpp 在 V100 上追平 1cat-vLLM 的速度与效果。**
> 本报告只写**实测过的数字**；每条都能在 `1cat-vllm-v100-study/` 的对应档案里找到原始证据。

---

## 1. 一句话结论

**我们落地了两个真实、可复现的 V100 专用改动（C4、C5）**，并把"还差什么才能追平 1cat"这个问题**用数据定了性**：
差距的**主因不在 kernel，而在投机解码的 draft**；而 kernel 与配置两条路都已接近榨干。

---

## 2. 交付物：代码改动（已核对 git diff）

```
$ git -C llama.cpp diff --stat
 ggml/src/ggml-cuda/mmvq.cu  | 35 ++++++++++++++++++++++++++++++++++-
 ggml/src/ggml-cuda/mmvq.cuh |  4 ++++
 2 files changed, 38 insertions(+), 1 deletion(-)
```
base = `ggml-org/llama.cpp` @ `1af554f8f`（tag `b11053`，`0.4.1-dev`）。**只改了这 2 个文件**（均为 CUDA 后端）。

### C4 — 给 V100 补上 MMVQ 的 nwarps 表
- `mmvq.cu`：新增 `MMVQ_PARAMETERS_VOLTA`，把 CC 700-749 从 `GENERIC` 表路由到 V100 专属表
  （5 处：enum / device 判定 / host 判定 / `calc_nwarps` / `calc_rows_per_block`）。ncols=1 时 **nwarps=2** 最优。
- **收益**：单卡 `llama-bench -p 512 -n 128`，**tg128 36.32 → 37.35 t/s（+2.8%）**（同源 A/B，见 §3.1）。

### C5 — 给 V100 补上 K-quant 的 mmvq↔mmq 交叉点
- `mmvq.cuh`：新增 `#define MMVQ_VOLTA_MAX_BATCH_SIZE_K 4`
- `mmvq.cu`：在 `ggml_cuda_should_use_mmvq` 里新增一个 `cc == GGML_CUDA_CC_VOLTA` 分支，
  Q2_K/Q3_K/Q4_K 用 4 而不是默认 8（其余类型不变）。
- **缺口本身**：上游**已为 Ada / Blackwell / DGX Spark / Jetson Orin 各调了一份**（都带 "tuned on <硬件>" 注释），
  **唯独 V100 落回未调的默认 8**。
- **收益**：见 §3.2 —— 生产模型 Q4_K_M 在 ne11=8 上 **+15.0%**。

---

## 3. 实测结果

### 3.1 C4 同源 A/B（单卡 GPU0，`-p 512 -n 128 -r 3`，Q2_K_XL）
| 变体 | pp512 | tg128 |
|---|---|---|
| pristine b11053 | 737.13 ± 15.56 | **36.32 ± 0.06** |
| b11053 + C4 | 728.16 ± 20.65 | **37.35 ± 0.09** |

**+2.8%**（pp512 在 ±16 噪声内）。与历史记录的 +3.4% 一致。方法见 §4。

### 3.2 C5 实测
**A. Q2_K_XL，单卡，扫 ne11（`-p 512 -n 32 -r 3`，`T` = 交叉点）**

| ne11（=`-ub`） | T=8（上游默认，MMVQ） | T≤6（MMQ） | 结论 |
|---|---|---|---|
| 4 | 96.71 | 95.39 | **MMVQ 好 1.2%** |
| 6 | 120.53 | 121.80 | **MMQ 好 1.1%** |
| 8 | 133.91 | **137.40** | **MMQ 好 2.6%** |

⇒ 最优交叉点 = **4**（与上游给 Ada 的 Q2_K 取值一致，但这是 V100 上独立测出来的）。

**B. 生产模型 Q4_K_M，2 卡（GPU0/1，`--tensor-split 1/1`）**

| ne11（=`-ub`） | pristine | volta2 | 差 |
|---|---|---|---|
| 4（对照，两边都走 MMVQ） | 99.58 ± 0.02 | 99.51 ± 0.02 | **−0.07%**（预期无差 ✓） |
| 8 | 127.37 ± 0.06 | **146.54 ± 0.04** | **+15.0%** |

**Q4_K_M 上 +15.0% 远大于 Q2_K 的 +2.6%** —— 印证上游注释 *"k-quants cost more to decode and
mvq redoes that per column, so MMQ wins sooner"*：位宽越高，MMVQ 每列重复解码的代价越大。
**而生产模型正是 Q4_K_M。**

**C. 机制确认（`nsys`，Q2_K_XL，`-ub 8`）**

| 变体 | 见到的 kernel |
|---|---|
| pristine | `mul_mat_vec_q` × 36，`mul_mat_q` × 0 |
| volta2 | `mul_mat_vec_q` × 33 **+ `mul_mat_q` × 3 + `mul_mat_q_stream_k_fixup` × 3** |

⇒ 确实有 3 个 matmul 从 GEMV 切到了完整 MMQ 路径（含 split-k 收尾），不是偶然差异。

**D. 默认档无回归**：`-p 512 -n 128`（ub=512 的 prefill 与 ne11=1 的 decode 都不命中新分支）
pp512 在噪声内，tg128 的 +2.6% 来自 C4。

### 3.3 L3 验收（生产模型 Q4_K_M，3 卡同 NUMA（0,1,2），256K）→ `l3-acceptance.md`
| 指标 | pristine | c4 | 结论 |
|---|---|---|---|
| 256K prefill（248901 tok） | 370.28 t/s（TTFT **672.2 s**） | 367.33 t/s（677.6 s） | **−0.8%，无差异** |
| decode（**关 MTP**，seed 42，各 3 次） | 33.97 t/s | 34.10 t/s | **+0.4%** |
| decode（**开 MTP**） | 56.58 t/s（接受率 0.608） | 39.11 t/s（接受率 0.336） | **不可归因**（差异来自内容） |

**两个重要事实**：
- **TTFT 672 s 占整个 256K 交互耗时的 ~99%** —— 谈 256K 体验，TTFT 才是主角。
- **多卡把单卡收益稀释到约 0**：C4 单卡 +2.8% → 3 卡 **+0.4%**。
  原因：`--split-mode tensor` 把每卡 GEMV 切成 1/N。**⇒ 多卡做 A/B 会掩盖单卡收益（已证实）。**

---

## 4. 测量方法（这部分比结果更重要，务必沿用）

### 4.1 同源 A/B 法（`same-source-ab.md`）
1. **两侧必须同一 build dir、同一套 CMake 参数产出**，唯一变量是待测代码。
2. **`llama-bench`/`llama-server` 只是 ~210 KB 启动壳**，真正的代码在 `libggml-cuda.so`（**~125 MB**）
   等共享库里；壳的 `DT_RUNPATH` 是**指向 build 目录的绝对路径**。
   ⇒ **只复制可执行文件做 A/B 完全无效**（两次都会加载 build 目录的库）。
   正确做法：**整套 `build/bin/` 各存一份**（`/root/libdir-{pristine,c4,volta2,...}`），用
   `LD_LIBRARY_PATH` 切换（`DT_RUNPATH` 在 `LD_LIBRARY_PATH` **之后**搜索）。
3. **有效性自检**：`md5sum` 承载改动的那份库（本项目的改动都在 CUDA 后端，所以看
   `libggml-cuda.so.0.24.0`），并按需保留**控制组**（例如 C5 的 ne11=4 就是控制组，实测无差）。
4. **注意**：本机构建**不是逐字节确定的** —— 同一份源码重编出的库 md5 可能不同
   （实测 `b4b46773` vs `9902c7c3`）。所以 **md5 相等**是强证据，**md5 不等不能单独当"变体确实不同"的证据**。

### 4.2 已知的坑（都踩过，别再踩）
| 坑 | 事实 |
|---|---|
| 「原版」二进制不是 b11053 | `/root/llm/llama.cpp/bin/*` 是 **`0.4.0-dev commit 434ddbb`**（Sep 10）；本项目 b11053 的构建在 `/root/llm/test/v100-opt/llama.cpp/build/` |
| 256K prefill 漂移 | 同配置同机器可漂 **8%**（423.89 vs 389.94 t/s）⇒ 单次测量不可信，≥2 次并报离散度 |
| 投机解码的 tg | `eval time = X ms / N tokens` 的 N 是**含投机接受**的 token 数 ⇒ tg 随**接受率**变，**不能**用来归因 kernel |
| **单 prompt 的投机结论** | 接受率随 prompt 在 **0.25~0.40** 摆动，足以让排序翻转 ⇒ 必须 **≥3 prompt + 固定 seed + 报接受率** |
| instruct 模型的请求 | 用 raw `/completion` 喂纯文本会让模型**第一个 token 就 EOS**（`stop_type:"eos"`）⇒ 必须走 `/v1/chat/completions` |
| 等服务就绪 | 要轮询 HTTP `/health`，不要 grep 日志里的 "model loaded"（本环境匹配不上） |
| 长任务 | **>10 min 的任务必须 `nohup` 到后台**（前台 ssh 会被工具超时杀掉） |
| profiler | `ncu`/`nsys` 在 `/usr/local/cuda-12.4/bin/`（**不在 PATH**）；且**都无法 profile 18 GB 的生产模型**（`failed to load model`）⇒ 换 9.14 GB 的 Q2_K_XL |
| batch 窗口 | V100 上 **`ne11≤8` 走 mmvq、9..63 走 MMQ、≥64 走 FP16 MMA** ⇒ 测 MMQ 类改动**只能用 `-b/-ub` 造批次**，`-p512/-n128` 两个都不命中 |

---

## 5. 过程中的判断更正（诚实记录）

| 早先的说法 | 更正 | 原因 |
|---|---|---|
| 「256K 下 C4 让 tg +31%」 | **作废** | 两侧 binary 不是同一上游版本（434ddbb vs b11053），且 MTP 接受率混淆 |
| 「MTP 让 tg 更慢（37 vs 43）」 | **作废**，MTP 是 **+107%**（41.33→85.81） | 跨上下文比较（256K vs 短上下文）造成的假象 |
| 「`draft-dflash` 接受率最高、最优」 | **作废** | 单 prompt 的偶然；3 prompt 均值下 dflash 0.361 vs mtp 0.328（同水平），且 **MTP 吞吐更快** |
| 「Volta 多出的 32 KiB smem 可利用（加 `I`）」 | **证伪，代码已回退** | I=160 无收益（−0.7%）、I=96 硬崩；瓶颈是**并行度**（40 block vs 80 SM） |
| 「6 卡会掩盖单卡收益」（原为推理） | **已证实** | C4 单卡 +2.8% → 3 卡 +0.4% |

---

## 6. 投机解码方向（决定后续优先级的关键一节）→ `dflash-in-llamacpp.md`、`mtp-sampler-cpu.md`

### 6.1 各类投机方式实测（ctx 4096，seed 42，**3 prompt 均值**，单卡，Q2_K_XL）
| spec | tg 均值 | 接受率均值 |
|---|---|---|
| `none`（无投机） | 34.42 | — |
| **`draft-mtp --spec-draft-n-max 4`** | **40.66（+18%）** | 0.328 |
| `draft-dflash`（用机器上现成的 1cat DFlash2 GGUF） | 37.88 | 0.361 |
| `draft-mtp` n=3 / 5 / 6 | 39.27 / 37.05 / 33.85 | 0.351 / 0.281 / 0.237 |
| `ngram-simple` | 33.74 | 0.107 |
| `ngram-map-k` / `ngram-mod` | ≈34.4（**完全没命中**） | — |
| `n-max 4 --spec-draft-p-min 0.30` | 39.17 | 0.365（接受率↑但吞吐↓） |
| `n-max 4 --spec-draft-n-min 2` | 40.66（**no-op**） | 0.328 |

### 6.2 结论
1. **生产的投机配置（`draft-mtp --spec-draft-n-max 4`）已经是本仓库可调范围内的最优** ——
   类型、n_max、p_min、n_min 全试过，**没有任何旋钮是净收益**。
2. **1cat 的 DFlash2 draft 我们本来就有**（`Qwen3.8-27B-DFlash2-GGUF/…-Q4_K_M.gguf`，1.14 GB），
   且 **llama.cpp b11053 原生支持 `--spec-type draft-dflash`**，能加载识别；但它**每步要读 1.14 GB
   独立 draft**，在带宽受限的 decode 里这点开销**大于**它多换的接受率 ⇒ **不用它**。
   （附带限制：`draft-dflash` 与 `--split-mode tensor` 冲突，`ggml-backend-meta.cpp:543` 硬崩；
   `none`/`layer` 正常。因它本就不赢 MTP，**不值得为它动代码**。）
3. n-gram 类投机在散文上无效。
4. **MTP 的 draft 采样器跑在 CPU**（`--split-mode tensor` 下 `src/llama-context.cpp:1233` 直接
   `return false`，后端采样未实现于 tensor 切分）—— 但实测 **tensor + CPU 采样（85.81）
   仍胜过 layer + GPU 采样（77.06）** ⇒ **保持 tensor，不为它动刀**（修它是大特性，上限仅约 7%）。

---

## 7. 与 1cat-vLLM 的差距：现在能说清的话

1cat 参考：**27B / 256K decode = 50.376 t/s**（DFlash2 投机解码）。
我们同模型同上下文（Q4_K_M，3 卡）**带 MTP** 的区间是 **33.7~56.6 t/s**，随接受率摆动。

**差距的主因是"投机解码的 draft 质量/开销"，不是 kernel**，理由是：
- **我们所有 kernel 改动加起来只有几个百分点**（C4 单卡 +2.8%；C5 在 ne11=8 +2.6%~+15%）；
- 而**光把 MTP 打开**（内容好时）就能把 tg 从 41 拉到 85 t/s；
- 用配置/开关无法再提升（§6.2 已穷举）。

⇒ **要真正追平 1cat，需要的是"更省或更准的 draft"这条工程线**（改 draft 结构、或降低 draft 每步读取开销、
或引入 1cat 那种为 SM70 特化的 draft 内核），**属于大工程**，不是小改动或调参。

---

## 8. 复现命令（可直接照抄）

```sh
# 服务器：ssh -o BatchMode=yes root@192.168.50.235（本机经 WSL 调用；脚本一律 scp 过去再 bash）

# 同源 A/B 骨架（唯一变量：源码）；已知的库目录：
#   /root/libdir-pristine  = b11053 原版
#   /root/libdir-c4        = +C4
#   /root/libdir-volta2    = +C4 +C5（= 当前工作树）
#   /root/libdir-final     = 当前工作树（回退 t3 后）
LD_LIBRARY_PATH=/root/libdir-volta2 /root/libdir-volta2/llama-bench \
  -m /root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf \
  -p 512 -n 128 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 -r 3

# C5 的窗口要靠 -b/-ub 造（ne11 9..63 才走 MMQ）：
LD_LIBRARY_PATH=/root/libdir-volta2 /root/libdir-volta2/llama-bench \
  -m <同上> -p 512 -n 32 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 -b 8 -ub 8 -r 3

# 编译（改 .cu 后）：Windows 写码 → scp 到 /root/llm/test/v100-opt/llama.cpp → 
#   sed -i 's/\r$//' <file>   # 清 CRLF
#   cd /root/llm/test/v100-opt/llama.cpp && source /opt/rh/gcc-toolset-12/enable \
#     && cmake --build build --config Release -j8 --target llama-server llama-bench
# 注意：.cu 无法在 WSL 编译（WSL 无 CUDA），AC922 是唯一编译关口。
```

---

## 9. 还没做（按价值排序，留给后续）
1. **投机解码的 draft 工程**（§7）—— 唯一可能真正追平 1cat 的方向，属大工程。
2. **MMQ 增加 block 数**：改 J 的查表逻辑（`ntx = ceil(ncols_max/J)`），预期边际小。
3. **`draft-dflash` × tensor 冲突**：已判定不值当，仅在将来需要 dflash 时再碰。
4. FA Volta 的小 M-tile：需要为 arch 700 **实例化**对应 MMA case（此前 C1 强行派发会崩，
   `fattn.cu:147` 那条 Turing 门控就是原因），属中等改动，且 FA 只覆盖 1/4 层。
5. GDN kernel（48 层，占比 ~1.4%）—— 优先级最低。

## 10. 档案索引（都在 `1cat-vllm-v100-study/`）
`FINAL-REPORT.md`（本文件）、`c5-volta-crossover.md`、`same-source-ab.md`、`l3-acceptance.md`、
`dflash-in-llamacpp.md`、`mtp-sampler-cpu.md`、`1cat-sm70-gemm-tactics.md`、`fa-v100-verified.md`、
`stress-test-256k.md`、`baseline.md`、`vllm-vs-1cat-vllm-diff.md`、`dflash2-llama-cpp-research.md`、
`core-changes.md`、`WORKPLAN.md`；以及全部可复现脚本（`p*.sh`、`*.sh`）。
