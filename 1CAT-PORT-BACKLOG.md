# 1CAT-PORT-BACKLOG.md — 1cat-vLLM(V100/SM70) → llama.cpp 移植清单

> **范围（用户 2026-09-20 明确）**：只取 **1cat-vLLM ≥ v1.5.0** 的内容 —— **DFlash2 支持自 v1.5.0 起，v1.3.0 还没有**。
> 本地参考 checkout：`F:\vllm+llama.cpp\1cat-vllm` = `b711d53045` = **v1.5.0-672**（已含 DFlash2）。
> 交付物只落 `llama.cpp/`；用户已授权**可直接搬运代码**，但**每行必须能理解、能维护**（`llama.cpp/AGENTS.md` 硬要求）。
> 排序依据：`AUDIT-2026-09-20-dsh.md` §3（F1–F6）。**开发期用短上下文内循环判定，长上下文只做验收**（用户 2026-09-20 指示）。

---

## 0. 移植的原则（来自 1cat 自己的做法）

1. **形状/硬件专用内核 + 环境变量开关 + 可回滚**（`ExactMKernelImpl` 的 `desc.m == ExactM`；注释明说 "provide an operational rollback"）。
2. **宁可为一个具体形状写一个内核**，也不让通用表在该形状上退而求其次。
3. **按能力自动选 fast-path**（v1.5.0 的明确设计：每个 verifier 算子自查 dtype / TP 形状 / KV 格式 / 实时 batch 形状，不满足就独立回退）——
   **而不是全局一刀切开关**。这一条对 llama.cpp 的启发：分派判据要按"实际张量形状"而不是"模型名"。
4. **每一步都要有验收门**：聚合 ≥15% + 逐位一致（或 AL 不变 / ppl ≤0.1%）。

---

## 1. 1cat v1.5.0 的 DFlash2 特性清单（= 移植素材的权威来源）

来源：`1CatAI/1Cat-vLLM` `RELEASE.md`（v1.5.0 supersedes v1.3.0；126 PR / 400 commits）。

| # | 特性 | 对我们的对应物 / 落点 |
|---|---|---|
| 1 | **target 与 draft 独立的 KV dtype** | llama.cpp 目前 KV 类型全局统一（`--cache-type-k/v`）⇒ 需按 context 区分 target/draft |
| 2 | **checkpoint 自带 7-token draft + selector top-K 16** | 我们 n_max=7 已有；**selector top-K 16** 是他们固定下来的量（我们的 CPU selector 每轮对 8 位置做全词表 top-k，rank=72） |
| 3 | **5 个层边界的 hidden-state 捕获** | llama.cpp 的 dflash 路径已有 hidden 收集（`src/models/dflash.cpp`）；对齐边界数 |
| 4 | **经 Flash-V100 的非因果滑窗 draft attention** | draft 侧 attention 走 V100 专用 kernel |
| 5 | **target 与 draft 各自的 CUDA Graph** | 我们现在 draft **不在图里**（每轮 13.8–14.4 ms）；1cat 两边都入图 |
| 6 | **精确概率拒绝采样** | llama.cpp DFlash2 已有 rejection；确认与官方一致 |
| 7 | **TP4 切分的 context projection + compact LM-head selection** | 我们 selector 在 CPU 4.3 ms/轮；对应 GPU 化 |
| 8 | **FP8 E5M2 KV 上的 grouped q8 验证** | 我们 KV 是 q8_0；思路：量化 KV 上的验证核 |
| 9 | **prefix-cache 与 Mamba-align 状态恢复** | llama.cpp 有 checkpoint（`PARTIAL_ONLY`，0.027 ms/轮，已排除是瓶颈） |
| 10 | **能力驱动的自动 fast-path 选择**（非全局开关） | 见 §0.3 |

**1cat v1.5.0 的验收数字**（公开，供对标）：**17.38 ms/完整 DFlash2 轮**（且不需要整服务配成 `max_num_seqs=1`）；32K cold prefill **4,039–4,069 t/s**；64K pure prefill **3,567 t/s**；GSM8K AL request-mean **5.3888**。

---

## 2. 1cat 的 SM70 代码区（只读参考，按可迁移性排序）

| 代码区 | 内容 | 可迁移性 |
|---|---|---|
| `csrc/attention/sm70_v37/`（`bridge.cu`/`prefill.cu`/`tail.cu`） | **D=256 重写**（正是本模型 `key_length=256`） | **高** —— 本模型就是 D=256 |
| `flash-attention-v100/kernel/`（`flash_decode_paged.cu`、`fused_mha_forward_paged*.cu`、`fp8_kv_bridge.cu`、`paged_kv_utils.cuh`） | V100 FA 主体 + KV 宽加载 / 分页 | **高**（长上下文） |
| `flash-attention-v100/flash_v100_traits.cuh` | Volta HMMA fragment 归属 / traits | 高（FA/GEMM 都要） |
| `csrc/attention/sm70_grouped_long/`（`grouped-attention.cu`、`scalar-attention.cu`） | DFlash2 相关：`VLLM_FLASH_V100_DFLASH2_FIXED_INTERLEAVED`、`..._STAGE_PAGE_IDS` | 高（draft 侧） |
| `csrc/sm70_turbomind/`（vendored lmdeploy SM70 GEMM，"884"=HMMA m8n8k4） | 小 M 战术表（`CTA_M=8`、K tile 16/32/**64**、stages=2、SplitK 全开、线程组 2×4/1×4/1×2） | 中（**该算子天花板 ~1.36×，放最后**） |
| `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu` | `VLLM_SM70_DFLASH2_QPN8_RERANK`，exact-shape `m 1..8, n=62080, k=5120` —— **selector/rerank 的专用 GEMM** | 中高（selector 线） |
| `csrc/attention/sm70_79t/`（`prefill.cu`、`prefill_q8192.cu`） | 75T/79T **prefill** | 中高（prefill 线） |
| `flash_qla/ops/gated_delta_rule/chunk/sm70/csrc/gdn_forward.cu` | GDN SM70 | 低（GDN 实测只占每轮 3–6%，且该数字本身待重测） |
| `csrc/quantization/marlin/sm70_*`、`csrc/moe/marlin_moe_wna16/sm70_*` | NVFP4/MXFP4/U4/U8 量化 GEMM + MoE | 低（GGUF 量化族不同） |
| `csrc/custom_all_reduce.cuh` | push-based 自研 allreduce（+ `VLLM_SM70_TP4_PUSH_ALLREDUCE_{CONCURRENCY,SMALL_MESSAGES}`） | 中（我们 NCCL 已 +18.4%，但仍只覆盖） |

---

## 3. 第三方 llama.cpp V100 先例（**同题，优先级不低于 1cat 本体**）

来源见 `EXTERNAL-REFS.md`。**这些是"已经用 llama.cpp 在 V100 上做过"的证据**，比 vLLM 侧代码更容易移植。

| # | 来源 | 内容 | 为什么对我们重要 |
|---|---|---|---|
| **X1** | `jackinthebox52/qwen38-v100-serve` 的 **T2-001** 补丁 | 本模型 **24 Q 头 / 4 KV 头（GQA=6）**；补丁给 `fattn-vec.cuh` 加 `ncols1/ncols2` 双参数与 **`ncols2=3`** 打包（真实效果：每序列 block 数 **24 → 8**）⇒ 128K KV DRAM 流量 **26.37 → 8.59 GB/token**、no-MTP decode **16.45 → 23.83 t/s（+44.9%）** | ⚠️ **见 §4.6：只对 F16/BF16 KV 生效；必须先做 P0（切 f16 KV）** |
| **X2** | 同项目的 MTP 摊销证据 | 4-wide 验证 **30.0 µs/token** vs 单 token **102.9 µs/token** | 支持"投机验证批摊薄 KV 读取"的判断 |
| **X3** | `mistrjirka/llama.cpp` 分支 `qwen38-lossless-agent-cache` | ① **GDN 四列复用** commit（SM70、state 宽 128、每 warp 同更 4 列）；② **Volta D256 FlashAttention** commits（D256/ncols64、Q 入 shared、K/V scratch 拆分、有条件 2CTA） | ②正对 D=256；① 是 GDN 侧唯一现成优化 |
| **X4** | 上游 `PR #27955`（CUDA: fix divergent FlashAttention barrier） | FA 屏障修复 | 改 FA 前先确认是否已含 |
| **X5** | 上游 `PR #27342`（DFlash2 支持，已合并） | 我们 b11053 已含 | — |
| **X6** | 上游 `PR #27858`（tensor 路径 selector 转 CPU） | 我们**已移植**（Phase 1a） | — |

---

## 4. 移植批次（按审计后的优先级；每批独立可回退）

> 判定层：**内循环** = 单卡 `llama-bench`（~20 s）或 3 卡 harness（~10 min）；**台阶** = 8K→32K→64K；**验收** = 8K/32K/128K/256K 曲线 + 对标 1cat 真机。

> **2026-09-20 更新（三点查证后）**：新增 **P0（零代码 KV dtype A/B）**；**P2 升为"最确定的一刀"**（B3 实测 draft 每轮只做一次块前向，12.6–14 ms ⇒ 头寸 ~8×，其中约 9 ms 是 tensor 下白付的 allreduce）；
> **★★ P5（allreduce）= 2026-09-20 审计后升为第一优先**（AUDIT F8：同 build 下 tensor 与 layer 差 **16.9 ms/轮**，而 1cat 同类整桶只有 **1.294 ms**） —— 1cat 自己的 push-based TP4 allreduce 是 **default-on**、且有一整个调优家族（见 `EXTERNAL-REFS.md` §3.5①）；**P6 增加 FP16 GEMV / GDN 融合**（开关名与内核已核实）。

| 批次 | 内容 | 落点（llama.cpp b11053） | 预期 | 判定层 | 风险 |
|---|---|---|---|---|---|
| **P0** | **KV dtype A/B（零代码）**：长上下文口径 `q8_0` KV → **`f16` KV**。依据：1cat 自己的 1.3.0 验收表，**128K prefill +61% / 256K prefill +59% / 128K decode +11% / 256K decode +17%**（`EXTERNAL-REFS.md` §3.5③） | 仅配置（`--cache-type-k/v f16`） | 长上下文显著 | **✅ 已实测（2026-09-20，llama-bench，3 卡 tensor + P2P + libdir-nccl，`-r 2`）**：32K prefill **1492.27→1540.02（+3.2%）**、32K decode@depth **40.87→42.52（+4.0%）**；64K prefill **1128.98→1231.20（+9.1%）**、64K decode@depth **36.11→38.01（+5.3%）**。离散度 ±0.10~±4.19。⇒ **f16 KV 胜，且 prefill 增益随深度增长**（与 1cat 表同向）⇒ **长上下文口径改用 f16 KV**，并解锁 P1（T2-001 要求 F16/BF16 KV）。**代价**：KV 显存 ×2（256K 下 3 卡约 2.9 GB/卡，4 卡约 2.15 GB/卡） | 台阶 32K/64K ✅ → 待 256K | 低 |
| **P1** | **GQA 打包 `ncols2=3`**（X1）：让 `flash_attn_ext_vec` 支持非 2 的幂打包，消掉 3× KV 重读 | `ggml/src/ggml-cuda/fattn-vec.cuh`（+ 选核逻辑 `fattn.cu`） | 长上下文 KV 流量 ↓3×；X1 实测 **+44.9%** | 台阶（8K/32K/64K） | 中：模板/宏展开较多，需按 D=256 复核实例化 |
| **P2** | **draft 侧编排**（1cat #1/#5/#7）：① draft 入 CUDA Graph；② selector 上 GPU（top-K 16）；③ target/draft 独立 KV dtype | `common/speculative.cpp`、`common/common.cpp`、`src/models/dflash.cpp`、新增 CUDA kernel | **−9 ms/轮**（F3）+ **−4 ms/轮**（selector） | 内循环 | 高：涉及 spec 编排；先只做①（最小） |
| **P3** | **D=256 FA 路径**（X3② + 1cat `sm70_v37`）：先补 **arch 700 实例化**再改派发（此前 C1 崩溃即因 `fattn.cu:147` 的 Turing 门控派发到未实例化的 case） | `ggml/src/ggml-cuda/fattn*.cuh` | 长上下文 attention；对标 1cat 的 17.92→60.8 TFLOP/s | 台阶 + 验收 | 高：易崩，必须先补实例化 |
| **P4** | **prefill 线**（1cat `sm70_79t`）：长 prefill 是我们 TTFT 的主因（256K 370 t/s / 672 s） | `fattn` 的 prefill/MMA 路径 + 调度 | TTFT 直接改善（体验 99%） | 台阶（32K/64K） | 中高 |
| **P5** | **多设备 allreduce**：现 `allreduce.cu:402` **只支持 2 卡**；1cat 的做法是**自己的 push-based TP4 allreduce，且 default-on**（`VLLM_SM70_TP4_PUSH_ALLREDUCE`/`_CONCURRENCY`/`_SMALL_MESSAGES`/`_SUM2_M1`/TP8 分层）。目标：压掉 target 整步里的 **21.8 ms 提交窗口**与 draft 的 ~9 ms 白付 | `ggml/src/ggml-cuda/allreduce.cu`、`ggml-cuda.cu:965-1250` | 通信 | 内循环 | 中（注意与 CUDA Graph 捕获的交互，1cat 文档里就有"all-fast graph + PUSH_ALLREDUCE=1 被拒"的记录） |
| **P6** | **GDN / FP16 融合**：① GDN 四列复用（X3①）；② 1cat 的 Qwen3.8 融合开关族 —— `VLLM_SM70_QWEN38_FP16_GEMV`、`VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16`、`VLLM_SM70_QWEN38_FUSED_HC_FP16`、`VLLM_SM70_DFLASH2_FUSED_GDN_{METADATA,VERIFY}`（均已核实存在；SM70 FP16 GEMV 内核有 4 个变体可参考 `glm53_fp16_gemv_sm70.cu`） | `ggml/src/ggml-cuda/gated_delta_net.cu`（+ 可能的 GEMV 路径） | 未知（GDN 占比待 B2 重测） | 内循环 | 低-中（改动局部） |
| **P7** | **884 HMMA + Volta 自己的 MMQ 配置表**（1cat `sm70_turbomind`；我们现在被路由到 Ampere 表 `mmq.cuh:252`） | `mmq.cuh`/`mmq.cu`（+ 可选新 `*.cuh`） | **该算子 ≤1.36×**（AUDIT E5 修正后）；多日工程 | 内循环 | 高（工程量） |

### 与旧计划的差异（务必注意）
- **KV dtype（P0）与 allreduce（P5）在旧档案里被低估**：旧文把"量化 KV 更慢"归给 1cat 却没落到计划里（且归属写错，实为 anyei fork 的实测 —— 但 1cat 1.3.0 的表独立证实了同一结论）；把 push-based allreduce 列成低优先"先守住 NCCL"。
- **B3 实测把 P2 抬到第一位**：draft 每轮只做一次块前向（n_max=1/3/7 → 12.59/12.75/13.99 ms），**头寸 ~8×**，不是旧文担心的 1.5×。
- **旧计划把 HMMA（P7）列为第一优先、把 attention 列为最低** —— 已按审计**反转**：本模型是 **D=256 / GQA=6 / 16 层全注意力**，且用户真实场景是 256K 长上下文。
- **X1（GQA 打包）是本清单里"最便宜且已被同题先例验证"的一刀**，且不依赖 1cat 代码即可实现（但可用 1cat 的 D256 traits 作参考）。

---

## 4.5 ★ 调研输入（2026-09-20，子代理产出 + 主代理抽查行号）

来源：`RESEARCH-1cat-dflash2-path.md`（786 行，21 项差异清单）。**已抽查通过的行号**：
- `1cat-vllm/docs/design/sm70_dflash2_target_graph_20ms.md:21-27` 的五段基线表（逐字一致）
- `llama.cpp/src/models/dflash.cpp:502-532` 的 in-graph 批量 selector 打分路径（逐字一致）

### 4.5.1 ★★ 1cat 自己的五段分解（128K，TP4，30 轮稳态；`[文档实测]`）

| 阶段 | 基线 (2026-08-23) | 优化后 (v25/v26) | 提交量 |
|---|---:|---:|---|
| Draft 图 | 4.046 ms | 3.324 ms | — |
| **Draft -> Target 边界** | **4.934 ms** | **0.158 ms** | **153 kernels** |
| **Target 图** | **24.740 ms** | **15.272 ms** | **2612 nodes** |
| Target -> Draft 边界 | 2.838 ms | 1.140 ms | — |
| **整轮** | **36.300 ms** | **19.894 ms** | — |

⇒ **Target 占 77%**；两个边界合计从 7.77 ms 压到 1.30 ms，手段是**把 153 个 kernel 的提交串行批量化 + 把 5 次 full-attention 元数据刷新合成一次 pointer-table 启动**
（文中实测 24.424136 -> 23.678170 ms；D->T 1.857847 -> 1.213916 ms）—— **不需要新算子，只需要减少每轮提交量**。
⇒ 对照我们：`enqueue` 窗口 **21.8 ms** 而 target 整步 32-34 ms；1cat 的**整个 target 才 15.3 ms** ⇒ **怀疑我们卡在 host 提交/跨设备路径，而不是 GPU 算力**。这正是 B1/B2 量具要回答的第一个问题。

### 4.5.2 由此新增/重排的批次

| 批次 | 内容 | 依据 | 判定层 |
|---|---|---|---|
| **P-sub（新增，排在 P2 之前）** | **target 步的提交/跨设备路径**：**已由 AUDIT F8 定论：enqueue 21.5 ms 里约 16.9 ms 是跨设备集合通信**（layer 4.55 vs tensor 21.46，同 build）；若确为 host 提交瓶颈，则照 1cat 的做法**减少每轮提交量**（批量提交、把元数据刷新合成单次启动、检查 TP3 下是否每层都在做 host 介导的跨设备同步） | 4.5.1 | 内循环 + `[OP]` 表 |
| **P-tail（新增，长上下文）** | **q1..q7 target 尾部图**：1cat 在 256K 下靠它省 **4.07-6.55 ms**（`tail_graphs_20260911.md:118-123`）；llama.cpp 只按 key 缓存图（`ggml-cuda.cu`），**没有等价机制** | 子代理 §2.4 / P1-4 | 台阶 32K/64K -> 256K |
| **P2（draft 侧）** | 维持：我们 13.8-14.4 ms vs 1cat **3.32-3.76 ms**（约 4x）。注意 1cat 的 draft 有**独立的一批图**（`dflash/cudagraph.py:85-133`），且它们的 draft 不受 tensor 分片 head 的牵制 | 子代理 [1] + AUDIT F3 | 内循环 |
| **P-sem（新增，语义级）** | **精确概率比拒绝采样**：1cat 判据 `log p_t > log u + log q_d`（`rejection_sampler_utils.py:656-658`）+ `relu(p-q)` 残差；llama.cpp 是「target 自己抽的 token == draft 才接受」（`common/sampling.cpp:699-727`）。**影响 AL 与 lossless 语义，不影响固定 seed 下的吞吐归因** | 子代理 P2-3 | 正确性门 |

### 4.5.3 其它已定位但暂不做的（子代理已给行号）
- 1cat 的 selector/lm_head 两段式：**QPN8 粗支持 top-64 + 精确 FP16 重排**；`n=62080 = 248320/4`（TP4 下 target LM head 的本地词表维），`k=5120`；两条 exact-shape 分支的作用是**禁用 autotuning、钉死固定 tile**（保证数值确定性）。
- `VLLM_SM70_DFLASH2_FUSED_GDN_VERIFY` **不在**他们的默认表里（只由出货脚本开）；`FUSED_GDN_METADATA` 在默认表里。
- `VLLM_FLASH_V100_DFLASH2_FIXED_INTERLEAVED` / `_STAGE_PAGE_IDS` 的唯一消费者在 `flash-attention-v100/.../flash_decode_paged.cu:4661-4704`；`csrc/attention/sm70_grouped_long/` 里那两个谓词**定义但无调用点**。
- grouped verifier（`(num_q, 6, 256)` 共享一个 KV 页 + 6 个 GQA 头一趟）：1cat 微基准 256K 下 **2.396 ms vs 旧 XQA 6.011 ms**；llama.cpp 无此变体。

> ⚠️ 待拍板疑点（子代理标「未验证」）：出货脚本用 `fp8_e4m3`，但 grouped verify 门控硬要求 `fp8_e5m2`（`flash_attn_v100.py:5698`）⇒ 两者是否真同时生效，需 route log 实测。**暂不影响我们**（我们用 GGUF + q8_0/f16 KV）。

## 4.6 ★ T2-001 补丁的落盘结论（2026-09-20，子代理取证 + 主代理采信）

来源 `RESEARCH-gqa-packing.md`（38 KB）。**原始 diff 已拿到**：`patches/0001-t2-001-gqa-packing-sm70.patch`（16534 B，sha256 `7AF65463…`，16 hunk，**只改 `ggml/src/ggml-cuda/fattn-vec.cuh`，+71/-21**）。

**可落地性（实测）**：`git apply --check` 退出码 0；hunk 1-8 偏移 0、hunk 9-16 偏移 **-2 行**（来源是 b10793→b11053 之间该文件唯一的一处改动：`__syncwarp()` 三行合成 `ggml_cuda_syncwarp()` 一行）；本地 `fattn-vec.cuh` 去 CRLF 后与上游 b11053 **逐字节相同** ⇒ 上游 diff 可直接当基线。**不需要改 `fattn.cu`**（打包判定放在 `ggml_cuda_flash_attn_ext_vec_case` 内部）。

### ⚠️ 三条决定性约束（决定批次顺序）
1. **只对 F16/BF16 KV 生效**：补丁用 `if constexpr (packing_supported)`，而 `packing_supported` 要求 K 与 V 都是 F16/BF16。
   我们所有口径都是 `q8_0` KV ⇒ `packing_supported` **编译期为 false、整段被丢弃** ⇒ **当前配置下是 no-op（零收益零风险）**。
   第三方自己的 `serve.sh:223` 写死注释「F16 KV cache required for ... T2-001 head packing」、232-233 行确实是 `-ctk f16 -ctv f16` ⇒ 他们知道这是前置条件。
   ⇒ **正确顺序：先做 P0（q8_0 → f16 KV，零代码），再在 f16 口径上评估 T2-001。先打补丁会得到「补丁无效」的假结论。**
2. **只对单 token decode 生效**（`Q->ne[1] == 1`）。Volta 分流（`fattn.cu:644-652`）：`ne[1]*gqa_ratio_eff <= 2` → VEC（decode 命中）；`<= 16` → TILE（**DFlash2 的 8-token 验证步命中**）；否则 MMA（prefill）。
   ⇒ **对 256K prefill / TTFT 无帮助**（那仍是我们最大头寸）；带投机时只影响每轮那次单 token target decode，增益被摊薄。
3. **文档宣称 vs 实测的更正**：第三方 README 说「vec kernel 按 2 的幂打包到 ncols2=2，故剩 3x 冗余」——**错**。实测 `fattn-vec.cuh:541` 是 `launch_fattn<D, cols_per_block, 1>`，**vec kernel 里 ncols2 恒为 1（零打包）**；真实是每序列 **24 个 block / 4 个 KV 头 = 6x 冗余**，打包后 8 个 block。被消掉的因子确实 = 3（24→8），与 3.07x 流量下降自洽；但「2 的幂阶梯」描述的是 **MMA 路径**（`fattn.cu:200-222`）。

### 验证门的已知坑（照做）
- **`test-backend-ops` 覆盖不到打包路径**：`tests/test-backend-ops.cpp` 的 `nr2` 列表是 `{1,4,8,12,16,20,32}`，**没有 3 或 6**；唯一能被 3 整除的 `nr2=12` 被限制 `hsk=128`，且 V100 上 `gqa_ratio_eff=8 > 2` 会走 TILE。
  ⇒ 要覆盖得往 `nr2` 列表加 3/6，**最小复现形状 = 我们自己的几何**：`hsk=hsv=256 / nh_kv=4 / nr2=6 / nb=1 / kv=512 / F16`。
- **硬门必须在 f16 KV 下做** greedy temp0 sha256 逐位一致；q8_0 下两边 kernel 相同，是**假门**。

## 4.7 ★ push-allreduce 与 FP16 GEMV 的取证结论（2026-09-20，子代理 + 主代理采信）

来源 `RESEARCH-allreduce-fp16gemv.md`（871 行，全部结论带 `[CODE]/[MEASURED]/[CLAIM]/[DERIVED]/[UNVERIFIED]` 标签）。

### 4.7.1 push allreduce（P5 的设计依据）
- **算法**：one-shot **push-broadcast + 本地按固定 rank 序归约**（不是 allgather+reduce、不是 ring）；transport = **CUDA IPC**（`cudaIpcGetMemHandle/OpenMemHandle` + `cudaIpcMemLazyEnablePeerAccess`），要求**全互联 P2P**，否则注册直接抛异常。
- **缓冲**：每 rank 约 **2.50 MiB**（512 B epoch words + 2 epoch × 4 rank × 320 KiB slot）；哨兵 = FP16 NaN `0x7f7f`，双 epoch 交替。
- **默认全开**：`PUSH_ALLREDUCE=1 / CONCURRENCY=1 / QWEN38_BATCH=1 / SUM2_M1=1 / SMALL_MESSAGES=1`（`MTP5=0`）。
- 实测（文档记录）：[8,5120] FP16 80 KiB、128 次调用链 **push 0.850-0.856 ms vs pull 1.854 ms**（6.64 vs 14.5 us/次）；M16/M32 提升 30-40%，**M64 反而回退 30.79→34.08 us 因而被移除**。
- 子代理的推导：M8→M16 边际 +4.37 us / +245,760 B ≈ **56 GB/s** ⇒ 已贴近 NV2 单向 50 GB/s 上限；外推剩 **固定 ~2.3 us/次**（发射+poll+归约）⇒ **可攻击的是这个固定成本，不是带宽**。
- **两处对简报的纠正**：
  1. 旧文说的「all-fast graph + PUSH_ALLREDUCE=1 被拒」**不是 CUDA Graph 捕获问题，而是质量拒收**（push-off 59/60 vs push-on 58/60 GSM8K，case 20 发散）。
  2. push 路径**本身就是 graph-only**（两处派发点都要求 `cudaStreamCaptureStatusActive`）——llama.cpp 不做捕获，所以移植时**是删掉这个守卫**，不是绕过它。
- ⚠️ **本机约束**：AC922 上 TP4 跨岛**不满足全互联 P2P**（{0,1,2} 与 {3,4,5} 内 NV2、跨组 SYS）⇒ 设备 IPC 版**必须用单岛卡集**（就是我们现在用的 0,1,2）或做两级分层 push（1cat 的 TP8 变体是现成参考）。
- **llama.cpp 落地最小设计（子代理给出，P5 照此做）**：把现有 internal pipeline 从 2 卡推广到 N 卡、改用 push 协议并保持 2 卡行为逐位不变。步骤：① 抬掉 `allreduce.cu:402-406` 的 `n_devices != 2` 门；② `host_buf/host_large` 改**按目的 rank 分槽**（现在只 `POOL_SIZE*1 MiB`，TP4 需要每设备 8 MiB pinned）；③ 复用已是 N 卡形状的 arrival ring；④ 固定 rank 序 FP32 归约；⑤ 保留 2 深 slot；⑥ **不需要 CUDA Graph 守卫**。触及面 = 1 个 init 函数 + 1 个 kernel + 1 处断言（`ggml-cuda.cu:1081`）。
- **动手前的两个便宜未知数**：① 每个 decode 步**到底有几次 allreduce、字节多大**（若是 ~65 次/步 × 160 KiB，11 us/次 ≈ 0.7 ms/步）——**先量**；② host-staging 版在同尺寸下能否打赢当前最好的 NCCL 配置。

### 4.7.2 FP16 GEMV（对 P7 的输入）
- **子代理纠正框定**：`glm53_fp16_gemv_sm70.cu`（CUDA，4 变体）与 DeepSeek-V4 的 fp16 gemv（**Triton**）是**两个不同内核**，别混谈。
- CUDA 变体：V0 基础（1 CTA/行、scalar `__ldg`）；V1 half2（`ld.global.cg.u32` + float2 累加）；V2 broadcast（固定 rows=4 / 256 线程，一行载入 x 后 `__shfl_sync` 给另三行）；V3 staged（batch 循环外移 + 编译期 swizzle，smem 降到 **2 KiB 常数**）。
- **V1/V2/V3 只在 M==8 可达**；M∈1..7 一律走 V0。**有效默认是 -5（staged/swizzled/broadcast OFF）**——因为 **broadcast 被实测为负收益而关闭**（`sm70_v100_migration_control.md:46042-46045`）。
- 实测：base [M=1,N=6416,K=4096] **65.5 us vs 配对 cuBLAS 77.5 us（+15.6%）**，有效带宽下限 ~803 GB/s；保留 -5 是逐位等于 -3 且 126.222→114.924 us（+9.8%）。
- ★ **对 llama.cpp 可迁移的结构性洞见**：llama.cpp 的 `mul_mat_vec_q` **给每个 warp 一整个 token**（`mmvq.cu:866`），于是 ncols_dst=8 时**同一权重行被走 8 遍、warp 内零复用**；1cat 给每个 warp **一个 K chunk 覆盖 4 行**，权重流每 4 行只读一次。**在 weight-bound 的 V100 上这就是可搬的点**。
  ⚠️ 但注意我们已有 C5：K-quant 在 ne11=8 已被路由到 MMQ（`mmvq.cuh:5-7`）⇒ **任何 M=8 GEMV 改动必须先尊重这条**。

## 4.8 ★★ prefill 差距的取证结论（2026-09-20，子代理 + 主代理采信）

来源 `RESEARCH-prefill-gap.md`（667 行）。**它同时纠正了我们档案里的对标数字**。

### 4.8.1 标尺更正（重要）
- 我们那个「370 t/s」= **248901 token / 672.2 s**；同长度 1cat 256K 是 **107.380 s = 2438.89 tok/s**（quality-valid）⇒ **差 6.3x，不是 11x**。
- 1cat 的 **32K 4039-4069 / 64K 3567 是 NVFP4+DFlash2**；**8K 5170.96 与 256K 2438.89 是 FP8 target-only**（其文档明说不是 DFlash2 契约）⇒ 引用时必须带契约。

### 4.8.2 实测：我们的 attention kernel 差 3.1x（本会话在 GPU3 独占实测）
- 用**我们的真实形状**（hsk=hsv=256, nh=4, nr23=[6,1] = hq24/hkv4/GQA6, causal, F16 KV）跑 `test-backend-ops perf -o FLASH_ATTN_EXT`：
  kv=4096 → **19.16 TFLOPS**；kv=16384 → 19.57；kv=65536 → 19.84；kv=131072 → **19.85**。
- 1cat **79T** 同类核（Q8000/KV256000）文档实测 **61.09 causal TFLOP/s** ⇒ **差 3.1x**（对上他们的旧 Split-D 也有 2.2x）。
- 根因（源码）：`fattn.cu:644-651` 把 D=256/GQA6 的 prefill 送去 **MMA_F16**，而 `fattn-mma-f16.cuh:113-126` 的 Volta 表**只有 512/576 两行**，其余**落回 Ampere 表并带注释 `// TODO tune specifically for Volta`** ⇒ 我们用 `ncols=64, nthreads=128, occ=2, nbatch_fa=32, nstages=0`。

### 4.8.3 拆解我们的 672 s（两个真实 prompt 点拟合 + FA 实测互验）
- 拟合 `t(P)=A*P+B*P²`：**线性项 295 s（44%）**、**二次项（attention）377 s（56%）**。
- **TP allreduce 只占 ~3.2%** ⇒ 与「prefill 瓶颈是 attention 而非通信」一致。
- 另：q8_0 KV 在 MMA/TILE 路径会被强制转 F16（`fattn.cu:706-715`）⇒ 256K 下 **7776 次 O(KV) 转换** —— **这正好是 P0 改 f16 KV 的第二个理由**。

### 4.8.4 ★ 零代码可验的发现（UNVERIFIED，值得立刻验）
`-ts 1/1/1` 在 3 卡上按 `granularity_kv = 256`（= 1 个 KV 头）切，而 `n_head_kv=4` ⇒ **KV 头 1/1/2、Q 头 6/6/12** ⇒ **rank2 干 2 倍的 attention 活，有效并行度是 2x 而不是 3x**（`src/llama-model.cpp:605-847` 的切分循环）。
⇒ **换 4 卡即得 1/1/1/1 = 正好是 1cat 79T 的 Hq6/Hkv1 准入形状**；2 卡也整除。
⇒ 验证方法（零代码、10 分钟）：`llama-server -v` 加载日志，看每个 tensor 落在哪张卡；或用 `llama-bench -sm tensor -ts 1/1/1` 对比 `1/1/1/1`。

### 4.8.5 由 prefill 调研新增的批次（都在 §4 表里可执行）
| 编号 | 内容 | 成本 | 依据 |
|---|---|---|---|
| **M1** | **换 4 卡（或 2 卡）** 让 KV 头整除，消掉 rank2 的 2x 偏斜 | **零代码，10 分钟** | 4.8.4 |
| **M2** | **加大 `-ub`（512 → 2048/4096）**：attention 总量与 ub 一阶无关，但 GEMM 的 M、反量化次数、allreduce 次数都改善 | 零代码 | 4.8.3 |
| **M3** | **KV 用 f16**（P0 已实测胜出）—— 顺带消掉 7776 次 O(KV) 转换 | 零代码 | 4.8.3 |
| **M5 ★ 最推荐** | **给 `fattn-mma-f16.cuh:113-126` 补一张 Volta 的 D=256 配置表**（扫 ncols / nthreads / nbatch_fa / nbatch_K2 / nbatch_V2 / occupancy / Q_in_reg）—— **与本 fork 已成功的 C4/C5 完全同一招**，单文件、1-3 天、对 decode 无回归风险（nb=1 走 VEC 不用这张表）；量具现成（`test-backend-ops perf -o FLASH_ATTN_EXT`） | 中 | 4.8.2 |
| **M4** | **纪律**：CUDA arch 列表必须保留 `sm_70` —— 若含 75+ 而不含 70，`volta_mma_available()` 变 false，prefill 静默掉到 TILE 核 | 零成本 | 4.8.2 |

> ⚠️ 大改写（复刻 1cat 79T 分解、Volta 原生 D256 split-D）属**数周工程 + 需要新图结构**（红线：新子系统要先问用户）⇒ 暂不启动。

## 4.9 M5 的具体做法（Volta D=256 FA 配置表）—— 2026-09-20 读码得出

**落点**：`ggml/src/ggml-cuda/fattn-mma-f16.cuh` 的 `ggml_cuda_fattn_mma_get_config_volta()`（**113-126 行**）。它现在只有 `(512,512)` 与 `(576,512)` 两组，末尾是 `// TODO tune specifically for Volta` → `return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);` ⇒ 我们 D=256 用的是 **Ampere 行**。

**宏签名**（`:27`）：
```
GGML_CUDA_FATTN_MMA_CONFIG_CASE(DKQ, DV, ncols, nthreads, occupancy, nbatch_fa, nbatch_K2, nbatch_V2, nbatch_combine, nstages_target, Q_in_reg)
```
**我们当前实际拿到的（Ampere `(256,256,64)` 行，`:73`）**：`nthreads=128, occ=2, nbatch_fa=32, K2=128, V2=128, combine=128, nstages=2, Q_in_reg=true`。

**Volta 行与 Ampere 行的风格差异（可借鉴的取向）**：Volta 用 **更大的 K/V batch（256 vs 96/64）**、**更小的 combine（64 vs 128）**、**nstages=1**（Volta 无 `cp.async`）；Ampere 用 nstages=2。

**smem 硬约束（我自己算的，决定哪些旋钮能动）**：K/V 暂存 ≈ `nbatch_K2 x D x 2 B`。
- D=256 + K2=**128** ⇒ **64 KiB** ✓（Volta 上限 96 KiB）
- D=256 + K2=**256** ⇒ **128 KiB** ✗ **放不下** ⇒ **K2/V2 在 D=256 上不能照搬 Volta 的 256**，只能保持 128。
⇒ **可扫的旋钮 = { ncols 32/64 } x { nthreads 128/256 } x { occupancy 1/2 } x { nbatch_fa 32/64 } x { nbatch_combine 64/128 } x { Q_in_reg true/false }**（K2=V2=128 固定，nstages 用 1）。

**量具（现成、秒级）**：`test-backend-ops perf -o FLASH_ATTN_EXT`，用**我们的真实形状** `hsk=hsv=256, nh_kv=4, nr23=[6,1], causal, F16 KV`；基线已测：kv=4096 **19.16** / 16384 19.57 / 65536 19.84 / 131072 **19.85 TFLOPS**。
⚠️ 已知覆盖缺口（子代理实测）：`test-backend-ops.cpp` 的 `nr2` 列表是 `{1,4,8,12,16,20,32}`，**没有 3 或 6** ⇒ 需要用自己的形状直接调用或临时加形状。

**成本（2026-09-20 实测更正）**：本机 **没有 ccache**（`build-instr/CMakeCache.txt` 里 `GGML_CCACHE:BOOL=ON` 但 `GGML_CCACHE_FOUND-NOTFOUND`，PATH 里也没有 `ccache`）
⇒ **每改一次表 = 全量重编 ggml-cuda.cu**（`-j82` 下估 **10-20 分钟**），一轮 3-5 个配置 ≈ **1-1.5 小时**。
⇒ **因此 M5 要么粗扫（4-6 个候选、接受 1 小时），要么先写一个独立微基准**（把 FA 模板在**一次编译**里实例化多个配置、运行时逐个扫 —— 与前任做 WMMA 探针同一套路）。**先做粗扫，不划算再考虑微基准。**
**验收**：TFLOPS 提升 → 再按 Goal 口径跑 prefill 台阶（32K/64K）确认端到端；**对 decode 无回归风险**（nb=1 走 VEC 不用本表）。

## 4.10 ★★ 首批实机数据（2026-09-20，instrumented 构建）

**回归（量具零开销）**：instrumented 库 + env 全关 ⇒ `MEDIAN_TG=95.46`（基线 95.20）✓；`libggml-cuda` md5 `797de403…` vs `faf9cc46…`（A/B 有效）。

### 4.10.1 F8 的解读修正（重要）
`[RT] perf` 新增字段给出：`splits=2`、`enqueue_us=17.5 ms/轮`、**`sync_us=33.9 ms/轮`**（`LLAMA_ROUND_TIMING_SYNC=1`）。
⇒ **每个图只切 2 块** ⇒ 「split 爆增」假设**否定**；
⇒ 提交窗口 17.5 ms 之后**还要等 GPU 16.4 ms** ⇒ **target 步是 GPU-bound（真时间 ~34 ms，与 `target decode+sync` 的 32.3-34.3 吻合）**。
⇒ 因此 P5 的攻击点**不是 host 提交**，而是**集合通信本身占用的 GPU 时间**（下一步由 `[AR]` 计数与 `[OP]` 表定量）。

### 4.10.2 ★ 两个零代码项当场生效（32K prefill，Q8_0，f16 KV）
| 配置 | pp32768 | 相对基线 |
|---|---:|---:|
| TP3 + ub=512（基线） | 1520.45 ± 1.04 | — |
| **TP4（1/1/1/1）** | **1829.84 ± 13.06** | **+20.3%** |
| **TP4 + ub=2048** | **2190.02 ± 39.26** | **+21.6%**（vs TP4/ub512）/ **+44.0%**（vs 基线） |
⇒ **M1 与 M2 双双成立**，且都是零代码；32K 与 1cat 的差距从 2.7x 收到 **1.84x**（对 1cat 的 32K 4039 t/s，NVFP4 契约）。
⇒ decode 侧（tg8）三组在 33.7-34.2（±12 噪声）⇒ **TP4/ub2048 对 decode 无明显影响**（符合预期：decode 走 VEC/MMVQ，不受该表影响）。

### 4.10.3 ★★ `[AR]` 集合通信实测（P5 的决定性数字）
```
[AR] calls=40580  tensors=121740  bytes=21,790,310,400  avg_tensors=3.0  avg_bytes=536,972
```
⇒ **每轮 ≈ 138 次集合通信**（40580 / ~294 轮 = 2 次/层 x 64 层 + 零头），**每次 ~537 KB**，整轮轨迹累计 **21.79 GB**。
⇒ 结合 F8 的 ~17 ms/轮 跨设备开销 ⇒ **单次集合 ≈ 123 µs**；对照 1cat 的 push allreduce（80 KiB 6.64 µs / 160 KiB 11.03 µs）⇒ **约 13x 差距**。
⇒ **P5 的预期收益：把 ~17 ms/轮 压到 ~1-1.5 ms ⇒ 每轮 58.9 → ~43 ms ⇒ tg 95 → ~125 t/s（+30%）**。这是目前唯一被完整定量的数量级杠杆。

### 4.10.4 量具现状（诚实记录）
- ✅ **可用**：`LLAMA_ROUND_TIMING_[SYNC]`（splits / enqueue / sync / build / alloc / setin）、`[AR]` 集合通信计数。
- ❌ **不可用**：`GGML_CUDA_OP_TIMING` 的 **per-op event 表** —— 多设备下 event 配对给出**负数**（stop 早于 start），且开销巨大（tg 95 → **27.23**，因为每节点 2 次 event + 每图一次 device sync）。
  ⇒ 结论：**暂时放弃 per-op event 方案**；需要 per-op 证据时改用「单设备 + 关图」或 CUDA Graph 外的 host 分段计时。**但 `[AR]` 计数不受影响**（它不依赖 event）。

### 4.10.3b 遗留（原小节）
- STEP3（`[OP]`+`[AR]`）首跑**崩溃**：`CUDA error: invalid resource handle` —— 我的 per-device event 池在 `ensure()` 里**没先切设备**（事件建在 device0 却被 device1 的流使用）。**已修**（`ggml_cuda_set_device(device)`）并**增量重编中**（已装 ccache）。

## 4.11 ★ 卡数 A/B（2026-09-20 同构建重扫；主口径 = 短上下文 decode）

| 卡数 | MEDIAN_TG | AL (p1/p2/p3) | 结论 |
|---|---:|---|---|
| **TP3（0,1,2）** | **98.84** | 5.55 / 4.22 / 6.38 | **decode 最优** |
| TP4（0,1,2,3） | 70.92 | 4.19 / 3.93 / 6.08 | decode −28% |
| TP2（0,1） | 82.90 | 4.87 / 4.64 / 6.67 | decode −16% |

⇒ **分场景选卡（有数据支撑）**：
- **短上下文 decode（主验收口径）= TP3**（旧结论在 P2P+NCCL+并行 selector 之后依然成立）；
- **长上下文 prefill = TP4 + `-ub 2048`**（32K：1520 → **2190 t/s，+44%**）——用户真实场景（256K agent，TTFT 占 99%）里这条更值钱。
- 注意两侧 AL 不同（5.55 vs 4.19）⇒ 卡数会改变数值 ⇒ 跨卡数比较必须**同时报 AL 与 ms/轮**（纪律第 4 条）。

## 4.12 ★★ P5 前提被证伪（2026-09-20，2 卡零代码实验）

**动机**：`internal` 管道只支持 2 卡（`allreduce.cu:402-406`），正好用它当对照台，先判「换协议」是否值得。

| 臂（2 卡 tensor，同构建/同 harness） | MEDIAN_TG | **enqueue ms/轮** | AL |
|---|---:|---:|---|
| `GGML_CUDA_ALLREDUCE=nccl` | 83.30 | **14.8** | 4.87 |
| `GGML_CUDA_ALLREDUCE=internal` | 84.42 | **37.8** | 3.99 |
| `GGML_CUDA_ALLREDUCE=none`（butterfly） | 88.81 | **14.4** | 5.49 |

（AL 三臂不同 ⇒ tg 不可直接比；**enqueue 是集合路径的可比量**。）

⇒ **llama.cpp 的 internal 管道比 NCCL/butterfly 慢 2.5x**，根因是它的传输**经 host 内存中转**（调研原话：the internal path is a HOST-MEMORY PCIe design by design），
   而 1cat 的 push 是**设备 IPC 直写 peer**（`cudaIpcGetMemHandle/OpenMemHandle` + `cudaIpcMemLazyEnablePeerAccess`）。
⇒ **结论：P5 不能建立在「把 internal 推广到 N 卡」之上**（那条路必然更慢）。要做就必须做**设备 IPC 版 push**（= 1cat 同款传输 + 单岛卡集，因为本机 TP4 跨岛不满足全互联 P2P）。
⇒ **代价评估**：设备 IPC 版需要 IPC handle 注册 + peer access + 每 rank 分槽 + 固定 rank 序归约，属**中大型改动**（远大于子代理原先估的「1 init + 1 kernel + 1 处断言」）。
⇒ **优先级调整**：在拿到「设备 IPC push 能把单次从 123 µs 压到 ~10 µs」的更硬证据前，**先做 P4/M5（FA 配置表：收益路径清晰、单文件、量具现成）与 P2（draft，头寸 4x）**；P5 降级为「有把握再上」的大工程。

## 4.13 M5 首轮实测：Volta 风格配置在 D=256 上**变慢**（2026-09-20）

候选 #1（在 volta 函数里加两行）：`(256,256,32,128,2,64,128,128,64,1,true)` 与 `(256,256,64,128,2,64,128,128,64,1,true)`
即 nbatch_fa 32->64、nbatch_combine 128->64、nstages 2->1（模仿 Volta 的 (512,512) 行；K2/V2 保持 128 以守住 64 KiB smem）。

| 配置 | 32K prefill（TP4 + ub2048 + f16 KV） |
|---|---:|
| Ampere 回退（原状） | **2190.02 ± 39.26** |
| 候选 #1（Volta 风格 staging） | 2044.22 ± 4.44（**−6.7%**） |

⇒ **结论：Volta 的 (512,512) 行风格不能外推到 D=256**；已回退。
⇒ **成本实测更正**：改这张表会触发**几十个 template-instance .cu 的重编**（不是 60 秒，而是 **~15 分钟/候选**）—— 因为这表是 constexpr 且被大量 TU 包含。粗扫必须按 15 分钟/候选 排。
⇒ 下一步候选方向（未试）：只动 **ncols/nthreads/occupancy**（保持 K2=V2=128、combine=128、nstages=2），例如 `(256,256,64,256,2,32,128,128,128,2,true)`（nthreads 128->256）、`(256,256,64,128,1,...)`（occupancy 2->1）。

## 5. 每批的验收模板（照抄）

```sh
# 1) 同源 A/B：独立 build 目录 -> /root/libdir-<tag>/
#    md5 自检（必须不同，否则 A/B 无效）
md5sum /root/libdir-<tag>/libggml-cuda.so.0.24.0 /root/libdir-nccl/libggml-cuda.so.0.24.0
md5sum /root/libdir-<tag>/libllama-common.so.0.4.1 /root/libdir-nccl/libllama-common.so.0.4.1

# 2) 内循环（短上下文，快）
CARDS=0,1,2 SPLIT=tensor TAG=<tag> L=/root/libdir-<tag> P2P=1 bash /root/p60-ab-harness.sh
# 判据：MEDIAN_TG 提升 > 2x 离散度；AL 不劣化；greedy sha256 或 ppl 通过

# 3) 台阶（长上下文，仅 attention/KV 类需要）
#    8K -> 32K -> 64K（对标 1cat 公开 3,567-4,069 t/s prefill）

# 4) 验收：8K/32K/128K/256K 曲线 + 与 1cat 真机同口径
```

**正确性门**：优先逐位一致（greedy temp 0 的 sha256）；涉及归约/allreduce ⇒ 走 AL + `llama-perplexity`（相对变化 ≤0.1%）。

---

## 6. 待补清单（本文件已知缺口）

- [ ] 1cat v1.5.0 的 DFlash2 具体 kernel 落点（`_dflash2_sparse_topk_rejection_kernel`、`_requires_sm70_tail`）逐文件定位 —— 目前只确认了 env 与两个文件（`sm70_grouped_long`、`awq_sm70_gemm.cu`）。
- [ ] `sm70_v37`（D=256）与 llama.cpp `fattn` 的接口差异评估。
- [ ] X1 补丁的实际 diff（需 clone `jackinthebox52/qwen38-v100-serve` 的 `patches/` 目录）。
- [ ] X3 的 GDN/D256 FA 两个 commit 的实际 diff。
- [ ] 1cat `docs/design/` 下与 DFlash2 相关的设计文档清单（已有 `sm70_awq_small_n_hmma_operator.md`、`sm70_awq_exact_m5_batched_gemv.md`、`sm70_qwen38_default_fastpath.md`、`sm70_dflash2_fp32_defaults.md`）。
