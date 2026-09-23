# P3-DESIGN-RECON — 1cat 79T（PR#286）架构只读侦察报告（P-P3 立项书输入）

> 任务性质：**只读侦察**。`v100-refs/` 与 `llama.cpp/` 源码全程未修改；本文件是唯一交付物。
> 纪律：所有结论带 `文件:行`；不确定处标「未验证」；禁止臆想。
> 侦察对象：`v100-refs/1cat-vllm/csrc/attention/sm70_79t/`（79T 主核）、`docs/design/sm70_flash_v100_prefill_operator_optimization.md`（优化史）、`sm70_v37/tail.cu`（Split-D 同源后裔）、我方 `llama.cpp/ggml/src/ggml-cuda/fattn-sm70-d256-kernel.cuh`、`v100-refs/flash-attention-v100/`（备料）。
>
> **一处口径更正（先读）**：任务背景称数值配方（stride-8 采样 max + 指数上限 + 中心化 value + 2 的幂缩放残差 + FP32 MMA 累加）出自 66KB 优化史文档。**经查该配方原文不在 `sm70_flash_v100_prefill_operator_optimization.md`**（全文无 stride/margin/headroom/center 相关段落，grep 见该文档 :1-1134），**原文实际在 `sm70_79t/README.md:37-51` 与 `sm70_79t/VALIDATION.md:41-72`，实现在 `sm70_79t/stable_rows.cuh`**。66KB 文档是另一条工作线（page-784 分页 prefill 算子，BM32/all-P/pair-scratch，逐位精确输出，不动数值配方），其价值是 **tile/调度演进与踩坑台账**（见 §3.3）。本报告按实际出处引用。

---

## 0. 十行摘要

1. 79T = 「QK 一次 cuBLAS 大 GEMM 物化 score 块 → PV 一次 CUTLASS 大 GEMM（FP16 操作数 / FP32 MMA 累加 / FP32 块输出）→ 每 kBlockN=24576 行 FP32 online rescale」，prefix 全行无 mask，因果尾另路批处理三角（`sm70_79t/prefill.cu:6121-6636`）。
2. score workspace 大小公式 = `rows × kBlockN × 2 B`，`rows = n_q × 6`（GQA 6 头打包进 M，每 KV 头一组）；1cat 的「约 2.4 GB」= 其 48000 行 × 24576 列 f16 = 2.36 GB（`prefill.cu:6007`）。
3. 数值守卫四件套（stride-8 采样 max + margin 4.0、exp 输入上限 10.0、value 中心化阈 0.05、value 残差 2 的幂缩放 64x 余量）+ FP32 MMA 累加 + FP32 块输出，是真模型不溢出的**必要条件**（`VALIDATION.md:41-72`；raw 配方 random 输入全过、真模型 inf）。
4. 必须件：GQA 打包大 GEMM 化、score workspace、FP32 块级 rescale、prefix/尾分离 + 尾部批处理三角 + 对角 mask + 首 64 token 精确修复、数值四守卫、Q/K 预转置、workspace 串行复用。可省件：SplitKV3、并发 tail_scores、idle-SM/wave-gated 重叠、Q8000/8192 双特化、E4M3 bridge、多请求 batch、历史变体（INT8 score / poly exp / grouped QK-PV…）。
5. **op 判断：不拆 QK/rescale/PV 三段；单 ggml 算子内部双流 + 私有 workspace**。1cat 即此形态（`prefill.cu:6121-6636` 单函数、双流四事件）；拆段会把 score workspace 变成图内张量、被多槽分配器按槽翻倍（R282 教训），并丢掉 prefix/tail 重叠。
6. 显存（TP3，KV 头 1/1/2 / Q 头 6/6/12，`1CAT-PORT-BACKLOG.md:201`）：n_q=2048、kBlockN=24576 → score ws **0.60 GB/组**（rank2 双组双缓冲 1.21 GB）；n_q=8192 → **2.42 GB/组（kBlockN=24576）或 0.81 GB（kBlockN=8192）**。
7. 16 GB 卡可行性：**n_q=2048 可行但贴边**（rank2 已负重 9.7 GB 权重 + 4.6 GB q8_0 KV + 0.5 GB f16 镜像）；**n_q=8192 基本不可行**（今天 `-b 8192 -ub 8192` 已 OOM，BASELINE-LEDGER UBLEVER 行），除非 kBlockN 降到 8192 并砍多槽。1cat 是 32 GB V100 预算（`VALIDATION.md:19-20,70-72`），常量不可照抄。
8. 高危风险：FP16 PV 分子溢出（实测 75310 > 65504，`VALIDATION.md:52`）、采样 max 漏稀疏尖峰（低估 15.31-16.00，:48-50）、KV 非 32 对齐的静默精度崩坏（+8/+16/+24 token 误差 72%/48%/24%，:236-241）、q8_0 镜像将再叠 2 份额外全 KV f16 副本、TP3 的 rank2 双 KV 头 = 2 组工作与 workspace 显存直接耦合、CUDA graph 捕获与 op 内多流互斥（须排除捕获或单流化）。
9. 数值口径必变（79T 是 FP16 score/prob + 采样 max 的**近似**路径，`README.md:46-48`）⇒ greedy sha256 门须重立一次（结构性重写配额）。
10. P-P3-T 衔接：显式 `mma.m8n8k4` PV + kDChunk 泛化正是 P-P3 的共同地基；79T 的 PV 本就走标准 CUTLASS B 碎片（`prefill.cu:505-516`）且指令原子同为 8×8×4（:470），P-P3-T 的 fragment 映射经验可直接用于 P-P3 的尾部 fine-PV。

---

## 1. 79T 数据流拆解

### 1.1 总流程（单算子入口内双流）

入口 `sm70_d256_gqa_architecture_fwd`（`prefill.cu:6640-6697`）→ `sm70_d256_gqa_half2_family_fwd`（:6121-6636）。准入硬约束：Q8000/Q8192 token、Hq6/Hkv1、D256、causal、scale=1/16、KV∈[Q,262144] 且 32 对齐（:6643-6687）。数据流：

```
Q,K,V(f16) ──transpose_half_32x32──> query_transposed[256,rows], key_transposed[256,n_kv]   (:6228-6241)
   │
   │ prefix = n_kv - n_q  (:6127)；kPrefixBlocks = ceil(prefix/kBlockN)  (:6284-6285)
   ▼
每 prefix 块 (kBlockN=24576 KV 行):
   (1) QK 大 GEMM   cublasGemmEx  → scores[kBlockN, rows] (f16, 转置布局)        (:6291-6293, 1925-1969)
   (2) 行统计       stable_row_max_partials<false> + stable_finish_max → block_max  (:6455-6465)
   (3) PV 大 GEMM   CUTLASS GEMMSoftmax→PrefixFloatPV  → prefix_numerator (FP32)  (:6295-6300, 2060-2096)
       · softmax 在 PV 的 A-operand transform 内完成（exp、行和），见 1.5
   (4) block 级 rescale  stable_merge_prefix (FP32)  (:6469-6472)
   │
   ▼  (与 prefix 并行：tail 流)
因果尾 (最后 n_q 个 KV 行, 三角):
   (5) 批处理三角 QK  cublasGemmBatchedEx (kTailTasks 个 tile-pair 任务)  (:6371-6396)
   (6) 对角 mask      mask_batched_tri_tail_diagonal  (:6400-6402, 定义 :2631)
   (7) tail 行 max    stable_row_max_partials<true>+stable_finish_max → tail_max  (:6403-6412)
   (8) fine-PV 批处理 (rounds × batched_tri_tail_pv_kernel) → tail_numerator(f16)+tail_row_sums(f32)  (:6421-6433)
   (9) 尾状态归并    finalize_round_major_tail_state  (:6497-6500)
   (10) 首 64 token 精确修复: onecat_sm70_d256_dense_state_raw(…, 64, 64, …)  (:6606-6610; repaired_rows=64×6 :6487)
   ▼
(11) 终合并 stable_merge_final: prefix/tail FP32 合并 + 恢复 value center → out(f16)  (:6612-6617)
```

- **prefix/causal-tail 分离的根据**：prefix 段 = 前 `n_kv - n_q` 个 KV 行，对所有 rows 全可见 ⇒ **完全无 mask 的规则大 GEMM**（:6288-6306）；因果性只存在于尾部三角（README.md:18-25 的 admitted shape）。
- **尾部 fine-PV 分组**：kFinePVGroupTiles=4，rounds=ceil(kTailTiles/4)（Q8192=8），每 round 一次带 `set_pv_task_base_kernel` 的批 launch（:6421-6433），round 间用 `accumulate` 标志（:6331 `round != 0`）累加进同一 tail_numerator。
- **双流重叠**：prefix_stream / tail_stream + 事件 `input_ready/prefix_pv_ready/tail_done/completion`（:6165-6166, 6450-6478, 6480-6485, 6496, 6633-6634）。显存允许时 `concurrent_tail_scores=true`，tail QK 在 prefix 第 0 块 PV 后即启动（:6450-6478）；否则全 prefix 后再跑 tail（:6480-6485）。

### 1.2 QK 大 GEMM：形状与累加精度

`CublasQKLauncher`（`prefill.cu:1925-1969`）：

| 项 | 值 | 出处 |
|---|---|---|
| 调用 | `cublasGemmEx(N, T, m=rows, n=width, k=256)` | :1964-1968 |
| A | query_transposed（`lda=rows=kRows`，转置存储） | :1965 |
| B | key_transposed（`ldb=total_kv`） | :1966 |
| C | scores（`ldc=rows`，**转置 score workspace** = [kBlockN, rows]） | :1967, :6007 |
| alpha | 0.0625（=1/16，softmax scale 前乘进 GEMM） | :1949-1959 |
| 数据/输出 | CUDA_R_16F / CUDA_R_16F | :1965-1967 |
| 累加 | 默认 `CUBLAS_COMPUTE_16F`（:1962）；`PREFIX_QK_CUBLAS_FP32_ACCUM` 时 `_32F`（:1952） | |
| 算法 | 默认 `CUBLAS_GEMM_ALGO9_TENSOR_OP`，可用 env `PREFIX_QK_CUBLAS_ALGO_RUNTIME` 覆盖 | :1937-1942 |

尾部 QK 是 **strided-batched**：每 (q_tile, k_tile)（k_tile ≤ q_tile）一个任务，`cublasGemmBatchedEx(m=kTailTileRows, n=kTailTileTokens, k=256)`，任务指针表在主机端构造后拷入（:6243-6276, 6381-6396）。任务数 = kTailTiles(kTailTiles+1)/2（Q8000=325，Q8192=528，:5948）。

### 1.3 PV 大 GEMM：形状与累加精度

`PrefixFloatPVLauncher`（`prefill.cu:2060-2096`）+ 类型配置（:460-516）：

| 项 | 值 | 出处 |
|---|---|---|
| problem | `(rows, 256, width)` — M=rows（GQA 6 打包），N=256（D），K=width（≤kBlockN） | :2068 |
| A | scores（转置布局 `PVLayoutA=ColumnMajor`，ld=rows） | :2076, :453-455 |
| B | value（RowMajor，ld=256） | :2077 |
| D | prefix_numerator **float**（`PREFIX_TORCH_PREFIX_FP32_OUTPUT`） | :2077-2078, :6008-6010 |
| **MMA 累加** | **FP32**（`PVAccumulator = float`，`PREFIX_PV_FP32_MMA_ACCUMULATE`；注释明说 Volta TC 收 FP16 操作数 + FP32 累加） | :460-467 |
| 拓扑 | TB 128×256×32、warp 64×64×32、指令原子 **8×8×4**、stages=2 | cmake:43-44, :422-439, :468-470, :505-510 |
| epilogue | `LinearCombination<float,4,…>`，accumulate 三态（首块/round 用 beta=0/1） | :2079, :2026 |

选择 128×256/64×64 拓扑的原因：FP32 输出不外溢寄存器（`README.md:44-45`）；对照实验里 32×64-warp FP32 PV 因 128 寄存器上限严重 spill 只到 54-59 TFLOPS（`VALIDATION.md:203-206`）。

尾部 fine-PV 用同一个 PVKernel 的批处理形态（`TailPVLauncher`/`batched_tri_tail_pv_kernel`，:2101-2125, 6322-6337），输出 tail_numerator（f16）+ tail_row_sums（f32 行质量，:6018-6019）。

### 1.4 score workspace：分配 / 复用 / 大小公式

结构体 `Sm70GqaHalf2Workspace`（`prefill.cu:5938-6098`），**按 device 缓存单例 + 互斥**（`get_sm70_gqa_half2_workspace`，:6102-6119；`launch_mutex`，:6129）——**跨调用常驻、整组复用**，不逐次分配。

核心公式（:5939-5958, 6005-6033）：

| 缓冲 | 形状 | dtype | 大小公式 | Q8000 实例 | Q8192 实例 |
|---|---|---|---|---|---|
| `scores`（score workspace） | [kBlockN, rows] | f16 | **rows × kBlockN × 2 B** | 48000×24576×2 = **2.36 GB** | 49152×24576×2 = **2.42 GB** |
| `tail_scores`（可选并发） | kTailTileRows×kTailTileTokens×kTailTasks | f16 | 1920×320×325×2 / 1536×256×528×2 | 399 MB | 417 MB |
| `prefix_numerator` | [rows,256] | **f32** | rows×256×4 | 49.2 MB | 50.3 MB |
| `tail_numerator` | [rows,256] | f16 | rows×256×2 | 24.6 MB | 25.2 MB |
| `prefix_sum/block_sum/block_max/prefix_max/tail_max/tail_sum` | [rows] | f32 | rows×4 | ~0.19 MB×6 | 同 |
| `tail_row_sums` | [kFinePVTasks, kTailTileRows] | f32 | 91×1920×4 / 144×1536×4 | 0.70 MB | 0.88 MB |
| `query_transposed` | [256, rows] | f16 | rows×256×2 | 24.6 MB | 25.2 MB |
| `key_transposed` | [256, n_kv≤262144] | f16 | n_kv×256×2 | 134.2 MB | 134.2 MB |
| `value_scaled`（stable 值镜像） | [n_kv,256] | f16 | n_kv×256×2 | 134.2 MB | 134.2 MB |
| `prefix_accumulator` | [rows,256] | f32 | rows×256×4 | 49.2 MB | 50.3 MB |
| `max_partials`/`tail_max_partials` | [16, rows] | f32 | 16×rows×4 | 3.1 MB | 3.1 MB |

其中 `rows = kQuery × 6`（:5940）。「约 2.4 GB score workspace」= 第一行 Q8000/Q8192 值（**用户背景里的 2.4 GB 由此而来**）。

复用策略两处关键：
- **tail_scores 显存水位探测 + 别名**：`cudaMemGetInfo` 后若 `free < tail_score_bytes + 2 GiB headroom`（或 env `PREFIX_TORCH_SERIAL_TAIL`），`tail_scores = scores`（复用主 score 缓冲，串行跑 tail）（:6064-6077；headroom 常量 :5958）。
- **static_assert 契约**：kQuery ∈ {8000,8192}、kTailTileTokens ∈ {320,256}、kTailTasks ∈ {91,144} 按块枚举校验（:6034-6039）；kBlockN 必须是 8192 的 1..16 倍（:6040-6041）——**这给了我们 kBlockN 的合法扫描区间 [8192, 131072]**。

### 1.5 rescale 频率

- **prefix：每 kBlockN=24576 KV 行 rescale 一次**——每块 PV 后一次 `stable_merge_prefix`（FP32）把块 partial 折进常驻 `prefix_accumulator/prefix_sum/prefix_max`（:6469-6472；kernel `stable_rows.cuh:140-170`：`next=max(old,block)`、两指数权重、FP32 乘加）。块 partial 复用同一个 `prefix_numerator`，无需驻留每块 partial（`prefill.cu:3-9` 头注释）。
- 对比现役 Path A：kBlockN=32 ⇒ 每 32 个 KV 行 rescale 一次（`fa-gemm-pp3.md:15`）——**rescale 降频 768x**。
- 块数（n_q=2048, n_kv=262144）：prefix=260096 ⇒ 24576×10+… ⇒ **11 次 merge**（:6284-6285）。
- tail：round 级 f32 行和（tail_row_sums）→ `finalize_round_major_tail_state`（:6497-6500）→ **终合并一次** `stable_merge_final`（:6612-6617）。

### 1.6 数值配方（原文与细节）

**原文位置**：`sm70_79t/README.md:37-51`（"The qualified integration samples score maxima at stride 8 with a fixed margin and exponent cap, centers biased values, and scales value residuals by an exact power of two with 64x headroom. PV uses FP16 Tensor Core operands with FP32 MMA accumulation, writes each 24K prefix block in FP32, and performs the online prefix/tail merge in FP32 before the value center is restored."）+ `sm70_79t/VALIDATION.md:41-72`（失效机制与四条守卫）。

四条守卫（`VALIDATION.md:54-65`）与实现：

| # | 守卫 | 常量 | 实现 |
|---|---|---|---|
| 1 | **stride-8 采样行 max + 固定 margin + 指数上限** | stride=8、margin=4.0、exp 输入 ≤10.0 | `stable_rows.cuh:8-11`；采样 `stable_row_max_partials` 每 8 列取一对 half2（:116-124）→ `stable_finish_max` 加 margin（:137）；`stable_exp` = `exp2f(fminf(v-max,10.0)*log2e)`（:26-29） |
| 2 | **value 中心化** | 前 4096 token 均值、阈 0.05 | `stable_value_center`（:53-63）；终合并恢复 center（`stable_merge_final` :198） |
| 3 | **value 残差 2 的幂精确缩放 + 64x 余量** | headroom=64 | `stable_value_amax`（:65-82）+ `stable_value_scale`（`frexpf`→`ldexpf(1, exp-(mant==0.5))×64`，:46-51）+ `stable_scale_values`（:84-94）——**2 的幂 ⇒ 缩放无舍入误差** |
| 4 | **FP32 MMA 累加 + FP32 块输出 + FP32 合并** | — | cmake:28-29（`PREFIX_PV_FP32_MMA_ACCUMULATE` + `PREFIX_TORCH_PREFIX_FP32_OUTPUT`）；`stable_merge_prefix/final` 全 FP32（`stable_rows.cuh:140-199`） |

边界语义：`stable_exp` 用于 tail/受控路径；prefix transform 内 `exp2f((value-row_max)*kLog2E)`（`prefill.cu:1191-1196`）依赖 row_max 已含 margin 与上限（clamp 分支 :1218-1226 [该分支逐条未验证]）。`stable_merge_final` 对修复行（repaired_rows）用 `ts*tail_sum` 特殊系数（`stable_rows.cuh:190`）。

---

## 2. llama.cpp 复刻的最小结构件清单 + op 拆分判断

### 2.1 必须件（缺一即不成立 / 变成另一条路线）

| # | 结构件 | 作用 | 依据 |
|---|---|---|---|
| M1 | **GQA 6 头打包进 M**（rows = n_q×6，每 KV 头一条 GEMM 链） | QK/PV 变大 GEMM 的 M 维来源；对 rank 内 KV 头 1..2 个各跑一组 | `prefill.cu:5940, 6121-6636`；我方 TP3 每组恰为 Hq6/Hkv1（`1CAT-PORT-BACKLOG.md:201-202`） |
| M2 | **score workspace**（rows × kBlockN × 2 B，转置布局） | 物化 score 块供 PV 消费；也是采样 max 的输入 | `prefill.cu:6007, 6291-6300` |
| M3 | **QK 大 GEMM（cuBLAS）** | QK 单次大 GEMM | :1925-1969 |
| M4 | **PV 大 GEMM（CUTLASS 或显式 mma）+ FP32 MMA 累加 + FP32 块输出** | 数值配方第 4 条；块输出 FP32 是 24K 块不溢出的前提 | :2060-2096, :460-467; `VALIDATION.md:63-65` |
| M5 | **块级 FP32 online rescale + 块行统计（max/sum）** | rescale 降频的本体 | `stable_rows.cuh:98-170`; `prefill.cu:6455-6472` |
| M6 | **prefix/因果尾分离 + 批处理三角尾 QK + 对角 mask + 首块精确修复** | prefix 才能无 mask 跑规则 GEMM；尾部精度靠 repair 兜底 | :6284-6306, 6371-6434, 6400-6402, 6606-6610 |
| M7 | **数值四守卫**（采样 max+margin / exp cap / value 中心化 / 2 幂残差缩放） | **真模型必要条件**：raw 配方 random 测试全过、真模型 inf（`README.md:37-40`；`VALIDATION.md:41-52`） | `stable_rows.cuh:8-94` |
| M8 | **Q/K 预转置（或等价 GEMM 布局适配）** | cuBLAS N/T 形状 + 转置 score 的连贯读取依赖它 | `prefill.cu:6228-6241`；`stable_rows.cuh:96-98` 注释 |
| M9 | **workspace 按 device 常驻缓存 + 串行互斥 + 显存水位降级** | 2.4 GB 级缓冲必须复用；并发/串行 tail 自适应 | :6064-6077, 6102-6132 |
| M10 | **KV 32 对齐准入 + 不合规形状 fallback** | 32 对齐是 PV K tile 正确性边界（非 32 对齐精度崩坏，`VALIDATION.md:236-241`）；llama.cpp 的 kv_len 任意 ⇒ 要么自建尾块补边、要么 fallback 现核 | `prefill.cu:6651-6655`; `VALIDATION.md:236-241` |

### 2.2 可省件（首版不做，或降级）

| # | 项 | 理由 / 依据 |
|---|---|---|
| S1 | **SplitKV3** | 现役 Path A 已有可选 SplitKV3（`fattn-sm70-d256-kernel.cuh`/`.cu:767-783`）；79T 主线不依赖它（其收益在小 n_q/decode），P-P3 首版按整 ubatch 大 GEMM 即可 |
| S2 | **并发 tail_scores 缓冲** | 1cat 自己显存不足就别名 scores（:6064-6077）；16 GB 下直接取串行版 |
| S3 | **idle-SM/wave-gated fine-PV 重叠**（`PREFIX_TAIL_*` 调度族） | 复杂 round/task-base 调度（:6421-6433）可先顺序 launch；fine-PV 分组本身保留（开销小） |
| S4 | **Q8000/Q8192 双特化 + leading-pad 机制** | `prefill_q8192.cu:1-12` 的宏特化套路可留作参考，但按我们 n_q 直接定 tail tile 即可 |
| S5 | **legacy_tail_adapter（state_max/state_sum、unnormalized 契约）** | `legacy_tail_adapter.cu:15-23` 是 vLLM state 接口的历史适配，llama.cpp 无此契约 |
| S6 | **E4M3 bridge / paged-KV gather workspace** | 我方 KV 是 q8_0→to_fp16 镜像（`fattn-common.cuh:1026-1086`），已有现成槽；无 paged 布局 |
| S7 | **多请求 batch 入口**（多序列 gather） | llama.cpp ubatch 单序列 prefill |
| S8 | **prefill.cu 的历史变体路径**（INT8 score、poly/async exp、grouped QK-PV、warp-specialized PV、superblock、phase-batched materialization…） | 全部 `#if` 考古层（:111-179 的组合约束表可见一斑）；只搬 cmake:26-44 声明的**生效配方** |
| S9 | **exact-tail debug 通道**（`PREFIX_TORCH_EXACT_TAIL/DUMP_TAIL`） | 保留思想（debug 后门）即可，不必逐行搬 |

### 2.3 FA 节点是否拆成 QK/rescale/PV 三段 op？—— **判断：不拆；单 op 内多流 + 私有 workspace**

依据：

1. **1cat 就是单算子形态且已生产验证**：`sm70_d256_gqa_architecture_fwd` 单入口（`prefill.cu:6640-6697`），内部一个函数完成 QK→统计→PV→rescale→tail→merge，prefix/tail 双流 + 4 个 event 同步（:6121-6132, 6165-6166, 6435, 6474-6478, 6633-6634）。语义上无需图层可见的中间张量。
2. **拆段会把 score workspace 变成图内张量**：2.4 GB 级 [kBlockN, rows] 张量一旦进 ggml 图，会被 gallocr 记账、并在 T8 多槽下**按槽复制**。R282 已经「8 槽 × 每形状 ~1 GB 计算缓冲」在 256K 逐级 OOM（BASELINE-LEDGER R282 行）；再叠 2.4 GB 图张量不可承受。op 私有缓存 workspace 全程一份（`prefill.cu:6102-6119` 同构做法）。
3. **拆段丢掉 prefix/tail 重叠与 merge 融合**：concurrent_tail_scores 让 tail QK 与 prefix 首块重叠（:6450-6478）；段间图边界会强制全序，且给 meta 主机层再加两次调度（本项目 meta 税是主账目，PLAN-GRAPH §METATAX）。
4. **CUDA Graph 的反向约束（重要，见 §5-R11）**：op 内多流与整图捕获天然互斥，无论拆不拆都存在；拆段并不能规避（三段串行单流倒可以捕获——但那正是第 3 条要放弃的东西）。结论：**宁可单 op + 捕获排除/单流降级**。
5. **何时才需要拆**：若将来要在图层让 QK 与其他 GEMM 共流合批、或做跨算子融合（如与 LM-head 重叠）才拆。当前无此需求。
6. 折中说明：「新图 op」一词仍部分成立——需要在 ggml 新增算子（或给 FLASH_ATTN_EXT 增加后端路由位）+ fattn.cu 分派钩子（Path A 已有先例：`fattn-sm70-d256.cu:431-480` 的 `BEST_FATTN_KERNEL_SM70_D256` 分派）。**不需要** llama-graph 层三段拆分。

---

## 3. 可直接借用的代码结构清单 + 「绝不该抄的常量」清单

### 3.1 可借用（文件:行 → 借什么）

| 出处 | 借什么 |
|---|---|
| `1cat-vllm/cmake/sm70_79t.cmake:26-44` | **生效配方唯一真源**（README.md:69 明示）：宏组合 = STABLE_ROWS + FP32_MMA_ACCUMULATE + PREFIX_FP32_OUTPUT + BLOCK_N=24576 + 预转置 + 转置 score + raw cuBLAS QK + batched tri tail + fine-PV + repair + fused prefix sum |
| `sm70_79t/prefill.cu:5938-6098` | workspace 结构与**大小公式**、static_assert 契约（kBlockN 扫描区间 8192..131072 :6040-6041） |
| `prefill.cu:6064-6077` | 显存水位探测 + tail_scores 别名降级（16 GB 直接抄这一段的语义） |
| `prefill.cu:6102-6132` | per-device workspace 缓存 + launch_mutex（生命周期/复用模式） |
| `prefill.cu:6121-6241` | 双流四事件的启动编排、memset/symbol 设置、Q/K 转置 launch |
| `prefill.cu:6243-6276` | 三角任务指针表构造（batched GEMM 的 q/k/score 指针算术） |
| `prefill.cu:6284-6306` | prefix 块循环构造（kPrefixBlocks、首块 accumulate 语义） |
| `prefill.cu:6371-6436` | `launch_approximate_tail`：批 QK→对角 mask→采样 max→fine-PV rounds→事件 |
| `prefill.cu:6448-6485` | 主循环：QK→行 max→PV→merge 的交错与 tail 重叠窗口 |
| `prefill.cu:6487-6500, 6606-6610` | 尾状态归并 + **首 64 token 精确修复**（近似尾的精度兜底结构） |
| `prefill.cu:6612-6631` | 终合并两种形态（stable_merge_final / merge_float_prefix_direct_round_major_tail） |
| `prefill.cu:1925-1969` | CublasQKLauncher（alpha=scale、算法 env 覆盖、16F/32F 双累加开关） |
| `prefill.cu:2060-2096` | PrefixFloatPVLauncher（FP32 输出 CUTLASS GEMM 参数化） |
| `prefill.cu:460-510` | PV 累加器/输出类型选择、PVInstructionShape 8×8×4、转置 A 布局 |
| `prefill.cu:955-1351` + `sm70_79t/include/cutlass/gemm/threadblock/mma_pipelined.h` | **MmaPipelined79T** 的 `set_valid()/finalize()` transform 钩子——softmax 融进 PV 的 A-变换（README.md:53-57）；行和归约（warp_row_sum）多种拓扑（:1320-1560） |
| `sm70_79t/stable_rows.cuh:8-94` | 数值四守卫常量与 kernel（采样 max/margin/cap/中心化/2 幂缩放） |
| `stable_rows.cuh:98-170` | 两段式采样行 max + 块级 FP32 rescale merge |
| `stable_rows.cuh:172-199` | 终合并（prefix/tail 权重、value center 恢复、修复行系数） |
| `prefill.cu:2631 + 320-324` | 对角 mask kernel + repair-token 常量位 |
| `sm70_79t/VALIDATION.md:41-72, 186-218` | 失效机制 + 被否变体清单（§5 风险表的证据源） |
| `sm70_v37/gemm_with_softmax.h`（被 `prefill.cu:79` include） | GemmSoftmax QK（GEMM 内融合行统计） |
| `flash-attention-v100/include/mma_m8n8k4.h:1-80` | 显式 `mma.sync.m8n8k4` 封装的 fragment 契约（A/B 每线程 4 half 打包 2×u32、C/D 8 float、wmma.load 不支持 sm70 m8n8k4 ⇒ 手写 lane 映射、f32 累加）——**P-P3-T 显式 mma 的规格书** |
| `flash-attention-v100/include/gemm_smem.h:14-60` | smem 分 stage 加载模板（uint4 粒度、XOR swizzle 清零、DUAL_LOAD 双缓冲形态）——大 tile/QK pipeline 备料 |
| `flash-attention-v100/include/swizzle.h`、`utils/docs/volta.md` | Volta smem swizzle 与 HMMA.884 拓扑笔记（备料） |

### 3.2 绝不该抄的常量（形状不同 / 预算不同，盲抄必错；X16 教训）

| 常量 | 值与出处 | 为什么不能抄 |
|---|---|---|
| `PREFIX_TORCH_QUERY_TOKENS` / kRows=kQuery×6 | 8000/8192（`prefill.cu:72-74`, `prefill_q8192.cu:7`）| 我们 n_q=2048（-ub 2048 测量形状）；tail tile 320/256、kTailTasks 91/144、static_assert（:6034-6039）全是 Q8000/Q8192 专属枚举 |
| `kBlockN=24576` | cmake:30 | 它是 **32 GB V100 预算**下调出的值（`VALIDATION.md:70-72` 明说 24K 为塞进 32-GiB 预算）；16 GB 必须按 §4 重算（合法区间 8192..131072） |
| `PV_TB 128×256 / warp 64×64` | cmake:43-44 | 为 rows≈48000 的大 M + FP32 输出寄存器不溢出实测选出（`README.md:44-45`）；反例在案：32×64 warp 版 spill 到 54-59 TFLOPS（`VALIDATION.md:203-206`）。我们 M 更小且 P-P3-T 是显式 mma 形态，拓扑须重测 |
| `QK_TB 128×128 / warp 32×64 / stages 2` | cmake:42 | 只影响备用 CUTLASS QK 路径（生效路线是 cuBLAS） |
| `PREFIX_BATCHED_TAIL_QK_ALGO=ALGO10_TENSOR_OP` | cmake:35 | cuBLAS 算法选择依赖形状/驱动/时钟（尾 QK 默认值 :277 还是 ALGO9）；只能实测（:1938-1942 有运行时覆盖钩子） |
| `kStable*` 数值常量（margin 4.0 / cap 10.0 / headroom 64 / center 阈 0.05 / 采样窗口 4096） | `stable_rows.cuh:8-12, :57` | 自我声明是 **model-qualified bound, not a proof**（`VALIDATION.md:67-68`），且标定对象是 E4M3 KV + 其模型 + FP16 值域；我们 q8_0 KV（量化噪声不同）必须重新资格验证。已知敏感：margin 4→6 使三例相对 L2 从 1.08/1.47/0.62% 恶化到 3.91/2.93/1.25%（:215-218）；headroom 4x/16x 分别在 16K/152K 翻车（:213-214） |
| `kTotalKVAlignment=32`、`kMaxTotalKV=262144` | `prefill.cu:6650-6655` | **32 是必须抄的"约束"而非可调常量**（非对齐静默错值，`VALIDATION.md:236-241`）；262144 上限可以放宽但须重验 |
| `kTailAllocationHeadroom=2 GiB` | `prefill.cu:5958` | 32 GB 卡的水位余量；16 GB 须重算（甚至首版直接强制串行 tail） |
| `PREFIX_BATCHED_TRI_REPAIR_TOKENS=64` | `prefill.cu:321-322` | 与其 320-token 首 tail tile 配套；我们尾 tile 划分不同，repair 宽度要按误差实测重定 |
| value center 采样前 4096 token | `stable_rows.cuh:57` | 采样窗口与其 KV 类型/模型绑定 |
| 半精度 softmax scale=1/16 硬校验 | `prefill.cu:6686-6687` | 我们 D=256 恰好也是 1/16，但该校验写死，泛化时别照抄写法 |

### 3.3 66KB 优化史文档（page-784 线）的可借教训（tile/常量演进 + 踩坑）

（注意：这条线是**另一架构**（分页 BM32、逐位精确），借"坑"不借"常量"。）

- tile/常量演进主链：sO=268 布局（+8.67%）→ early sP store（-2.6%）→ 跨块 steady/drain 流水（-2.4%，局部内存 16.04M→3.58M）→ Q/P 紧凑 swizzle（-12.3%）→ BM32 相位复用（-18~20%）→ all-P（-3.7%）→ pair-slab scratch（-5.1%）（该文档 :27-43, :95-124, :416-468, :516-560, :580-650）。
- **踩坑台账（与我们相关的）**：
  - 寄存器硬预算：512 线程双 CTA ⇒ 64 reg/线程；4 个常驻 PV 累加器就吃 32 个 FP32 寄存器，任何流水加深都 spill（:307-348）——**大 tile/dual-CTA 拓扑必须先过寄存器门再谈速度**（P-P3-T 的 Wide 1 CTA/SM 风险同源）。
  - raw-HMMA 手拼 M8 拆原生 M16 = 反而 +18~27%（:259-271）——**对手拼 m8n8k4 的警示**：拆原生操作数会加串行链/重读 B。这与 P-P3-T 用显式 mma 不冲突（Volta 无别的选择），但提示大 GEMM 段尽量用原生 WMMA/CUTLASS（79T 正是这么做的）。
  - half-lane/shuffle/逐 lane 向量 B 载入全负（:243-257, 118-121）——P-P3-T 的 B 碎片设计的"别做"清单。
  - M64 FA2 式大 tile 反而 -2%：1 CTA/SM 丢延迟隐藏（:901-940）——**大 tile ≠ 自动快**，占用率权衡必须实测（与我们 kBlockM=64 现状直接对话）。
  - CTA-wave 量化：192 CTA vs 144/波 ⇒ 尾波吃掉 1/3 吞吐（:1020-1037）——启动形状按波对齐思路可借。
  - 硬件校准：他们机器是 72-SM PG503，实测 FP16 TC 上限 93.4 TF/s（降频后），**125 TF/s 不可持续**（:1066-1091）——引用 TFLOPS 时的口径警示。
  - 纪律条目：「无 NCU 瓶颈证据不得开新流水」（:479-484, 707-712）可直接并入我们 A/B 纪律。

---

## 4. 显存账（TP3 / 16 GB V100）

### 4.1 每卡形状（依据 `1CAT-PORT-BACKLOG.md:201-202`，切分循环 `src/llama-model.cpp:605-847`；该文标 UNVERIFIED，我方未另行实机核对 ⇒ 标**未验证**）

| rank | KV 头 | Q 头 | GQA 组数 | rows/组 = n_q×6 |
|---|---|---|---|---|
| 0 | 1 | 6 | 1 | 6·n_q |
| 1 | 1 | 6 | 1 | 6·n_q |
| 2（最重） | 2 | 12 | 2 | 6·n_q ×2 组 |

### 4.2 score workspace 峰值（公式：rows × kBlockN × 2 B；rank2 取"组间共享单缓冲"与"双缓冲"两值）

| n_q | rows/组 | kBlockN=24576 | kBlockN=8192（最小合规） | rank2（24576，单/双缓冲） |
|---|---|---|---|---|
| **2048**（现役 -ub 2048 测量形状） | 12288 | **0.60 GB（576 MiB）** | 0.20 GB（192 MiB） | 0.60 / 1.21 GB |
| **8192**（1cat 同级大 chunk） | 49152 | **2.42 GB（2.25 GiB）** | 0.81 GB（0.75 GiB） | 2.42 / 4.83 GB |

辅助缓冲（每组，rows=6·n_q；n_q=2048 / 8192）：prefix_accumulator f32 12.6/50.3 MB、prefix_numerator f32 12.6/50.3 MB、tail_numerator f16 6.3/25.2 MB、query_transposed f16 6.3/25.2 MB、max_partials f32 0.8/3.1 MB；每 KV 头全 KV：key_transposed f16 = n_kv×256×2 = 134 MB@262144、value_scaled f16 = 134 MB@262144（若启用数值守卫）；tail_scores 并发缓冲（**建议不做，别名 scores**）：n_q=2048→28 MB、8192→417 MB。

### 4.3 16 GB 预算表（rank2 最重卡；标注哪些是实测锚、哪些是估算）

| 项 | 大小 | 来源 |
|---|---|---|
| 权重（Q8_0 27 GB 模型 TP3） | **9.68 GB** | BASELINE-LEDGER R198「权重/卡 9.68 -> 7.25 GB」（实测锚） |
| q8_0 KV cache @262144（16 全注意力层 × K+V × 2 KV 头 × 256 × 262144 × 1.0625 B） | **≈4.56 GB** | 公式估算；层数 16 来自 R264/`VALIDATION.md:21`；q8_0 块 34B/32 元素；**未在本机核算** |
| to_fp16 镜像（现役 Path A，每调用 scratch，2 KV 头全 KV K+V） | **≈0.54 GB** | `fattn-common.cuh:1026-1086`、`fattn-sm70-d256.cu:119-157`（公式实锤，量为估算） |
| 计算缓冲（T8 多槽；256K 已知须限槽） | ≈3 GB（3 槽）或 ~1 GB（单槽） | R282「8 槽 × 每形状 ~1 GB 逐级 OOM，SLOTS=3 放行」（量级锚） |
| GDN 状态 / CUDA graph / 杂项 | ~0.5 GB（估） | 未验证 |
| **基线小计** | **≈18.2 GB（3 槽）/ ≈16.2 GB（单槽）** | — |
| **+ P-P3 score workspace（n_q=2048, kBlockN=24576, 单缓冲）** | **+0.60 GB** | §4.2 |
| **+ 同上（双缓冲 rank2）** | **+1.21 GB** | §4.2 |
| **+ P-P3 score workspace（n_q=8192, kBlockN=24576）** | **+2.42 GB** | §4.2 |

**结论（可行性）**：
1. **n_q=2048 + kBlockN=24576 + 组间共享单缓冲 + 强制串行 tail + 限槽 1~2：可行但贴边。** 关键锚点：R272 已实测 TP3 + 256K + q8_0 KV 全量预填充跑通（BASELINE-LEDGER R272，当时为单槽分配器），说明"权重 9.7 + KV 4.6 + 镜像 0.5 + 计算 ~1"≈ 15.8 GB 是现实水位；加 0.6 GB score workspace 需要以限槽/收紧计算缓冲换取。
2. **n_q=8192 + kBlockN=24576：不可行**（2.42 GB/组，rank2 更甚）。独立证据：今天 `-b 8192 -ub 8192` 就已 abort 显存不足（BASELINE-LEDGER UBLEVER 行）——n_q=8192 尚未加 workspace 就放不下。
3. **n_q=8192 + kBlockN=8192（0.81 GB/组）**：数值上可争，但需先解决 ub/b 钳制与显存，属二期；且 rescale 频率升 3×（每 8192 行一次），性能代价约 -2~3%（`VALIDATION.md:207-209`：8K 块 73.39 vs 24K 75+ TFLOPS）。
4. **降档开关**（借鉴 `prefill.cu:6064-6077`）：按 `cudaMemGetInfo` 自动在 {24576 / 16384 / 8192} 三档选 kBlockN 并决定 tail 并发与否，是 16 GB 的必备设计（1cat 的 2 GiB headroom 常量须换成我们自己的水位公式）。
5. ⚠️ 1cat 的 24K 块是按 **32 GB** V100 预算定的（`VALIDATION.md:19-20, 70-72`）；其整套 workspace（score 2.36 GB + tail_scores 0.40 + value_scaled 0.13 + … ≈ 3.2 GB，§1.4）在 16 GB TP3 rank2 上**照抄即 OOM**。

---

## 5. 风险表

| # | 风险 | 机制 | 1cat 证据（实锤） | 我方暴露 | 缓解 |
|---|---|---|---|---|---|
| R1 | **FP16 PV 分子溢出** | 未缩放的 PV partial 超 f16 有限值 65504 | 实测最大 **75310**，输出 inf、模型吐 token -1（`VALIDATION.md:45-52`） | 同构必现（我们 D=256 同形状 | M4/M7 必装：FP32 MMA 累加 + FP32 块输出 + value 缩放 |
| R2 | **采样 max 漏稀疏尖峰** | stride-8 采样低估行 max | 被漏尖峰比采样值高 **15.31-16.00**（:48-50）；margin 4→6 误差反升（:215-218） | q8_0 KV 噪声下尖峰分布未知 | margin 重新标定；对照"精确行 max"变体（其代价 54-59 TFLOPS，:203-206）做精度/速度 A/B |
| R3 | **零平移 exp / FP16 分子的老配方** | random 输入测试**全部通过**，真模型溢出 | `README.md:37-40`；raw 配方 81.5/80.5 TFLOPS 但无效输出（:201-202） | 同 | 测试门必须含真模型冷请求 + FP32 oracle 采样行对照（`VALIDATION.md:191-200` 的三类回归可抄） |
| R4 | **value 偏置/幅度未知 ⇒ 余量常量失配** | headroom 4x/16x 分别 16K/152K 翻车 | `VALIDATION.md:213-214`；64x 自称 "model-qualified bound, not a proof"（:67-68） | 我们模型 + q8_0 KV 值域未测 | 上线前抓真张量做 amax 标定；headroom 做成可调参数 |
| R5 | **causal mask 处理边界** | 79T 依赖 prefix = total_kv - n_q 的 bottom-right 对齐切分（`prefill.cu:6127, 6676`），prefix 段零 mask | README admitted shape（`README.md:18-25`） | llama.cpp FLASH_ATTN_EXT 的 mask 是任意的（`fattn-sm70-d256.cu:564` 用 mask->ne[0] 取真实 KV）；混合 mask/滑窗不适用 | 准入门写死"整 chunk 自回归 + bottom-right"；其余形状 fallback 现核（P-P5 再扩） |
| R6 | **KV 非 32 对齐静默错值** | PV K tile=32 读残块错误 | KV=Q+8/16/24 相对 L2 误差 **72%/48%/24%**，Q+32 才回 0.034%（`VALIDATION.md:236-241`） | llama.cpp kv_len 任意 | M10：32 对齐准入 + 不满足走 fallback（或自建补边，须重新数值资格验证） |
| R7 | **q8_0 KV / to_fp16 镜像交互** | 现役每个 ubatch 对整条 KV 做 to_fp16（O(n_kv)×ub 数） | 我方 `fattn-common.cuh:1026-1086`、`fattn-sm70-d256.cu:119-157`；R263 已证它不是速度墙 | 79T 另需 **key_transposed + value_scaled 两份额外全 KV f16 副本**（`prefill.cu:6006, 6081`）⇒ 镜像×3，rank2@262144 约 0.8 GB | 把"转置 + 中心化 + 2 幂缩放"融合进反量化 kernel（省 1-2 份）[设计选项，未验证]；中心/缩放必须在反量化**后**做（per-block scale 语义） |
| R8 | **TP 切分 × GQA 打包耦合** | TP3 KV 头 1/1/2 / Q 头 6/6/12（`1CAT-PORT-BACKLOG.md:201`） | 每组恰为 Hq6/Hkv1 ⇒ 与 79T 准入形状逐字相同（:202「4 卡 1/1/1/1 更佳」） | rank2 双组 = 2×工作 + 2×workspace 需求；有效并行仅 2x（:201） | workspace 组间共享单缓冲（串行）首版；P-P0/M1（换 4 卡或 2 卡）与本项**同账**，建议一起拍板 |
| R9 | **数值口径必变 ⇒ greedy sha256 门必破** | 79T 是近似路径（score/prob f16 + 采样 max） | `README.md:46-48` 明言 approximate | 我们正确性门 = greedy sha256（红线） | 结构重写允许重立门一次（`fa-gemm-pp3.md:66` 口径）；立门时配 FP32 oracle 采样行对照 + 真模型问答检查（抄 `VALIDATION.md:102-113` 的 token 序列法） |
| R10 | **拓扑/寄存器反例在案** | 32×64-warp FP32 PV spill ⇒ 54-59 TFLOPS；half2 Taylor exp ⇒ 67.31 TFLOPS 且误差升 | `VALIDATION.md:203-212` | 我们的 M 维更小（n_q=2048），最优拓扑未知 | 禁抄这两形态；拓扑按我们形状重扫（P-P4 范畴） |
| R11 | **CUDA graph 捕获 × op 内多流** | 图捕获只捕获调用流；侧流 launch 在捕获期非法/丢失 | 1cat 生产跑在 FULL_AND_PIECEWISE 分段图下（`VALIDATION.md:284-296`）——**分段图按字面即在 attention 处断开**；其 op 内双流在图外执行 | 我方 ggml-cuda 会对 compute 图捕获；该 op 若含侧流必须**排除捕获**（直发）或捕获时单流化（丢重叠） | 设计时二选一并实测两条路的墙钟；**未验证**我们图捕获栈对"排除单算子"的支持形态 |
| R12 | **workspace 指针稳定性** | 常驻缓存 workspace + `cudaMemcpyToSymbolAsync` 器件全局（`prefill.cu:6196-6226`）在图捕获下是 memcpy 节点 | 1cat per-device 缓存（:6102-6119） | 我方若走捕获直发混排，symbol 更新时序须重审 | workspace 静态化 + 每调用参数化；device symbol 尽量换成 kernel 参数 [设计选项] |
| R13 | **kBlockN 调小的性能税** | rescale 频率 ↑、采样 max 的块数 ↑ | 8K 块 73.39 vs 24K 75+ TFLOPS（`VALIDATION.md:207-209`） | 16 GB 大概率要 8192-16384 档 | 显存三档自适应 + 每档 A/B 定价 |

---

## 6. 与 P-P3-T（显式 mma PV + Wide + RegP）的衔接点

### 6.1 契约对齐（先读 `docs/compose/spec/fa-gemm-pp3.md`）

P-P3-T = 把 `splitd_pv_gemm_tt` 的手拼 B 碎片 + `cute::gemm` 换成显式 `mma.sync.m8n8k4`，常量按 kDChunk 泛化（`fa-gemm-pp3.md:46`）；P-P3 = 79T 分解（GQA 打包 / prefix-tail 分离 / block rescale / score workspace，:74）。**两者共享"标准 fragment 化的 PV"这块地基**。

### 6.2 可复用地基（P-P3-T 做完后 P-P3 直接继承）

| 地基 | 现状出处 | P-P3 复用方式 |
|---|---|---|
| 显式 `mma.m8n8k4` PV 内核（P-P3-T 真交付物） | 我方 `fattn-sm70-d256-kernel.cuh:256-307`（现状为手拼 B + cute::gemm，:292/:275） | 直接充当 P-P3 的**尾部 fine-PV**内核（79T 尾部本来就是小块 PV，`prefill.cu:6322-6337`）；PV 大 GEMM 可走 CUTLASS（79T 形态）或同内核放大 |
| kDChunk 泛化的常量族 | `Sm70D256SplitDTraitsT<int DChunk>`（:48-59）；O 常驻断言 `kORegElements == kDChunk/4`（:778-781）；RegP per-warp PV TiledMma（:90 附近） | P-P3 的 PV tile（N=256=D）虽须重定拓扑，但 fragment/寄存器映射代数复用 |
| fragment 契约与 lane 映射 | `load_v_fragment_tt`（我方 :228-254 与 `sm70_v37/tail.cu:176-200` 逐字同源）；`flash-attention-v100/include/mma_m8n8k4.h:1-34` 的规格注释 | P-P3-T 改写后即"无 cute 依赖的 B 碎片加载器"，79T 移植时不必再引 CUTLASS SmemIterator 也可自建 |
| HMMA.884 原子事实 | 79T `PVInstructionShape=GemmShape<8,8,4>`（`prefill.cu:470`） | 证明 P-P3 的大 GEMM 与 P-P3-T 的显式 mma 是**同一 ISA 层**；无路线冲突 |
| GQA 打包的落点 | 79T kRows=n_q×6（:5940）；我方现 grid.z=gqa（`fattn-sm70-d256.cu:667`） | P-P3 把 (q_len, gqa) 折进 M rows —— 与 P-P3-T 的 PV 内部改造**正交**，不冲突 |

### 6.3 P-P3-T 现存阻塞点（P-P3 也会撞上的同一批）

- `BLayout Shape<_4,_2> + Stride<_1,_4>` 手拼（我方 :292；`sm70_v37/tail.cu:230`）——1cat 也没泛化（`fa-gemm-pp3.md:18` 已核实）⇒ **无处可抄，自有地基**。
- `load_v_fragment_tt` 的 `[%4+128]` 立即数（我方 :248；`tail.cu:194`）——kDChunk=64 专属。
- `SmemLayoutV` 在 kDChunk=128 非双射（`fa-gemm-pp3.md:17`：偏移 128 两解碰撞）——Wide 是错布局，不是没调好。
- `splitd_pv_gemm_tt` 内硬引用 `Sm70D256SplitDTraitsT<64>`（我方 :291, :298-299）。

### 6.4 衔接注意事项

1. **门值配额**：P-P3-T 若改变累加序，按规约消耗一次重立门（`fa-gemm-pp3.md:66`）；P-P3 又一次（R9）——建议 P-P3-T 保持"逐位等价回退开关 `LLAMA_SM70_FA_GEMM=0`"（:49），把重立门的机会留给 P-P3。
2. **P-P3 的数值守卫（stable_*）与 P-P3-T 无关**：P-P3-T 不必等它；但 P-P3 立项书应把 §1.6 四守卫列为独立工作包（它决定了真模型可用性，与性能无关）。
3. **显存先行**：P-P3-T 的 Wide（smem 约 53 KB、1 CTA/SM）与 P-P3 的 score workspace 都吃资源，但不同池（smem vs 全局）；真正冲突在 §4 的全局预算——**建议 P-P3 立项书把 kBlockN 自适应降档（§4.4）列为必须项而非可选项**。
4. **shape 适配现实**：P-P3 目标形状按任务书为 n_q=2048/8192、n_kv≤262144、TP3；其中 n_q=8192 受显存与 `-ub/-b` 钳制双重约束（BASELINE-LEDGER UBLEVER 行），立项预期（FA +25~30%、256K pp +20%）应以 **n_q=2048 为主锚**重算后登记（本报告不代替预登记）。

---

## 附：本报告的已知空白（诚实声明）

1. `sm70_flash_v100_fp8_kv_long_context.md`、`sm70_v100_migration_control.md`（2.7 MB）未通读——数值配方经 grep 确认不在其中（关键词无命中）。
2. 79T `prefill.cu` 的非生效 `#if` 历史路径（约 4000 行）只读了约束表与生效路径；未逐行审计（按 cmake:26-44 采信生效配方）。
3. TP3 头切分 1/1/2（Q 6/6/12）引自 `1CAT-PORT-BACKLOG.md:201`（其自标 UNVERIFIED），我方本次未实机核对。
4. 显存账中 q8_0 KV cache 4.56 GB、镜像 0.54 GB、杂项 0.5 GB 为公式估算（公式出处已注），未在本机核算。
5. R11/R12（CUDA graph 与多流/符号）是设计推断 + 1cat 生产证据（FULL_AND_PIECEWISE）旁证；我方 ggml 捕获栈的行为**未验证**。
