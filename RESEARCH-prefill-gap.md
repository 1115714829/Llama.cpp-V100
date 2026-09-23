# RESEARCH: 长上下文 prefill 差距的结构性原因 (256K prefill: 370 t/s vs 1cat 2438-4069 t/s)

> ⚠️ **口径时效（R289 重过滤，2026-09-23）**：本文写于 **Path A 之前**。「kernel 效率差 3.1x」是 stock MMA_F16 vs 79T 的跨源对比（U4 自认功耗串扰）；Path A 之后真实内核差距 = **45.9 -> 60.8 TFLOPS ≈ 1.33x**。本文 19.85 TFLOPS 与 R269 的 35.8 是口径（×24 头）与形状（nb 512 vs 2048）差异；「1.49 TFLOPS = 峰值 1.2%」已作废。**引用一律以 `docs/v100-dev/` 收口为准**。

日期: 2026-09-20 (DSH 会话)
作者: 子代理 (research, 只读)
范围: 只回答「为什么我们的 prefill 慢」+「最小可行改动」。不改任何源码。

---

## 0. 结论速览 (TL;DR)

1. **llama.cpp 在 ne11 >= 64 时完全不跑 MMQ/MMVQ**: 所有量化权重 GEMM 走
   `ggml_cuda_mul_mat_cublas` -> **把整张权重张量反量化成 F16** 到临时 buffer -> `cublasGemmEx`
   (F16 x F16 -> F32, `CUBLAS_COMPUTE_32F`)。闸门是**唯一一个常数** `ne11 < 64`
   (`mmq.cu:333-335`, `mmq.cuh:8`)。prefill 的每一层、每一个 ubatch 都重新反量化一次。
2. **prefill 的 attention 走 MMA 路径(不是 VEC/TILE)**, D=256/GQA=6 时 `n_tokens >= 9` 即
   `BEST_FATTN_KERNEL_MMA_F16` (`fattn.cu:644-651`)。但 Volta **没有专用配置表**:
   `ggml_cuda_fattn_mma_get_config_volta()` 只有 512/576 两项, 256 落回 **Ampere 表**
   (`fattn-mma-f16.cuh:113-126`, 原文注释 `// TODO tune specifically for Volta`)。
   实测 (本会话, test-backend-ops, 我们的真实形状 D256/Hq24/Hkv4/F16 KV/causal/V100):
   **19.16 - 19.85 TFLOP/s**。
3. **1cat 的 79T 公开算子在同形状 (Q8000/KV256000) 是 61.09 causal TFLOP/s**
   (`sm70_qwen38_fp8_prefill_decay.md:494-497`)。**attention kernel 效率差 3.1x**。
4. **但更大的结构性差异是「工作分解」而不是「同一个 kernel 慢」**:
   1cat 把 attention 变成**巨型 GEMM**: M = Q*6 (=48000/49152 行, 6 个 GQA query head 打包进 M),
   QK 是**一次 cuBLAS FP16 GEMM (N=24576, K=256)**, PV 是**一次 CUTLASS SM70 128x256x32 GEMM
   (N=256, K=24576)**, online-softmax 的 rescale **每 24576 个 KV token 才做一次**
   (`prefill.cu:6284-6306`, `cmake/sm70_79t.cmake:30,42-44`)。
   llama.cpp 的 FA 是**融合 online-softmax**, KV tile 只有 **nbatch_fa=32** 行, 每个 tile 都要
   rescale 一次, 且 **Volta 上 nstages=0 (没有 cp.async 流水)** (`fattn-mma-f16.cuh:73,349-351`)。
5. **我们自己的 3 卡 tensor split 把 attention 切成 25% / 25% / 50%**: qwen35 的切分粒度是
   `lcm(2*n_embd_q, 128) = 3072` (Q 侧) 和 `granularity_kv = 256` (KV 侧, = 1 个 head);
   `n_head_kv = 4` 用 3 张卡切 -> **1 / 1 / 2 个 KV head**
   (`llama-model.cpp:717-727,776-791,830-847`)。
   关键路径是 rank2, 它干 2 倍的 attention 活。**换 4 卡 (1/1/1/1) 才是 1cat 的形状 Hq6/Hkv1**。
6. 用我们自己的实测数据做两段拟合 (`/tmp/l3.log` 672204.71 ms / 248901 tokens, 以及同机
   1708-token 点) 得 `t(P) = 1.185e-3*P + 6.086e-9*P^2`:
   在 248901 token 上 **二次项(attention) 约 377-418 s, 一次项(GEMM/GDN/通信) 约 295 s**。
   用 FA 实测独立算出的 attention 二次系数 (TP3 关键 rank) 是 **4.95e-9**, 与拟合的 6.09e-9
   同量级 (差 23%)。**attention 占我们 256K prefill 的 50-60%**。
7. **TP allreduce 在 prefill 不是瓶颈**: ub=512 时每 ubatch 约 1.34 GB, 全程约 654 GB,
   即使按 3 卡 ring 的 1.33x 放大也只有 ~870 GB; 在 NV2 上(0,1,2 同 NUMA) 约 20 s = **3%**。
   (1cat 在同模型 8K 的实测是 17.72%, 但那是 M=8000 的大 ubatch, 不是我们的 512。)

---

## 1. 证据分级约定

| 标记 | 含义 |
|---|---|
| **[实测]** | 本会话在 AC922 上真跑出来的数 (test-backend-ops / 服务器日志) |
| **[文档]** | 仓库内文档/README/注释里的宣称, 未在本会话复现 |
| **[源码]** | 本会话直接读到的 file:line |
| **[推导]** | 我根据以上两类做的算术, 含假设, 可能错 |

---

## 2. 要做对比的两个数 (先把标尺对齐)

### 2.1 我们的数

**[实测]** `/tmp/l3.log` 两行原文:

	task 0 | prompt eval time =  672204.71 ms / 248901 tokens ( 2.70 ms per token,  370.28 tokens per second)
	task 0 | prompt eval time =  677586.02 ms / 248901 tokens ( 2.72 ms per token,  367.33 tokens per second)

即 248901 token (不是 262144; 这是 l3 验收的 prompt 长度) 两次: 672.2 s / 677.6 s,
370.28 / 367.33 tok/s。**注意 370 t/s 这个口径的 prompt 长度是 248901, 不是 256K。**

### 2.2 1cat 的数 (同一个模型 Qwen3.8-27B, TP4, 4xV100)

必须先分清几组不同口径, 它们经常被混用:

| 口径 | 数值 | 出处 |
|---|---|---|
| 8K pure prefill (FP8 checkpoint, FP8 E5M2 KV, chunk 8000) | **5170.96 tok/s** (1.547100 s) | [文档] `docs/design/sm70_qwen38_fp8_prefill_5500.md:33-34` |
| 32K cold prefill (NVFP4 + DFlash2, E5M2 KV) | **4039-4069 tok/s** | [文档] `README.md:1017`, `README.md:307` |
| 64K pure prefill (NVFP4 + DFlash2) | **3567 tok/s** | [文档] `README.md:1018`, `README.md:307` |
| 64K (FP8 target-only, restored E4M3 run) | 3420.2 / 3651 tok/s | [文档] `sm70_qwen38_fp8_prefill_decay.md:28`, `sm70_dflash2_context_cost_20260909.md:227` |
| 128K (FP8, chunk 15680) | **2702.9 tok/s** (48.492 s unprofiled) | [文档] `sm70_qwen38_fp8_prefill_decay.md:29,36` |
| **256K pure prefill (FP8 E5M2 KV, chunk 8192)** | **2438.89 tok/s = 107.380095 s** | [文档] `sm70_qwen38_fp8_prefill_decay.md:507` |
| 256K 同机 quality-valid 控制组 (architecture disabled) | 2040.64 tok/s = 128.336222 s | [文档] `sm70_qwen38_fp8_prefill_decay.md:506` |
| 256K performance-only (32 个 token ID 全零, 被拒) | 2934.82 tok/s = 89.234794 s | [文档] `sm70_qwen38_fp8_prefill_decay.md:436,444-445` |
| 256K (Qwen3.6-27B-AWQ, 1312 MHz 固定时钟, chunk 4096) | 1424.8 tok/s = 183.9463 s | [文档] `sm70_fa2_d256_prefill_pipeline.md:308` |

**同长度可比口径: 248901 token 时我们 672 s, 1cat 256K 是 107.4 s (quality-valid) 到 89.2 s
(perf-only)** => **6.3x / 7.5x**。不是 11x (370 vs 4069 是 256K vs 32K, 不同长度)。

**[文档]** `sm70_dflash2_nvfp4_prefill_promotion.md:29-31` 明确说: "The often-quoted 5170.96
tok/s exact-8K and 2438.89 tok/s 256K results are FP8 target-only contracts", 也就是说
**256K 的 2438.89 和 8K 的 5170.96 是 FP8 target-only, 不是 DFlash2 的数**;
DFlash2 的 32K/64K 是 3959/3450 (promotion 之后)。

---

## 3. 1cat 的 prefill 路径 (问题 1)

### 3.1 组件与 selector

**[源码]** `1cat-vllm/csrc/attention/sm70_79t/` 是"D256 GQA 长 prefill"的 Q8000/Q8192 核:

- `csrc/attention/sm70_79t/README.md:9-16`: 默认路由 (Q8000-core), 回滚开关
  `VLLM_FLASH_V100_PREFILL_D256_GQA_V37=1`。
- `csrc/attention/sm70_79t/README.md:18-27`: **准入形状** = FP16, Q 在 8000..8192, Hq6/Hkv1/D256,
  causal, scale 1/16, KV >= Q 且 <= 262144 且 **32-token 对齐**。
  Q8001..8191 被 **leading-pad 到 8192**; Q8000 用 320-token tail specialization,
  Q8192 用原生 256-token tail。per-request `long_prefill_token_threshold=8192`。
- `csrc/attention/sm70_79t/prefill_q8192.cu:6-12`: 同一个 `prefill.cu` 用另一套宏再编译一遍:
  `FLASH_NAMESPACE onecat_79t_q8192`, `PREFIX_TORCH_QUERY_TOKENS 8192`,
  `PREFIX_BATCHED_TAIL_TILE_TOKENS 256`。
- `csrc/attention/sm70_79t/prefill.cu:72-73` (默认 Q=8000), `:267-268` (默认 tail tile 1000),
  `:443-444` (默认 BLOCK_N 8192)。
- `csrc/attention/sm70_79t/register.cpp:12-21`: 注册成 `_vllm_fa2_C` 的
  `sm70_d256_gqa_architecture_q8192_fwd` op, 不是独立 DSO。

### 3.2 关键参数 (源码里的数, 直接抄)

**[源码]** `1cat-vllm/cmake/sm70_79t.cmake:26-44` 是"tile shape 与 flag 的唯一真相"
(README.md:69 原话: "The CMake recipe is the source of truth for tile shapes and flags"):

	PREFIX_TORCH_BLOCK_N=24576                     <- 前缀 KV 分块宽度 = 24576
	QK_TB_M=128 QK_TB_N=128 QK_WARP_M=32 QK_WARP_N=64 QK_STAGES=2
	PV_TB_M=128 PV_TB_N=256 PV_TB_K=32
	PV_WARP_M=64 PV_WARP_N=64 PV_WARP_K=32
	PREFIX_BATCHED_TAIL_TILE_TOKENS=320
	PREFIX_BATCHED_TAIL_QK_ALGO=CUBLAS_GEMM_ALGO10_TENSOR_OP
	PREFIX_TAIL_FINE_PV_GROUP_TILES=4
	PREFIX_BATCHED_TRI_REPAIR_TOKENS=64
	PREFIX_PV_FP32_MMA_ACCUMULATE PREFIX_TORCH_PREFIX_FP32_OUTPUT
	PREFIX_QK_CUBLAS_RAW PREFIX_TRANSPOSED_SCORE_WORKSPACE
	PREFIX_TORCH_STABLE_ROWS PREFIX_PV_FUSED_PREFIX_SUM ...

编译选项 (同一文件 `:48`): `-gencode=arch=compute_70,code=sm_70 -O3 --use_fast_math`。
指令形状 `cutlass::gemm::GemmShape<8,8,4>` (`prefill.cu:335,470`) = **Volta HMMA.884**。

`prefill.cu:5938-5958` (Workspace 常量): `kRows = kQuery*6`, `kHeadDim = 256`,
`kMinTotalKV = kQuery`, `kMaxTotalKV = 262144`, `kBlockN = 24576`。
`prefill.cu:6036-6041` 把它钉死: `(kQuery==8000 && kTailTileTokens==320 && kTailTiles==25) ||
(kQuery==8192 && kTailTileTokens==256 && kTailTiles==32)`, 且 `kBlockN` 是 8192..131072 的 8192 倍数。

### 3.3 计算流程 (这才是结构差异的核心)

**[源码]** `prefill.cu:6640-6697`: 因为 `PREFIX_QK_CUBLAS_RAW && PREFIX_BATCHED_TRI_TAIL &&
PREFIX_TAIL_IDLE_SM_FINE_PV && PREFIX_TAIL_FINE_PV_DIRECT_ACCUMULATE` 全在 cmake 里定义了,
真正的实现是 `sm70_d256_gqa_half2_family_fwd` (`prefill.cu:6121`)。

**[源码]** `prefill.cu:6126-6127`: `prefix = total_kv - kQuery`。
把 KV 分成两段: **长的、完全可见的前缀** + **最后 Q 个 token 的因果三角尾巴**。

**[源码]** `prefill.cu:6228-6241`: 先把 Q 和 K **整体转置**到 workspace
(`transpose_half_32x32`), 于是 QK 变成规整的 GEMM。

**[源码]** `prefill.cu:6284-6306`: 对前缀按 `kBlockN = 24576` 循环, 每块:
1. `CublasQKLauncher` -- **一次 cuBLAS**;
2. `PrefixFloatPVLauncher` -- **一次 CUTLASS SM70 kernel**。

**[源码]** `prefill.cu:1925-1969` (CublasQKLauncher):

	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, rows, width, 256,
	             &alpha, query_transposed, CUDA_R_16F, query_stride,
	                     key_transposed,   CUDA_R_16F, key_stride,
	             &beta,  scores, CUDA_R_16F, rows,
	             kComputeType, qk_algorithm)

- `rows = 48000/49152` (= Q x 6, 把 6 个 GQA query head 打包进 M);
- `width = 24576` (KV 块宽); `k = 256` (= head_dim);
- `kComputeType = CUBLAS_COMPUTE_16F` (默认分支, `:1962`; `PREFIX_QK_CUBLAS_FP32_ACCUM` 不在 cmake 列表里);
- `qk_algorithm = CUBLAS_GEMM_ALGO9_TENSOR_OP` (`:1937`), 可用 env 覆盖 (`:1938-1942`)。

**[源码]** `prefill.cu:2060-2096` (PrefixFloatPVLauncher):
`cutlass::gemm::GemmCoord problem(rows, 256, k=width)`, tile `(PV_TB_M, PV_TB_N, PV_TB_K) =
(128, 256, 32)`, `PrefixFloatPVOutputOp::Params(1.0f, accumulate ? 1.0f : 0.0f)` --
即 PV 把结果**累加**进 FP32 前缀累加器, 每个 24576 块只做一次 online 合并。

**[源码]** `prefill.cu:6371-6402` (三角尾): `cublasGemmBatchedEx` 一次性发
`kTailTasks = 25*26/2 = 325` (Q8000) 或 `32*33/2 = 528` (Q8192) 个 tile 对,
算法 `CUBLAS_GEMM_ALGO10_TENSOR_OP`, 之后 `mask_batched_tri_tail_diagonal` 打对角 mask。
尾部的 PV 用 `kFinePVTasks = 91 / 144` 个 task 的 fine-grained 调度 (`prefill.cu:6308-6343`,
`PREFIX_TAIL_FINE_PV_GROUP_TILES=4`), 并且和前缀 **在不同 stream 上重叠**
(`prefix_stream` / `tail_stream`, `prefill.cu:5961-5968, 6165-6166`)。

**[文档]** README.md:37-51 描述了数值配方 (这是**宣称**, 我只在文档里看到):
零漂移指数 + FP16 累加会溢出真实模型输入; 合格版改成"stride 8 采样的 max + 指数上限"、
"中心化 value"、"用 2 的整数次幂缩放 value 残差 (64x 余量)"; PV 用 FP16 Tensor Core 操作数 +
**FP32 MMA 累加**, **每个 24K 前缀块写成 FP32**, 前缀/尾合并也在 FP32 做。
"Current qualified medians exceed **75 TFLOPS** at KV128K and KV256K" -- 这是文档宣称。

### 3.4 1cat 侧 GEMM (dense projection) 的路径

**[源码]** `csrc/sm70_turbomind/ops/qwen38_prefill_cutlass.cu:24-36`: prefill dense GEMM 是
CUTLASS SM70 TensorOp: threadblock **128x256x32**, warp **64x64x32**, instruction **8x8x4 (s884)**,
**2 stages**, align 8, `GemmIdentityThreadblockSwizzle<8>`,
epilogue `LinearCombination<half_t, 8, float, float>`。与 79T 的 PV 配置**同一族**。

**[源码]** `qwen38_prefill_cutlass.cu:68-77`: 准入是**精确形状**: `M == 8000` 且
`(K=5120,N in {4096,3584})` / `(5120,8704)` / `(4352,5120)` / `(1536,5120)`。
这就是 `sm70_qwen38_fp8_prefill_5500.md` 里那些 trace 的 GEMM。

### 3.5 1cat 的 prefill 时间去向 (实测级文档)

**[文档]** Qwen3.8-27B-FP8, TP4, 8000 token, 每 rank kernel service
(`sm70_qwen38_fp8_prefill_5500.md:58-68`):

| 类别 | ms | 占比 |
|---|---:|---:|
| gate/up exact CUTLASS | 468.787 | 30.27% |
| QKV/QKVZ exact CUTLASS | 325.594 | 21.02% |
| fused collective + Gemma norm | 274.364 | 17.72% |
| down exact CUTLASS | 218.743 | 14.12% |
| FlashQLA GDN core | 78.876 | 5.09% |
| full attention | 76.801 | 4.96% |
| FP8 权重反量化 | 31.126 | 2.01% |
| interleaved SiLU/mul | 20.101 | 1.30% |
| other | 54.302 | 3.51% |

合计 1548.7 ms, 与 pure prefill 1.547 s 基本相等 => **几乎没有任何重叠余量**。

**[文档]** 128K 关键 rank trace (同模型, chunk 15680, `sm70_qwen38_fp8_prefill_decay.md:36-54`):

| 类别 | 时间 | 占 profiled wall 49.821 s |
|---|---:|---:|
| D256 exact-dense attention | 17.710 s | 35.55% |
| D256 direct-paged attention | 3.099 s | 6.22% |
| FP8 exact-dense projections | 12.787 s | 25.67% |
| TurboMind FP8 projections | 5.551 s | 11.14% |
| TP communication | 4.895 s | 9.82% |
| GDN / linear attention | 1.640 s | 3.29% |
| other FP16 GEMM | 1.547 s | 3.11% |
| norm / elementwise | 1.240 s | 2.49% |
| KV cache and gather | 0.042 s | 0.08% |
| host + unattributed | 1.309 s | 2.63% |

=> **在 128K, attention 已经是最大单项 (41.8%)**, 通信 9.8%, GEMM 约 40%。
"Mean per-layer latency grows from 18.50 ms in the first group to 256.59 ms in the eighth group"
(`:53-54`), 与 operator 表里 KV=125440 的 242.954 ms 对上 (`:58-59`)。

### 3.6 operator 级 TFLOP/s 阶梯 (对标用)

**[文档]** 同形状 (Q8000/6 heads/1 KV head/D256) 的 operator:

| 版本 | 延迟 | causal TFLOP/s | 出处 |
|---|---:|---:|---|
| 老 exact Split-D (N32 Split-D) | 289.605637 ms @ KV256000 | 42.77 | `sm70_qwen38_fp8_prefill_decay.md:496` |
| **79T "stable max-aware architecture"** | **202.749443 ms @ KV256000** | **61.09** | `sm70_qwen38_fp8_prefill_decay.md:497` |
| Split-D @ Q4096/KV64K | 42.7131 ms (split-KV3) | 37.4 (我算) | `sm70_fa2_d256_prefill_pipeline.md:199` |
| Split-D @ Q4096/KV256K | 192.8289 ms (gather+dense) | 31.0 (我算) | `sm70_fa2_d256_prefill_pipeline.md:165` |

**[文档]** README.md:81-83 的阶梯: "17.92 -> 46.63-47.1 -> ~60.8 TFLOP/s" (同一代 GPU),
79 TFLOP/s 是"实验上限, 不作为生产质量声明"。README.md:48-49 说 qualified medians > 75 TFLOPS
@ KV128K/256K -- 这两处口径不同, **75 是文档宣称, 61.09 是同文件里带 SHA 的实测**。

---

## 4. llama.cpp 侧的长 prefill 路径 (问题 2)

模型确认: **[实测]** `general.architecture = qwen35`
(在 `/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf` 头部 strings 读到,
对应 `src/llama-arch.cpp:41 { LLM_ARCH_QWEN35, "qwen35" }`), 即 hybrid GDN arch。

### 4.1 attention: 走 MMA, 不是 VEC/TILE

**[源码]** 判据全在 `ggml/src/ggml-cuda/fattn.cu`:

- `fattn.cu:726-741`: `ggml_cuda_flash_attn_ext()` 只是个壳, 真正决策在
  `ggml_cuda_get_best_fattn_kernel()` (`fattn.cu:517-688`)。
- `fattn.cu:552-564`: switch on `K->ne[0]`; `case 256:` 要求 `V->ne[0] == K->ne[0]`。
- `fattn.cu:638-642`: `gqa_ratio_eff` = gqa_ratio 里最大的 2 的幂 (cap `ncols2_max = 8` for D=256)。
  我们 gqa_ratio = 24/4 = 6 => **gqa_ratio_eff = 2**。
- **`fattn.cu:644-651` (Volta 分支, 这就是答案)**:

      if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
          if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) return BEST_FATTN_KERNEL_VEC;
          if (Q->ne[1] * gqa_ratio_eff <= 16) return BEST_FATTN_KERNEL_TILE;
          return BEST_FATTN_KERNEL_MMA_F16;
      }

  `Q->ne[1]` = 这次调用的 query token 数。gqa_ratio_eff=2 => **n_tokens 1 -> VEC;
  2..8 -> TILE; >= 9 -> MMA_F16**。prefill (ub=512) **必然是 MMA_F16**。
- `fattn.cu:200-217`: Volta 专门分支 -- gqa_ratio 不能被 4 整除时走
  `switch_ncols1<DKQ,DV,2>` (我们的 6 就是这条), 注释原文
  `// On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:`。
- `fattn.cu:309-312` -> `switch_ncols2<256,256>` -> `case<256,256,32,2>`
  (`template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_2.cu:10`)。
- `fattn.cu:706-715`: **TILE 和 MMA_F16 都要求 K/V 是 F16** (`need_f16_K = need_f16_V = true`)。
  => 用 `-ctk q8_0 -ctv q8_0` 时, 每次 FA 调用都要先把 K/V 转成 F16 (额外 O(KV) 的活)。
- `common.cuh:360-361`: `volta_mma_available(cc) = IS_NVIDIA(cc) && highest_compiled_arch(cc) == 700`。
  **[实测]** 出货 `.so` 只有 sm_70 cubin (`/root/libdir-nccl/libggml-cuda.so.0.24.0`), 所以为真。
  唯一能把它翻掉的是 build 的 arch list 里没有 70 (那时会掉到 TILE, `fattn.cu:674-687`)。

### 4.2 attention: 实际 kernel 配置 (Volta 用的是 Ampere 表)

**[源码]** `fattn-mma-f16.cuh:113-126`:

	static constexpr ... ggml_cuda_fattn_mma_get_config_volta(const int DKQ, const int DV, const int ncols) {
	    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8, ...);
	    ... 只有 512/512 和 576/512 ...
	    // TODO tune specifically for Volta
	    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
	}

**[源码]** 于是 D=256 命中 Ampere 表 `fattn-mma-f16.cuh:73`:

	GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2, 32, 128, 128, 128, 2, true);
	// (DKQ, DV, ncols, nthreads, occupancy, nbatch_fa, nbatch_K2, nbatch_V2,
	//  nbatch_combine, nstages_target, Q_in_reg)

=> **nthreads=128 (4 warps), occupancy=2, nbatch_fa=32, nbatch_K2=128, nbatch_V2=128,
nbatch_combine=128, Q_in_reg=true**。

**[源码]** `fattn-mma-f16.cuh:349-351`:

	static __host__ int ggml_cuda_fattn_mma_get_nstages(...) {
	    return cp_async_available(cc) && ncols2 >= 2 ? ggml_cuda_fattn_mma_get_nstages_target(...) : 0;
	}

`cp_async_available` 要求 >= AMPERE (`common.cuh:372-374`) => **Volta 上 nstages = 0,
即 K/V 的 shared memory 只有单缓冲, 没有软件流水**。
同族: `fattn-swizzle.cuh` 的 XOR swizzle / ldmatrix 也是 `TURING_MMA_AVAILABLE` 专属,
Volta 只能拿 `tile_stride = nbatch_2 + 4` (`fattn-swizzle.cuh:34-40`)。
`fattn-mma-f16.cuh:1816-1821`: Volta 要求 `ncols1*ncols2 >= 32` (我们 64, 满足)。
sparse FA 是 Turing-only (`fattn.cu:125`)。

=> **一句话: 我们的 prefill attention 是一个"KV tile 只有 32 行、单缓冲、用 Ampere 参数、
注释里写着 TODO 没调过"的 HMMA.884 融合 kernel。**

### 4.3 attention 实测 [实测]

在 GPU3 (独占) 上跑 `build-nccl/bin/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0`,
形状正好是 `hsk=256,hsv=256,nh=4,nr23=[6,1]` (即 hq=24, hkv=4, GQA=6), mask=1 (causal),
`type_K=type_V=f16`, permute [0,1,2,3]:

| kv | nb | us/run | GFLOP/run | TFLOPS |
|---:|---:|---:|---:|---:|
| 4096 | 1 | 57.63 | 0.10066 | 1.75 |
| 4096 | 512 | 2689.52 | 51.54 | **19.16** |
| 16384 | 1 | 182.55 | 0.40265 | 2.21 |
| 16384 | 512 | 10534.46 | 206.16 | **19.57** |
| 65536 | 1 | 720.03 | 1.61 | 2.24 |
| 65536 | 512 | 41556.08 | 824.63 | **19.84** |
| 131072 | 1 | 2009.94 | 3.22 | 1.60 |
| 131072 | 512 | 83069.31 | 1650.0 | **19.85** |

注: 报告的 GFLOP 是**全平方**口径 (`2*2*D*hq*nb*kv`)。因为 nb=512 << kv,
因果三角只占 0.2%, 所以 **19.85 同时就是 causal TFLOP/s**。
(来自 `test-backend-ops.cpp:11309-11316`, 那 8 个 case 就是照我们模型几何写的。)

**对标: 19.85 vs 1cat 79T 的 61.09 => 3.1x; vs 老 Split-D 的 42.77 => 2.2x。**
这是 [实测] vs [文档] 的对比, 不是同机同轮, 但两边都是 "per rank, 4xV100, 同 GPU 代数"。

### 4.4 大 batch matmul: ne11 >= 64 走"整张权重反量化 + cuBLAS"

**[源码]** 分派链 (`ggml/src/ggml-cuda/ggml-cuda.cu`):

- `ggml-cuda.cu:1823-1877` `ggml_cuda_mul_mat()`, 首次命中即返回。
- MMVF / MMF 对量化类型**直接 false** (`mmvf.cu:872-874`; `mmf.cu:135-137`)。
- MMVQ 闸门 `mmvq.cu:324-428`: V100 上 Q8_0 只在 `ne11 <= 8` (`mmvq.cuh:3`,
  以及本 fork 的 C5 patch `mmvq.cu:374-383`, `mmvq.cuh:7`)。
- **MMQ 闸门 = 关键** (`mmq.cu:333-335`):

      if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
          return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
      }

  `MMQ_DP4A_MAX_BATCH_SIZE = 64` (`mmq.cuh:8`), `fp16_mma_hardware_available(700) = true`
  (`common.cuh:326-330`) => **V100 上 MMQ 只在 ne11 <= 63 生效。**
- `ggml-cuda.cu:1876` 落到 `ggml_cuda_mul_mat_cublas()`。

**[源码]** cuBLAS 分支内部:

- `ggml-cuda.cu:1624-1626`: 量化 -> `compute_type = fast_fp16_hardware_available(cc) ? F16 : F32`
  => V100 上是 F16。
- `ggml-cuda.cu:1447-1470`: `src0_alloc.alloc(ggml_nelements(src0))` + `convert_func(...)`
  => **每次调用都把整张权重反量化成 F16 临时 buffer** (`convert.cu:472/478` 有 Q8_0/Q4_K)。
- `ggml-cuda.cu:1472-1495`: src1 (F32 激活) 也转成 F16。
- `ggml-cuda.cu:1509-1522`: **Volta 专属**: `prefer_f32_output = (cc == GGML_CUDA_CC_VOLTA)`
  => `CUBLAS_COMPUTE_32F`, A/B = `CUDA_R_16F`, C = `CUDA_R_32F`。
- `ggml-cuda.cu:1550-1558`: `cublasGemmEx(OP_T, OP_N, ne01, ne11, ne10, ...,
  CUBLAS_GEMM_DEFAULT_TENSOR_OP)`。
- `ggml-cuda.cu:1641-1656`: `GGML_CUDA_CUBLAS_COMPUTE_TYPE` 可覆盖, 但 Volta 的
  `prefer_f32_output` 在其后仍生效。
- `ggml-cuda.cu:1637-1639`: `dst->op_params[0] == GGML_PREC_F32` 也会改 compute_type。

**[源码]** MMQ 内部(只对 ne11 <= 63 有意义): Volta 走 **Ampere 配置表**
(`mmq.cuh:252-254`, 判据是 **compiled arch**, 不是 device) 但用 **dp4a tile-x 布局**
(`mmq.cuh:189-194` `use_mma_data_layout()` 对 Volta 返回 false)。I=128, nthreads=256,
occupancy=1, stream_k=true (`mmq-config-ampere.cuh:104-119`)。
=> **Volta 的 MMQ 不用 HMMA, 用 dp4a SIMT 整数路径。**

### 4.5 -b / -ub 分块影响 [推导, 部分源码]

**[源码]** `src/llama-context.cpp:249`: `n_ubatch = min(n_batch, n_ubatch)`;
`llama-context.cpp:1783`: `GGML_ASSERT(n_tokens_all <= cparams.n_batch)`;
`:1811` 每个 ubatch 自己的 graph; 一次 MUL_MAT 节点 = 一次 GEMM, **不按 ne11 再切**
(`ggml-cuda.cu:1553` 把整个 ne11 当 cuBLAS 的 n)。

推论:

1. **GEMM 侧**: `ub` 越小, GEMM 的 M 越小(效率越低), 而且**每个 ubatch 都重新把整张权重反
   量化一次**。ub 从 512 提到 4096 => 反量化次数降到 1/8。
2. **attention 侧**: 总量 `sum_over_chunks a*nb*KV ~ a*P^2/2`, **与 ub 无关**(一阶)。
   但 (i) 每次 FA 调用有固定开销; (ii) 用 q8_0 KV 时每次调用都要把 K/V 转 F16
   (`fattn.cu:706-715`), 这个转换是 O(KV) 每次 => 总 O(P^2/ub), ub 大则这项小。
   **[实测]** nb=1 的 FA 是另一个 kernel (VEC), 不能拿 1.60-2.24 TFLOPS 去跟 nb=512 比。
3. **allreduce 侧**: 每次 allreduce 的数据量正比于 ub, 调用次数正比于 P/ub => 总量与 ub 无关;
   但 ub 大时单次更大、更接近带宽, 延迟占比更低。
4. `n_batch` (默认 2048) 与 `n_ubatch` (默认 512) 不同: 我们的 256K prefill 实际是
   **~486-512 个 ubatch** (248901/512)。1cat 用的是 **chunk 8192/15680** (见 3.5)。

### 4.6 3 卡 tensor split 把 attention 切成 25/25/50 [推导 + 源码]

这条是本报告最重要的"结构性"发现。`-sm tensor` 的权重切分在
`src/llama-model.cpp:605-847`:

- `:608` qwen35 走 hybrid 分支; `:612-617` **fused QKV** 的 segment 是
  `{{n_embd = n_head*n_embd_head_k*2, 1}, {n_embd_gqa, 2}}` (Q 侧翻倍因为有 Q gate)。
- `:717-727` (非递归层, 即 16 层 full attention):
  `n_embd_q = n_gqa * n_embd_head_k`; `blck_size_perf` 从 32 翻倍到 128;
  `granularity_q = lcm(n_embd_q, blck_size_perf)`。
- `:776-782`: **`granularity_kv = granularity_q / n_gqa`** (对一个 KV head 的粒度)。
- `:787-791`: qwen35 的 qkv 段粒度 = `{lcm(2*n_embd_q, blck_size_perf), granularity_kv}`。
- `:830-847` 切分循环, 关键两行:

      int64_t high = ne_s * tensor_split_scan[j]/tensor_split_scan.back();
      if (high % g_s != 0) { high -= high % g_s; }

代入我们的数: `n_gqa = 6`, `n_embd_head_k = 256`, `blck_size_perf = 128`
(32->64->128, 因为 `128*3 < 1536`), 所以 `granularity_q = lcm(1536,128) = 1536`,
`granularity_kv = 1536/6 = 256` (= 正好 1 个 KV head), Q 侧粒度 `lcm(3072,128) = 3072`。

- **KV 权重**: `ne_s = n_embd_gqa = 4*256 = 1024`, 3 卡 `1/1/1`:
  high0 = 1024/3 = 341 -> 341%256=85 -> 256; high1 = 682 -> 682%256=170 -> 512;
  rank2 拿 1024-512 = **512**。=> **256 / 256 / 512 = 1 / 1 / 2 个 KV head**。
- **Q 权重**: 若带 Q gate 则 `ne_s = 2*6144 = 12288`, 粒度 3072:
  high0 = 4096 -> 4096%3072=1024 -> 3072; high1 = 8192 -> 8192%3072=2048 -> 6144;
  rank2 拿 12288-6144 = 6144。=> **3072 / 3072 / 6144 = 6 / 6 / 12 个 Q head**。

两者自洽: **rank0/rank1 各 1 个 KV head + 6 个 Q head, rank2 有 2 个 KV head + 12 个 Q head。**
在 TP 里所有 rank 跑同一批 layer 并 allreduce => **wall time 由 rank2 决定, 它做 2 倍的
attention 活**。有效 attention 并行度是 **2x, 不是 3x**。
而 1cat 的 79T kernel 准入形状正是 **Hq6/Hkv1** (每 rank 6 个 Q head + 1 个 KV head)
= **TP4**。

**[未验证 U1]** 我没有在运行日志里看到 per-tensor 的 device 分配 dump。上面的算术完全
来自源码, 需要一次 `llama-server -v` 的 load 日志或 tensor-offload 打印来确认。
如果 `pattern_qkv_weight` 不匹配我们的 GGUF (即模型用分开的 attn_q/attn_k/attn_v),
KV 侧结论不变 (`pattern_kv_weight` 走 `:777-782` 同一个 `granularity_kv`),
Q 侧会变成 `:754-762` 的 `lcm(n_embd_q, blck_size_perf) = 1536` 或 qwen35 的 3072,
结果仍是 1/1/2 的头部不平衡。

### 4.7 TP allreduce 占比 [推导]

- 每次 allreduce 张量 = `[n_embd=5120, ub]` F32; 每层 2 次 (attn_out + ffn_out), 64 层
  => 128 次/ubatch。
- ub=512: `128 * 5120 * 512 * 4 B = 1.342 GB/ubatch`; 486 个 ubatch => **654 GB**。
- 3 卡 ring 放大 `2*(N-1)/N = 1.33x` => **~870 GB**。NV2/NVLink2 有效 40 GB/s 单向
  => **~22 s = 3.2%** of 672 s。(0,1,2 全部在 NUMA0, 全 NV2 组内。)
- 结论: **allreduce 在 prefill 不是主因。** 与 decode 侧的结论 (=每轮延迟的头号嫌疑) 不同。
- 对照 1cat: 同模型 8K/TP4 的 `fused collective + Gemma norm` 是 **274.364 ms / 17.72%**
  (`sm70_qwen38_fp8_prefill_5500.md:62`), 128K 是 `TP communication 4.895 s / 9.82%`。
  他们的 ub 是 8000, 我们是 512, 即 **ub 大 16 倍, 通信占比却只有 17.7%**, 说明
  1cat 的瓶颈是"通信没有和计算重叠", 而我们的瓶颈根本不是通信。

---

## 5. 结构性差异的定量拆解 (问题 3)

### 5.1 用我们自己的实测把 672 s 拆成 A*P + B*P^2

**[实测]** 服务器日志里能拿到的两个不同长度的 prompt eval:

| P (tokens) | 时间 | 来源 |
|---:|---:|---|
| 1708 | 2042.30 ms (836.31 tok/s) | `/tmp/*.log` (同机 llama-server, 具体臂未标注) |
| 248901 | 672204.71 ms (370.28 tok/s) | `/tmp/l3.log` |

解 `A*P + B*P^2`:

	A = 1.1852e-3 s/token       (一次项: GEMM + GDN + allreduce + 固定开销)
	B = 6.086e-9  s/token^2     (二次项: attention)

在 P = 248901: 一次项 295.0 s (43.9%), 二次项 377.2 s (56.1%)。
在 P = 262144: 预测 729 s (比实测 672 s 高 8% -- 与本机已知的 8% 漂移同量级)。

### 5.2 独立验算二次项 (用 4.3 的 FA 实测)

**[推导]** 单次 FA 调用 (nb=512) 的每 (query token x KV token) 代价:
`a = 83069.31 us / (512 * 131072) = 1.238e-3 us = 1.238e-9 s`。

整个 prefill (所有 24 Q head / 4 KV head, 16 层, 全部在一张卡上) 的总 attention 时间:

	16 * a * P^2 / 2 = 16 * 1.238e-9/2 * P^2 = 9.90e-9 * P^2

即"一张卡算全部 attention"的二次系数 **9.90e-9**。
按 4.6 的 1/1/2 切分, **关键 rank (rank2) 承担一半** => **4.95e-9**。
拟合得到的是 **6.09e-9**, 同量级, 差 23% (拟合对两个点的位置很敏感, 且 1708 那个点含固定开销)。

=> **我们的 256K prefill 里 attention 占 50-60%, 且这个量级完全由 D256/GQA6 的 MMA kernel
效率 (19.85 TFLOP/s) 加上 3 卡头部不平衡决定。**

如果 attention 只有 1cat 79T 的效率 (61.09 causal TFLOP/s), 同样的关键路径会变成:

	B_79T = 9.90e-9 * (19.85/61.09) / 2 = 1.61e-9   (rank2 承担一半)
	=> 二次项从 ~340-418 s 降到 ~110-136 s

### 5.3 一次项 (约 295 s) 里有什么

**[推导]** 模型 27B 参数 (`AGENTS.md` §1: Q8_0 4 卡 7.25 GB/卡 => 总 ~29 GB),
per rank 参数 = 27e9/3 = 9e9; P=248901 时 per-rank GEMM FLOPs
`2*9e9*248901 = 4.48e15`。若一次项 295 s 全是 GEMM => **15.2 TFLOP/s per rank**。

- V100 peak (FP16 TC, 1.53 GHz) = 125 TFLOP/s => **12% MFU**。
- 已知的固定损耗: 每个 ubatch 都要把整张权重反量化 Q8_0 -> F16。
  每 ubatch: 读 9.56 GB (Q8_0) + 写 18 GB (F16) + GEMM 再读 18 GB (F16 而非 9.56 GB Q8_0)
  = **~36 GB 额外流量/ubatch**; 486 个 ubatch = 17.5 TB; 按 ~700 GB/s = **25 s (3.7%)**。
- GDN (48 层线性注意力): `AGENTS.md` §1 记录"GDN 在大 n 掉到 42-50 GB/s",
  也是 O(P) 项。
- allreduce ~22 s (3.2%)。
- 剩余 (295 - 25 - 22 = 248 s) 才是 GEMM 本身 => `4.48e15/248 = 18.1 TFLOP/s` per rank,
  离 cuBLAS FP16 在 M=512 上应该能到的 40+ TFLOP/s 有明显距离。
  **[未验证 U2]** 具体是 M=512 太小、还是 dequant 与 GEMM 没重叠、还是 cuBLAS 选了差
  algorithm, 没有 profiler 无法定论 (ncu/nsys 在本机封死, 见 `AGENTS.md` §1)。

### 5.4 归因排序 (结论)

| 排名 | 结构性差异 | 量级 | 证据 |
|---|---|---|---|
| 1 | **attention kernel 效率**: 19.85 vs 61.09 causal TFLOP/s | **3.1x** | [实测] 4.3 vs [文档] 3.6 |
| 2 | **工作分解**: 1cat = 巨型 GEMM + 每 24576 KV 一次 rescale; 我们 = 32 行 KV tile 的融合 online-softmax, Volta 无流水 (nstages=0) | 体现在 #1 里 | [源码] 3.3 / 4.2 |
| 3 | **3 卡头部不平衡 25/25/50** | attention 关键路径 x1.5 (相对理想 3 等分) | [源码+推导] 4.6 |
| 4 | **量化 GEMM 走 dequant+cuBLAS, 每 ubatch 重来** | ~25 s (3.7%) + GEMM 效率损失 | [源码] 4.4 |
| 5 | **ub 太小 (512 vs 1cat 8192/15680)** | 放大 #4、#6, 但 attention 总量不变 | [推导] 4.5 |
| 6 | TP allreduce | ~22 s = **3.2%**, 不是瓶颈 | [推导] 4.7 |

---

## 6. 最小可行改动 vs 必须重写 (问题 4)

### 6.1 最小可行 (只动运行参数或一两张表, 天级)

**M1. 换 4 卡 (或 2 卡), 不要 3 卡。** 零代码。
理由: 4.6 的切分算术。`n_head_kv = 4` 在 3 卡上是 1/1/2; 在 4 卡上是 1/1/1/1
(每 rank 正好 6 Q head + 1 KV head = **1cat 的准入形状**); 2 卡是 2/2。
预期: attention 关键路径 **减半**, GEMM 也从 1/3 变 1/4。
验证方式: `-ts 1/1/1/1` 跑同一个 256K prompt, 比 `-ts 1/1/1`。
(用户已明确"卡数 1-6 自由, 跨岛不是瓶颈"。)

**M2. 加大 `-ub` / `-b`。** 零代码 (`src/llama-context.cpp:249` 是默认值来源)。
- 从 `-ub 512` 提到 2048/4096: GEMM 的 M 变大、每 ubatch 的反量化与 allreduce **次数**降 4-8 倍。
- 代价: 需要更多临时显存 (反量化 buffer + 激活)。
- 注意: 这与 1cat 的 chunk 8192 是同一个方向。
- **[未验证 U3]** attention 总量对 ub 的一阶无关性是我的推导; 实测需要两个 ub 点。

**M3. KV cache 用 f16 而非 q8_0。** 零代码 (`-ctk/-ctv`)。
理由: `fattn.cu:706-715` -- MMA/TILE 路径强制 K/V 为 F16, q8_0 KV 每次调用都要多一趟
O(KV) 的转换, 而 prefill 有 ~486 次 x 16 层 = **7776 次**这样的转换。
(这也解释了为什么 KV dtype 的 A/B 在长 prefill 上差异明显。)

**M4. 确认 `-fa on` 且 build 的 arch list 含 70。** `fattn.cu:644` 依赖
`volta_mma_available()` = `highest_compiled_arch(cc) == 700`。当前 `.so` 只有 sm_70 cubin,
但如果以后有人把 arch list 改成 `75;80`, **prefill 会静默掉到 TILE kernel** (`fattn.cu:674-687`)。
这一条应该写进 build 纪律。

### 6.2 中等工程 (单文件, 一到两周) -- 我最推荐的一条

**M5. 给 `ggml_cuda_fattn_mma_get_config_volta` 补一张真正的 Volta 表 (D=256/GQA6)。**
文件: `ggml/src/ggml-cuda/fattn-mma-f16.cuh:113-126`。

现状原文就是 `// TODO tune specifically for Volta` + 落回 Ampere 表, 拿到的是
`(256,256,64, nthreads=128, occ=2, nbatch_fa=32, nbatch_K2=128, nbatch_V2=128,
nbatch_combine=128, nstages=2, Q_in_reg=true)`。
sweep 维度: `ncols` (8/16/32/64 -> 决定 ncols1 与 ntiles), `nthreads` (128/256),
`nbatch_fa` (32/64/128), `nbatch_K2`/`nbatch_V2`, `occupancy`, `Q_in_reg`。

这是本 fork **已经成功过的同一招**: C4 (`MMVQ_PARAMETERS_VOLTA`) 和 C5 (V100 mmvq/mmq
交叉点 = 4) 都是"上游独缺 Volta 分支, 补一张表"(`AGENTS.md` §5)。
- 风险: 极低 (只改 constexpr 表, 不碰 kernel 逻辑; 回滚是改回一行)。
- 量具: `test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0`, 形状用现成的
  `hsk=256,hsv=256,nh=4,nr23=[6,1],kv=131072,nb=512` (test-backend-ops.cpp:11315-11316)。
- 上限: **不确定**, 但 Ampere 表在 Volta 上"没有任何针对性"是事实。
- 副产物: 同一张表也服务 decode (nb=1 走 VEC, 不走这张表, 所以不会碰到 decode 回归)。

**M6. 让 K/V 的 shared memory 在 Volta 上双缓冲。**
`fattn-mma-f16.cuh:349-351` 把 nstages 硬绑在 cp.async 上; Volta 没有 cp.async,
但可以用 **寄存器预取 + 显式双缓冲** (1cat README.md:402-445 描述的正是这套)。
文件: `fattn-mma-f16.cuh` (`get_nstages*`、`launch_fattn` 的 smem 尺寸计算)。
中等风险 (要动 kernel 主体), 但比 B1/B2 小得多。

**M7. 减少 `nbatch_combine` 的 66 KiB shared memory 占用。**
`fattn-mma-f16.cuh:1997` `nbytes_shared_combine = 4*32*(128+4)*4 = 67584 B` 是总 smem 的
最大项 (总共 67584 B)。它挤掉 occupancy, 而 Volta 的 `tile_stride = nbatch_2 + 4` 又因为没有
swizzle 而浪费 bank。改 `nbatch_V2`/`nbatch_combine` 的取值可以直接换 smem/occupancy 平衡
(与 M5 是同一张表里的旋钮)。

### 6.3 必须重写 kernel 的大工程 (周级, 不要轻启)

**B1. 复刻 1cat 的 79T 分解: "prefix-block GEMM + block-level online rescale"。**
即: QK 用一次 cuBLAS/CUTLASS GEMM 算 [M = Q*6, N = 24576, K = 256], 物化 score 块到
workspace, 再做 PV as GEMM [M, N=256, K=24576] 并**累加**进 FP32 前缀累加器, 每 24576 个
KV token 只 rescale 一次 (`prefill.cu:6284-6306`, `2060-2096`)。
代价:
- 需要在 `src/llama-graph.cpp` 里把 FA 节点换成"QK op + rescale op + PV op"三段,
  或者在 ggml 里加一个新 op; 这是一个**新的子系统** (红线: 大改动先问用户)。
- score workspace 显存: 前缀 score 块是 `kRows * kBlockN * 2 B` = 49152*24576*2 = **2.4 GB**。
  1cat 是靠 `cudaMemGetInfo` + 复用 cache 做的 (`prefill.cu:6064-6077`)。**这是最硬的约束。**
- 数值: 1cat 自己踩过坑 -- 早期 zero-shift exponential + unguarded FP16 累加会溢出真实输入
  (`README.md:37-43`), 最后必须做 stride-8 采样 max + FP32 累加。

**B2. 写一个 Volta 原生的 D256 split-D FA kernel。**
即 1cat 的 `sm70_fa2_d256_prefill_pipeline.md:55-72`: `BLOCK_M=64, BLOCK_N=32`, 4 个 D64 slice,
8 warp 组成 4 个两 warp 组, 每 warp 8 个 Q 行, warp-pair 共享 N32 P tile, 每 warp 拥有 D128
输出, V 用 Volta 的 TT PV 映射, 两个交替的 PV 寄存器 fragment, 45568 B 动态 smem,
255 寄存器/thread, 1 CTA/SM。
这是一篇 66 KB 的优化史 (`sm70_flash_v100_prefill_operator_optimization.md`) 加数月的
pipeline search。**在我们这里等于从零写一个新 kernel, 不在"最小可行"范围。**

**B3. 把 HMMA 用到 MMQ 里。** `mmq.cuh:189-194` 明确把 Volta 排除在 mma data layout 之外,
Volta 的 MMQ 是 dp4a。补一条 Volta HMMA 的 MMQ 路径 = 重写 `mmq.cuh` 的 tile 体系。
但注意: **prefill 的 ne11 >= 64 根本不走 MMQ** (4.4), 所以这条对 prefill **没有直接收益**,
只对 ne11 in 9..63 的投机验证批有用 (已被 `AGENTS.md` §1 判定为"天花板 1.36x")。

---

## 7. 建议主代理下一步做的三件事 (按性价比)

1. **[10 分钟, 零风险]** 用 `-ts 1/1/1/1` (4 卡) 复跑同一个 256K prompt, 与 `-ts 1/1/1`
   比 attention 相关的那一段。若 4.6 的推导成立, 应该看到接近 1.5-2x 的改善。
2. **[1 小时]** 做 `-ub 512 / 2048 / 4096` 三点扫 (同一 prompt), 验证 4.5 的
   "attention 与 ub 一阶无关, 其余项随 ub 改善" 的预测。
3. **[1-3 天]** M5: 给 `fattn-mma-f16.cuh:113-126` 补一张 Volta 的 D=256 表, 用
   `test-backend-ops perf -o FLASH_ATTN_EXT` 扫 `nbatch_fa` / `nthreads` / `ncols`。
   这是本 fork 已经验证过的模式 (C4/C5), 风险最低、量具现成。

---

## 8. 未验证清单 (不要当成结论用)

- **U1** 3 卡切分 25/25/50 的推导没有运行日志佐证 (纯源码算术)。需要一次
  `llama-server -v` 的 per-tensor device 分配 dump。若模型用分开的 q/k/v 张量,
  结论方向不变但具体数字要重算。
- **U2** 一次项 295 s 里 GEMM / GDN / dequant / allreduce 各占多少, 没有 profiler
  (ncu/nsys 在本机封死)。上面 5.3 的拆分全部是算术估计。
- **U3** attention 总量对 ub 的一阶无关性是推导, 未实测。
- **U4** FA 的 19.85 TFLOP/s 是 `type_K/V = f16` 的实测; 生产用 q8_0 KV 时还要加转换开销,
  所以生产数字 **<= 19.85**。另外那次测量是在 GPU3 独占地跑, 但同机 0/1/2 正在被主代理的
  llama-bench 满载, **没有排除功耗/时钟串扰**, 建议空闲时复测一次。
- **U5** 1cat 的 61.09 / 42.77 TFLOP/s 是文档值, 不是本会话复现; 我们与 1cat 不同机、不同模型
  (FP8 vs Q8_0)、不同 batch 形状, 3.1x 是**跨源对比**。
- **U6** 248901-token 的两条日志 (`370.28 / 367.33 tok/s`) 对应的确切启动参数我**没有从日志里
  拿到命令行**; 只能说它是本项目 l3 口径的 256K prefill。
- **U7** 1708-token 那条只出现在未命名臂的 `/tmp/*.log` 里, 未确认与 248901 同配置;
  两段拟合因此有偏。

---

## 9. 复现命令 (本会话实际用过的)

	# FA 实测 (GPU3 独占; GPU 0-2 当时被别的 job 占满)
	ssh -o BatchMode=yes root@192.168.50.235 \
	  'cd /root/llm/test/v100-opt/llama.cpp/build-nccl/bin ; \
	   CUDA_VISIBLE_DEVICES=3 nohup ./test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 \
	   > /tmp/faperf.txt 2>&1 < /dev/null &'

	# 我们自己的 256K prefill 原话 (两条)
	grep -h -e prompt.eval.time /tmp/l3.log

注意: `test-backend-ops perf -o MUL_MAT` 跑不完 (测试用例上千个, Q8_0 的
`m=8192,n=512,k=5120` / `m=1024,n=4096` 排在很后面), 本会话已把它 kill。
要做 GEMM 归因的话需要另想办法 (例如给 test-backend-ops 加形状过滤, 或写一个最小 driver)。
