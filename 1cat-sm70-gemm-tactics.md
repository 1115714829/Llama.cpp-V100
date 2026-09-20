# 1cat-sm70-gemm-tactics.md — 1cat-vLLM 的 SM70 GEMM 战术表，与 llama.cpp 的对照

> 目的：把 1cat-vLLM 在 Volta 上做量化 GEMM 的**具体战术**提取出来，作为移植到 llama.cpp
> 的 `ggml/src/ggml-cuda/mmq.cu` / `mmq.cuh` 的依据。日期 2026-09-20，base b11053。
> 全部结论带 `文件:行号`，未核实的写"未核实"。

## 1. 1cat 的位置

- 配置模板：`csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/config_sm70_s884.h`
- 战术表（按 MMA 变体分文件）：`.../gemm/kernel/sm70_884_4.cu`、`sm70_884_8.cu`、`sm70_884_16.cu`
  （`884` = HMMA **m8n8k4**，Volta 的基础 MMA 形状）
- 支撑设施：`arch/mma_sm70.h`、`arch/operand_sm70_s884.h`、`arch/smem_copy_sm70.h`、
  `mainloop_sm70.h`、`scheduler_sm70.cuh`、`iterator_sm70.h`

## 2. 模板签名（解码）

`config_sm70_s884.h:39-41`：

```cpp
template <int CTA_M, int CTA_N, int CTA_K, int TG_M, int TG_N, int TG_K,
          class PolicyA, class PolicyB, int Stages, bool SplitK,
          int GroupSizeU = 1, int GroupSizeV = 1, int TILE_C_M_ = -1,
          int TILE_C_N_ = -1, int GmemLookahead = 1, bool FullTiles = false>
struct Type
```

- `CTA_{M,N,K}` = CTA 级 tile；`TG_{M,N,K}` = 线程组划分（warp 数 = TG_M * TG_N）
- `Stages` = 流水线级数；`SplitK` = 是否 K 方向切分累加
- `TILE_C_{M,N}` = epilogue 的输出 tile；`GroupSize{U,V}` = 分组量化（MoE/grouped）的粒度
- `CHUNK_K = lcm(lcm(GroupSizeU, GroupSizeV), CTA_K)`（`config_sm70_s884.h:61-62`）

## 3. 战术表实例（`kernel/sm70_884_4.cu:64-79`，逐行照抄）

```
Type<128, 256, 16, 2, 4, 1, D, D, 2, true, 1, 128, 128, 128>
Type<128, 128, 16, 2, 2, 1, D, D, 2, true, 1, 128,  64, 128>
Type<128, 128, 16, 2, 2, 1, D, S, 2, true, 1, 128,  64, 128>
Type< 96, 128, 32, 2, 2, 1, D, S, 2, true, 1, 128,  48, 128>
Type< 64, 128, 32, 2, 2, 1, D, D, 2, true, 1, 128,  32, 128>
Type< 64, 128, 32, 2, 2, 1, D, S, 2, true, 1, 128,  32, 128>
Type< 64, 128, 16, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>
Type< 64, 256, 16, 1, 4, 1, D, S, 2, true, 1, 128,  64, 128>
Type< 32, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128>
Type< 32, 256, 32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>
Type< 16, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128>
Type< 16, 256, 32, 1, 4, 1, D, S, 2, true, 1, 128>
Type<  8, 128, 64, 1, 4, 1, D, S, 2, true, 1, 128>
Type<  8, 128, 32, 1, 4, 1, D, S, 2, true, 1, 128>
Type<  8, 256, 64, 1, 4, 1, D, S, 2, true, 1, 128>
Type<  8, 256, 64, 1, 4, 2, D, S, 2, true, 1, 128>
```

（`D` = `cache_policy::Default`，`S` = `cache_policy::Stream`，见 `sm70_884_4.cu:12-13`）

## 4. 提炼出的 6 条战术（"技术亮点"）

1. **形状专用（exact-shape）内核 + 环境变量开关 + 可回滚**
   `sm70_884_4.cu:18-30` 的 `ExactMKernelImpl` 用 `desc.m == ExactM` 做可行域判定；
   `:49-56` 的 `Qwen38Nvfp4W13TailN64KernelImpl` 更进一步 `m/n/k` 全精确匹配，且由
   `getenv("VLLM_SM70_NVFP4_QWEN38_MOE_FAST_PREFILL")` 控制 —— 注释明说 "provide an operational rollback"。
   > **指导思想：宁可为一个具体形状写一个内核，也不让通用表在该形状上退而求其次；但必须能一键关掉。**
2. **M 从 8 到 128 全覆盖的战术表**：最小的 `CTA_M = 8`（第 13-16 行）——即**小 batch/decode 方向也有专用 tile**，
   而 llama.cpp 在小 batch 直接不走 MMQ（转 `mul_mat_vec_q`）。
3. **K tile 偏大**：16 / 32 / **64**。K=64 出现在 M=8 的行里 —— 小 M 配大 K，把 K 维的 smem 复用拉满。
4. **Stages = 2（双缓冲）**：全表统一 2，没有 3/4 级深流水 —— 省 smem 换更大的 tile。这与 Hopper/Blackwell
   库（`gemm_universal_sm90_v4.h:250` 用 4 级）形成对比。
5. **SplitK 普遍开启**（`true`）：K 方向切分 + 归并，提升小 M 大 N 情形的并行度。
6. **线程组形状随 tile 变**：`TG_M × TG_N` = 2×4 / 1×4 / 1×2 → warp 数 8 / 4 / 2。
   即**不是固定 256 线程**，而是按 tile 选线程数以提高 occupancy。

## 5. 映射到 llama.cpp：mmq 的配置机制

### 5.1 配置字段（已核实）
`mmq.cuh:206` 的宏定义给出字段顺序：
```cpp
#define CASE(type_, nthreads_, occupancy_, I_, J_, sram_layout_, K_vram_, stream_k_, fallback_)
```
对照 `mmq-config-ampere.cuh:157-164` 的 Q4_K 行：
```
CASE(GGML_TYPE_Q4_K, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
CASE(GGML_TYPE_Q4_K, 256, 1, 128,  16, ..., true, true);
...
CASE(GGML_TYPE_Q4_K, 256, 1, 128,   8, ..., true, false);
CASE(GGML_TYPE_Q4_K, 256, 1, 128,  16, ..., true, false);
CASE(GGML_TYPE_Q4_K, 256, 1, 128,  24, ..., true, false);
...
```
- 即 `nthreads=256, occupancy=1, I=128`，**J 是变化量**：`fallback=true` 的行只给粗档
  {8,16,32,64,128}；`fallback=false` 的行给细档（8 起步、步长 8 直到 128）。
- 访问器与结构体见 `mmq.cuh:291-359`。

### 5.2 shared memory 账（部分核实，公式待补完）
- `extern __shared__ int data_mul_mat_q[];` —— 内核入口 `mmq.cuh:890`。
- **激活（y）部分**：`mmq.cuh:1391` `const size_t nbs_y = config.J * (sizeof(block_q8_1_mmq));`
- `sizeof(block_q8_1_mmq)` = **144 字节**（`mmq.cuh:27` 定义、`:56-57` 断言；`mmq.cuh:118` 注释
  "128 8-bit ints == 32 32-bit ints + 4 32-bit scales"，且等于 `4 * sizeof(block_q8_1)`）。
  -> J=128 时激活部分 = 128 × 144 = **18 KiB**。
- 但 `mmq.cu:310` 注释写明 **"MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise."**
  -> 18 KiB ≠ 48 KiB，说明**权重（sram_layout）那一块才是大头**，`ggml_cuda_mmq_get_sram_stride()`
  （`mmq.cuh:134-152`）与 `K_vram`/`I` 共同决定其大小。
- **待补完**：把权重部分 smem 的公式算出来，才能知道 ampere 表在 Volta 上究竟用了多少、还剩多少。
  这是动 `mmq-config-volta.cuh` 之前的**前置条件**（不能凭"Volta 有 96 KiB"就动手）。

### 5.2b 算出来了：ampere 表在 Q4_K 上已贴着 Turing 的天花板

常数（全部已核实）：
- `MMQ_TILE_NE_K = 32` —— `mmq.cuh:116`
- `QK8_1 = 32`、`QR8_1 = 1`、**`QI8_1 = QK8_1/(4*QR8_1) = 8`** —— `ggml-common.h:258 / :125 / :124`
- 于是 `sram_stride(Q8_1) = 2*32 + 2*32/8 + 4 = **76**`
  （校验 `76 % 8 == 4`，与 `mmq.cuh:154` 的 static_assert 一致）

smem 公式（`mmq.cuh:1390-1392`）：
```
total = nbs_ids + nbs_x + PAD(nbs_y, nthreads*4)
nbs_x = config.I * sram_stride * 4        // mmq.cuh:429（权重，与 J 无关）
nbs_y = config.J * 144                    // mmq.cuh:1391（激活，sizeof(block_q8_1_mmq)=144）
```

Q4_K 在 ampere 表里用 `SRAM_LAYOUT_Q8_1`、`I = 128`（`mmq-config-ampere.cuh:157+`）：

| 项 | 计算 | 大小 |
|---|---|---|
| 权重 `nbs_x` | 128 × 76 × 4 | **38.0 KiB（固定）** |
| 激活 `nbs_y` @ J=8 | 8 × 144 | 1.1 KiB |
| 激活 `nbs_y` @ J=64 | 64 × 144 | 9.0 KiB |
| 激活 `nbs_y` @ J=128 | 128 × 144 | 18.0 KiB |
| 合计 @ J=128 | 38 + 18 (+ids+padding) | **约 56 KiB** |

对照硬件上限：**Turing 64 KiB / block，Volta 96 KiB / block**。

**结论（定量）**：ampere 表在 Q4_K / J=128 上已用到约 **56 KiB**，几乎顶满 Turing 的 64 KiB；
而 **Volta 手里还有约 40 KiB 完全没用上**。这也解释了 `mmq.cu:310` 的"至少 48 KiB，
否则退 BLAS"（38 KiB 权重 + 小 J 的激活 ≈ 39~48 KiB 量级，吻合）。

### 5.2c Volta 表的可调旋钮（t3 的具体设计方向）
按 1cat 战术 3/4/6，墙上能动的量：
1. **加大 `I`**（权重 tile 行数）：I 128 -> 192/256 ⇒ 权重 38 -> 57 / 76 KiB。
   Volta 装得下，Turing 装不下 —— 这正是"分表"的理由。
2. **加大 `J`**（每 block 列数）：激活部分线性长，每 +32 列约 +4.5 KiB；Volta 可把 J 从 128 往上推。
3. **`K_vram` / `MMQ_ITER_K` 加深**（对应 1cat 的"更大 K tile / 更多 stage"）。
4. 参考 `sram_stride` 的其它布局（如 `Q2_K` = `2*32+32+4 = 100`）按类型选更合适的布局。

### 5.3 V100 的量化 matmul 分派图（全部已核实）+ 两个可落地候选

#### 5.3.1 分派顺序与边界（`ggml-cuda.cu:1865-1877`，按序判定，先命中先返回）
```
should_use_mmf   -> mul_mat_f     (FP16 路径)
should_use_mmvq  -> mul_mat_vec_q (GEMV)     <- 在 MMQ 之前判！
should_use_mmq   -> mul_mat_q     (dp4a MMQ)
                 -> cublas
```
相关常量与条件：
- `MMVQ_MAX_BATCH_SIZE = 8` —— `mmvq.cuh:3`
- `MMQ_DP4A_MAX_BATCH_SIZE = 64` —— `mmq.cuh:8`（注释：*"Max. batch size to use for dp4a MMQ kernels
  when FP16 tensor cores are available"*）
- `mmq.cu:334`：NVIDIA 上 `return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;`
  V100 **有** FP16 tensor core -> 上限 64
- `mmq.cu:308-317`：`devices[id].smpbo >= 48 KiB` 才允许 MMQ；`smpbo = prop.sharedMemPerBlockOptin`
  （`ggml-cuda.cu:341/348`）— V100 约 96 KiB，Turing 64 KiB

**于是 V100 上的实际窗口：**

| ne11（每批 token 数） | 走哪个 kernel | 常量 |
|---|---|---|
| 1 | `mul_mat_vec_q`（GEMV） | — |
| **2 ~ 8** | **mmvq** | `MMVQ_MAX_BATCH_SIZE = 8` |
| **9 ~ 63** | **MMQ（dp4a）** | `MMQ_DP4A_MAX_BATCH_SIZE = 64` |
| >= 64 | FP16 MMA / cublas | 同上 |

-> **MMQ 在 V100 上只吃 `ne11 ∈ [9, 63]`。**
-> 这解释了为什么 L0（`-p512` 的 ne11=512、`-n128` 的 decode ne11=1）**两个都不命中 MMQ** ——
   MMQ 类改动**必须**用 L1 的 `-b/-ub` 造出 9~63 的批次来验证。

#### 5.3.2 候选①：`mmq-config-volta.cuh`（smem 余量）
见 §5.2b/5.2c：ampere 表在 Q4_K 上已用约 56 KiB（权重 38 KiB 固定 + 激活），
顶到 Turing 的 64 KiB；Volta 有 96 KiB。旋钮：`I` 128->192/256、加大 `J`、加深 `K_vram`。
约束（已核实）：`I % warp_size == 0`（`mmq.cuh:1456`）、`nthreads % warp_size == 0`（`:1403`）、
`nty = ceil(nrows_x/I)`（`:1412`）、`ntx = ceil(ncols_max/J)`（`:1413`）—— 改 I/J 不改变数学结果，只改变分块。

#### 5.3.3 候选②（最上游友好、先做）：V100 的 **mmvq↔mmq 交叉点**

`ggml_cuda_should_use_mmvq`（`mmvq.cu:324+`）里上游**已经为 5 个具体架构**调过交叉点，每个都带
"tuned on <硬件>" 注释 —— 这是本项目能做的**最符合上游既有模式**的改动：

| 架构 | 分支位置 | 调过的取值 |
|---|---|---|
| Ada Lovelace（RTX 4090） | `mmvq.cu:330-338` | Q2_K `<=4`；Q3_K `<=6` |
| Blackwell（RTX 5090） | `mmvq.cu:340-350` | Q2_K/Q3_K/Q4_K `<=5`；Q5_K `<=6`；Q6_K `<=7` |
| DGX Spark GB10 | `mmvq.cu:352-358` | Q2_K `<=6` |
| Jetson Orin | `mmvq.cu:360-370` | Q2_K/Q3_K/Q4_K/Q5_K/Q6_K `<=1` |
| （AMD CDNA / CDNA1 等） | `mmvq.cu:372+` | 各自一小组 |
| **V100（GGML_CUDA_CC_VOLTA）** | **无** | **落回默认 `ne11 <= MMVQ_MAX_BATCH_SIZE = 8`（`mmvq.cuh:3`）** |

注释原文（`mmvq.cu:327-328`）：*"k-quants cost more to decode and mvq redoes that per column,
so MMQ wins sooner. Only list quant-types MMQ supports, others would fall back to cuBLAS."*

**为什么这是首选**
- 两个目标模型都是 **K-quant**（Q2_K_XL 快速迭代 + Q4_K_M 生产），正是上游认定"该更早切 MMQ"的那类。
- 改动形状 = **再加一个 `if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA) { switch(type) {...} }`**，
  与上面 5 个块同构（几行），不新增机制、不动 kernel。
- **不能照抄 Ada 的数字**（不同架构）。正确做法是先**实测**：见 5.3.4 的 L1 扫掠脚本
  `l1-sweep.sh`（变 `-ub` 改变 prefill 的 ne11，找 mmvq/mmq 的真实交叉点），再按测得值填常量，
  并像上游那样注明 "tuned on Tesla V100"。


#### 5.3.4 L1 验证矩阵（两个候选共用）
用单卡 + `llama-bench -b/-ub` 扫出 `ne11 = 2,4,6,8,12,16,24,32,48,63,64,96` 这些点
（`-ub` 控制 ubatch；`-b` 控制 batch），对比两个变体（同源 + md5 自检）。
两种量化各跑一遍：**Q2_K_XL（9.14 GiB，单卡）** 与 **TurboFCFusion Q4_K_M（17.23 GiB，2 卡）**。

**建议顺序**：先做候选②（小、快、有上游先例），再做候选①（大、需要设计 smem 预算）。


### 5.4 t3 首选配置（下一步照此实现，勿再重新推导）

**前置疑点：已解决（2026-09-20 核实）**
原本担心"放大到 >64 KiB 需要 opt-in，而 mmq 可能没做"。核实结果：**upstream 已经做了**：
- `mmq.cuh:1409-1410`：`CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<type, J, false>), nbytes_shared)` 与 `...<..., true>` —— 两个 kernel 都按**算出来的 `nbytes_shared`** 设了动态 smem 上限；
- `mmq.cuh:1430` / `:1459`：launch 时把 `nbytes_shared` 作为动态 smem 传入；
- `mmq.cuh:1492`：`if (mmq_get_nbytes_shared(config, cc) > smpbo) ...` —— 超出设备上限时**有兜底处理**（不会静默失败）。
⇒ 在 Volta（smpbo≈96 KiB）上使用 ~75 KiB 的配置**是合法且有保护的**。t3 可直接实现。

**首批候选：只改 `I`（其余沿用 ampere 表）**

| 项 | ampere（现状，Q4_K） | volta 首选 |
|---|---|---|
| nthreads / occupancy | 256 / 1 | 不变 |
| **I** | 128 | **192** |
| sram_layout / K_vram | Q8_1（stride 76）/ MMQ_ITER_K | 不变 |
| 权重 smem = I×76×4 | 38 KiB | **57 KiB** |
| 合计 @J=128（+18 KiB 激活） | ~56 KiB（顶 Turing 64） | **~75 KiB → 超 64、≤96，真正的 Volta-only 配置** |

- 约束满足：192 % 32 == 0（`mmq.cuh:1456`）、nthreads % 32 == 0（`:1403`）。
- 兜底：若 75 KiB 导致 launch 失败或掉到不可接受的 occupancy，退 **I=160**（5 warp；权重 47.5 KiB，合计 ~65 KiB，仍超 Turing）。
- 验证方法：**只能用 L1**（`llama-bench -b/-ub` 造 ne11 = 9..63），对比现基线 `libdir-volta2`。
  **L0 不适用** —— `-p512` 走 MMF/FP16 路径，`-n128` 的 decode 走 MMVQ，两个都不命中 MMQ。
- **未实测前不得声称收益**；先把"能不能跑起来 + 跑起来快不快"分开确认。

### 5.5 架构分派的已核实缺口
- 已有按架构分文件：`mmq-config-{ampere, pascal-dp4a, pascal-older, gcn, cdna, rdna2, rdna3, rdna3-5, rdna4, blackwell}.cuh`
  —— **唯独没有 `mmq-config-volta.cuh`**。
- `mmq.cuh:252-253`（host）与 `:279-280`（device）：
  ```cpp
  if (ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) {
      return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
  }
  ```
  即 **Volta(700) / Turing(750) / Ampere(800+) 共用同一张 `mmq-config-ampere.cuh`**，
  而该文件内**没有** `__CUDA_ARCH__` / `CC_TURING` 分支（扁平表）。
- **推论（需 5.2 补完后才能定量）**：该表必须对 **Turing 的 64 KiB/block** 可行，而 **Volta 允许 96 KiB/block**
  -> Volta 至少少用约 32 KiB 的 smem 预算。对应 1cat 战术 3/4：**更大的 CTA_K / 更多 stage**。
- 上游确有为架构分表的先例（blackwell 独立成表），所以新增 Volta 表符合既有模式。

### 5.6 t3 结论：**按 `I` 调 MMQ 是死路（已实测，代码已回退）**

2026-09-20 实测（Q2_K_XL，单卡 GPU0，`-p 512 -n 32 -r 3`，同源 libdir 法）：

| 配置 | ne11=4（对照，走 MMVQ） | 16 | 24 | 32 | 40 | 48 |
|---|---|---|---|---|---|---|
| **I=128**（上游 / ampere 表） | 96.58 | 204.40 | 257.43 | 307.78 | 342.19 | 371.22 |
| I=160 | 96.58 | 204.19 | 257.36 | 307.64 | 341.35 | **368.62** |
| I=96 | 96.73 | **崩溃** | 崩溃 | 崩溃 | 崩溃 | 崩溃 |

- **I=160**：对照组（ne11=4，走 MMVQ）**完全一致** ✓（说明改动确实生效），但 MMQ 区间**无收益、
  且随 J 增大而变差**（ne11=48 -> **−0.70%**）。
- **I=96**：**硬崩**（`SIGABRT`，exit 134，abort 在 `llama_context::process_ubatch`）——
  **`I` 不是自由参数，存在下限**（共享表用的 128 就是下界）。
- **机制解释**：`nty = ceil(nrows_x / I)`（`mmq.cuh:1412`）。nrows_x=5120、I=128 -> **只有 40 个 block**，
  而 V100 有 **80 个 SM** -> **一半 SM 空着**。加大 I 只会让 block 更少、更差。
  ⇒ **"Volta 多出来那 ~32 KiB smem 没被用上"是真的，但它不是瓶颈** —— 这个区间 MMQ 是
  **并行度受限**，不是 smem 容量受限。
- **代码已回退**：`mmq-config-volta.cuh` 删除；`mmq.cuh` 三处（include / 宿主派发 / 设备派发）全部还原。
  回退后 UB=8 -> **137.99**（= C5 的 MMQ 行为）、UB=32 -> **307.86**（= volta2 的 307.78）→ **C4/C5 行为完好**。
- **顺带得到的方法学细节**：回退后 `libggml-cuda.so` 的 md5（`b4b46773`）**不等于**原 volta2 的
  （`9902c7c3`），尽管源码一致 -> **本机构建不是逐字节确定的**。所以：
  **md5 相等**是"同一份代码"的强证据；**md5 不等**只能说明"可能不同"，**不能单独当作变体确实不同的证据**
  （必须配功能证据：kernel 名、时间、控制组）。

**下一步方向（留给以后，勿重复 I 这条路）**：既然瓶颈是并行度（40 block / 80 SM），
可考虑**增加 block 数**——例如给 Volta 用**更小的 J**（`ntx = ceil(ncols_max/J)`：ne11=16 时 J=8 -> ntx=2 -> 80 block）。
但 J 由"查表匹配 >= ne11 的最小 J"决定，需要改**查找逻辑**（比改常量侵入），要先设计再动。

## 6. 待核实 / 下一步
- [ ] **前置**：算出权重部分 smem 公式（`get_sram_stride` × `I`/`K_vram` 的关系），得出 ampere 表在
      Volta 上的实际 smem 占用 -> 确认"余量"确实存在且有多少。
- [ ] llama.cpp 到底在哪些 batch 区间走 MMQ（`MMQ_DP4A_MAX_BATCH_SIZE` 等阈值）-> 决定改动值不值得。
      已知线索：QWEN.md §0.2 记"prefill→cublas、decode→mul_mat_vec_q"，MMQ 只吃中间批次。
- [ ] 设计 `mmq-config-volta.cuh`：只对 Volta 生效，先只改最常用类型（Q4_K/Q2_K/Q8_0/Q4_0），
      用 `-p/-b/-ub` 造 batch 8~256 在 L1 验证（L0 的 -p512/-n128 不命中 MMQ，**不能**用来验这个）。

