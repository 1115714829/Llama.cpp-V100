# c5-volta-crossover.md — C5：给 V100 补上 K-quant 的 mmvq↔mmq 交叉点（已落地 + 已实测）

> 日期 2026-09-20，base b11053。这是本项目**第二个落地的优化**（第一个是 C4 = mmvq nwarps 表）。
> 动机：学 1cat-vLLM 的"按硬件/形状特化"思路，落到 llama.cpp 已有的**per-arch 调参模式**上。

## 1. 改了什么

**缺口**：`ggml_cuda_should_use_mmvq`（`ggml/src/ggml-cuda/mmvq.cu`）里上游已为 Ada / Blackwell /
DGX Spark / Jetson Orin 分别调过 K-quant 的 mmvq↔mmq 交叉点（各自带 "tuned on <硬件>" 注释），
**唯独 Volta 没有**，落回默认 `MMVQ_MAX_BATCH_SIZE = 8`。

**改动**（两处，形状与既有 5 个架构块完全同构）：
- `ggml/src/ggml-cuda/mmvq.cuh`：新增常量
  ```c
  // Max. batch size for which to use MMVQ kernels for K-quants on Volta.
  // Measured on V100: MMQ beats MMVQ at ne11=8 (+2.6%) and ne11=6 (+1.1%), MMVQ wins at ne11=4 (-1.2%).
  #define MMVQ_VOLTA_MAX_BATCH_SIZE_K 4
  ```
- `ggml/src/ggml-cuda/mmvq.cu`：在 Orin 块之后新增
  ```c
  if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA) {
      switch (type) { // tuned on Tesla V100
          case GGML_TYPE_Q2_K:
          case GGML_TYPE_Q3_K:
          case GGML_TYPE_Q4_K:
              return ne11 <= MMVQ_VOLTA_MAX_BATCH_SIZE_K;
          default:
              return ne11 <= MMVQ_MAX_BATCH_SIZE;
      }
  }
  ```

含义：V100 上 **K-quant 在 ne11 ∈ [5, 63] 走 MMQ**（原来 5..8 走 MMVQ）；非 K-quant 不变。

## 2. 实测数据（V100 GPU2，Q2_K_XL，`-p 512 -n 32 -r 3`，q8_0 KV + FA，单卡）

`T` = `MMVQ_VOLTA_TUNE_MAX`（调参脚手架，用 `-ub` 把 prefill 的 ne11 固定到 UB）：

| ne11 (=UB) | T=8（MMVQ，即上游默认） | T≤6（MMQ） | 结论 |
|---|---|---|---|
| 4 | 96.71 | 95.39 | **MMVQ 好 1.2%** |
| 6 | 120.53 | 121.80 | **MMQ 好 1.1%** |
| 8 | 133.91 | 137.40 | **MMQ 好 2.6%** |
| 10 | 130.40 | 130.3（全为 MMQ，T 不生效） | — |
| 12 | 155.34 | 155.5（同上） | — |

离散度极小（±0.01~0.08 t/s），所以 1% 的差异是可信的。
**结论：V100 上 K-quant 的最优交叉点 = 4**（ne11 ≤ 4 走 MMVQ，≥5 走 MMQ）。
这与上游给 Ada 的 Q2_K 取值（`<= 4`）**一致**，与 Blackwell（`<= 5`）接近 —— 是独立测出来的，不是照抄。

## 3. 验证（去掉脚手架、硬编码 4 之后）

- **脚手架已移除**：`MMVQ_VOLTA_TUNE_MAX` 0 处、`getenv` 0 处、`#include <cstdlib>` 已删。
- **硬编码生效**（无 env，单卡 GPU2）：
  - `UB=4` -> pp512 = **96.46**（= MMVQ，与 T=4 的 96.38 一致）
  - `UB=8` -> pp512 = **137.65**（= MMQ，与 T=4 的 137.33 一致）
  -> 说明最终代码的行为与调参时一致，不是脚手架残留造成的假象。
- **默认档无回归**（`-p 512 -n 128 -r 3`）：
  | 变体 | pp512 | tg128 |
  |---|---|---|
  | pristine | 729.74 ± 14.74 | 35.58 ± 0.11 |
  | volta2（= C4 + C5） | 727.33 ± 13.44 | **36.49 ± 0.09** |
  - pp512 在 ±14 噪声内（-0.3%）；tg128 的 +2.6% 是 **C4** 的贡献（本 build 建在 C4 之上），
    不是 C5 —— 符合预期：C5 只影响 ne11 5..8，而 `tg128` 的 decode 是 ne11=1。
- **同源有效性**：三个 libdir 的 `libggml-cuda.so.0.24.0` md5 各不相同
  （pristine `3baf1fbc` / c4 `0276aed7` / volta2 `9902c7c3`）。
- 构建二进制在 `/root/libdir-volta2/`（同源快照法见 `same-source-ab.md`）。

## 3.5 在**生产模型 Q4_K_M** 上复测（t9b，2026-09-20，2 卡 GPU0/1，`--tensor-split 1/1`）

同源 A/B：`libdir-pristine`（交叉点 8）vs `libdir-volta2`（交叉点 4，= C4 + C5）。

| ne11 (`-ub`) | pristine | volta2 | 差 |
|---|---|---|---|
| 4（**control**：两边都命中 MMVQ） | 99.58 ± 0.02 | 99.51 ± 0.02 | **−0.07%** |
| 8（volta2 命中 MMQ） | 127.37 ± 0.06 | **146.54 ± 0.04** | **+15.0%** |

- **control 组无差异**，正是机制证据：在 ne11=4 两个 binary 走**同一条分支**（MMVQ），
  所以"唯一差别是交叉点常量"这个前提成立；到 ne11=8 才分叉 -> 差异只能来自换 kernel。
- **Q4_K_M 上的收益（+15.0%）远大于 Q2_K（+2.6%）** —— 印证上游注释
  *"k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner"*：
  位宽越高，MMVQ 每列重复解码解码代价越大，越该早切 MMQ。
- **生产模型正是 Q4_K_M**，所以 C5 对这个项目的实际价值比 Q2_K 基准显示的大得多。
- GPU 0/1 各 2 MiB 收尾（干净退出）。

### 3.6 机制确认（`nsys`，Q2_K_XL，`-ub 8` -> ne11=8）

profiler 说明：`ncu`（`/usr/local/cuda-12.4/bin/ncu`，**不在 PATH 上，必须用全路径**）与 `nsys`
在本机 **profile 18 GB 的生产模型时都会 `failed to load model`**（profile 环境限制，非代码问题）；
换成 9.14 GiB 的 Q2_K_XL 后正常。所以机制确认用 Q2_K_XL 做。

| 变体 | 内在 ne11=8 见到的 kernel |
|---|---|
| pristine（交叉点 8） | **`mul_mat_vec_q` × 36，`mul_mat_q` × 0** |
| volta2（交叉点 4） | `mul_mat_vec_q` × 33 **+ `mul_mat_q` × 3 + `mul_mat_q_stream_k_fixup` × 3** |

- 交叉点下调后，**确实有 3 个 matmul 从 GEMV 切到了 MMQ**，并且带上 MMQ 的 split-k 收尾 kernel
  （`mul_mat_q_stream_k_fixup`）—— 说明走的是完整 MMQ 路径，不是别的偶然差异。
- 同一次运行里其余 33 个仍是 `mul_mat_vec_q`（<=4 的那些），与"只影响 ne11∈[5,8]"一致。
- 同时这也**间接支持了历史结论**：decode（ne11=1 / UB=4）下全是 `mul_mat_vec_q`，没有 `mul_mat_q`。

## 4. 影响范围（诚实说明）

- **不影响**单流 decode（ne11=1 仍是 MMVQ）—— 那是 C4 的领域。
- **影响**以下场景：小 ubatch 的 prefill（`-ub` 5..8）、以及 decode 时单批 token 数落在 **5..8** 的情况
  （例：`--parallel` 多个槽同时解码、或 MTP/投机解码一次送 5~8 个 token 去验证）。
- 收益量级：这些区间里 **+1.1% ~ +2.6%**。属于"小而实"的一类，不是大杀器。
- **未测**：Q4_K_M（生产模型）上的交叉点未单独测；本实验用的是 Q2_K_XL。
  两者都是 K-quant，理论上同族，但**应补测**（见待办）。

## 5. 与 1cat 的关系

1cat 的做法是**按形状/硬件给专用战术**（`sm70_884_*.cu` 的大表 + exact-shape 内核 + getenv 开关回滚）。
C5 是同一哲学在 llama.cpp 里**最轻量**的落法：不写新 kernel、不加新机制，只把"该用哪个现成 kernel"
这台按架构调好 —— 而 llama.cpp 自己已经有 5 个这样的先例。

## 6. 待办
- [ ] 在 **Q4_K_M（生产模型）** 上复测交叉点（2 卡即可，用 `libdir-volta2` vs `libdir-pristine`）。
- [ ] 用 `ncu` 确认 ne11=8 时确实从 `mul_mat_vec_q` 切到了 MMQ kernel（机制确认，而非只看时间）。
- [ ] 评估 `MMVQ_VOLTA_MAX_BATCH_SIZE_K` 对**非 K-quant**（Q4_0/Q8_0 等）是否也该调（本次只动 K-quant，最保守）。
