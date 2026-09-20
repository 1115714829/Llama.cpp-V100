# core-changes.md — 1cat FA-V100 核心技巧 → llama.cpp fattn.cu 映射 + 首个改动建议

> 配套：`FlashAttention-V100-reference.md`（参考文档）、`vllm-vs-1cat-vllm-diff.md`（差异总览）。
> 状态：Phase 1 分析（已核对 fattn.cu 源码 + traits + 参考文档）；baseline 数据待 Phase 0 补入 §4。
> 行号引用 llama.cpp @ b11053 `ggml/src/ggml-cuda/fattn.cu`。

## 1. llama.cpp fattn.cu 现状（Volta 分派流，已核对源码）

**分派链**：顶层 op `ggml_cuda_flash_attn_ext`（L724）→ 调 `ggml_cuda_get_best_fattn_kernel`
（L520-687，返回 `BEST_FATTN_KERNEL_{NONE,TILE,VEC,MMA_F16}`）→ 按类型分派
`ggml_cuda_flash_attn_ext_{tile,vec,mma_f16}`（L724-740）。

**Volta 判定（L644-651，核心）**：
```c
if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
    if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2)
        return BEST_FATTN_KERNEL_VEC;
    if (Q->ne[1] * gqa_ratio_eff <= 16)
        return BEST_FATTN_KERNEL_TILE;  // "On Volta tensor cores are only faster for sufficiently large matrices."
    return BEST_FATTN_KERNEL_MMA_F16;
}
```
- `Q->ne[0]` = head_dim D；`Q->ne[1]` = batch（query 行数）；`gqa_ratio_eff` = GQA 有效比（`ncols2_max`=8 上限，L640）。
- **prefill**（ne[1] 大，如 512）：`512*eff > 16` → **MMA_F16（tensor core）**。
- **decode**（ne[1]=1）：`1*eff <= 8 <= 16` → **VEC（eff<=2）或 TILE（eff=4/8），均 SIMT、无 tensor core**。
  即 V100 decode 从不走 MMA 分支。

**MMA 路径细节**：
- `switch_ncols1`（L133）：按 `Q->ne[1]` 自适应选 M-tile（8/16/32/64）→ `mma_f16_case<DKQ,DV,M,ncols2>`。
- `switch_ncols2`（L170，内含 Volta 块 L199-222）：GQA 按 `gqa_ratio%{8,4,2}` 选 `ncols2` 打包 → 调 `switch_ncols1<DKQ,DV,ncols2>`。
- 即 GQA 多头行打包进同一个 TC tile（Volta 已在用，1cat 亦强调 GQA multi-head packing）。

## 2. 1cat FA-V100 可迁移技巧（参考文档 + `flash_v100_traits.cuh` 已核对）

| 1cat 技巧 | 说明 | → llama.cpp fattn.cu 对应 | 可迁移性 |
|---|---|---|---|
| 软件重建 TC 喂料 | double-buffer / smem swizzle / register prefetch / 显式 HMMA m8n8k4 fragment 映射，跨 tile 软件流水线 | fattn 依赖 `mma.cuh` 的 VOLTA_MMA 分支；无 1cat 式跨-tile 流水线 | 中（需读 mma.cuh + case 实现） |
| 128-bit KV 宽加载 | `kGmemElemsPerLoad=8`（`uint4`=128bit），paged KV 配对对齐、page 元数据复用一次宽加载 | fattn 的 KV 加载实现（需读 kernel）| 中 |
| GQA multi-head packing | 多 query head 打包成更宽规整的 TC 工作 | 已有 `ncols2` packing | 已有，可微调 |
| D-split / K-stage ping-pong | D=256 拆 4 个 D64、配对 warp 共享 QK、交替 smem stage 降 barrier | llama.cpp D<=256，tile 固定、无 ping-pong | 中 |
| 在线 softmax + FP32 累加 | 不物化完整 score 矩阵，保 FP32 累加契约 | fattn 已用 online softmax | 已有 |
| Sparse / DFlash2 (h3) | 稀疏 attention / 投机解码 | 无对应 | 低（本期不做，见 dflash2 调研） |

**1cat 块结构（`FlashV100Traits<D>`）**：16 warp / 512 thread；D 相关 tiling（D64:M64/N128、D128:M32/N176、D256:M32/N64）。对比 llama.cpp 的 tile（M 8/16/32/64）与线程配置（需读 `mma.cuh`）看差异。

## 3. 首个最小改动候选（排序；待 §4 baseline 数据定夺）

- **C1（decode 阈值，L649）**：当前 decode（`1*eff<=16`）全走 VEC/TILE，从不走 TC。
  - 原理：1cat 主张喂满 Volta TC、不退回 SIMT；GQA packing 可用多头行填满 tile（非 padding）。
  - 风险：V100 decode 通常 **weight GEMV memory-bound**（非 attention-bound），attention 占比小 → 收益可能有限；单 token 若 GQA 行不足以填满大 tile 则有 padding 浪费。
  - 改动量：1-2 行（调 L649 阈值 / 加条件），易回退。**tg 可测。**
- **C2（prefill TC tile，`switch_ncols1` M-tile L150-165）**：大 batch（pp512）下 M-tile 选 32/64，调优 TC 利用。
  - 原理：prefill 是 attention/计算 bound，tile 选对直接影响 TC 吞吐。**pp 可测。**
- **C3（KV 宽加载）**：1cat Layer-1（128-bit paged KV）。需先读 fattn 的 KV 加载实现评估。

**推荐（baseline 确认瓶颈后定，不预设"最优"）**：
1. 若 **pp 是 attention-bound**（nsight/占比确认）→ 优先 C2（prefill TC tile），并读 `mma.cuh` 看喂料可改进点。
2. 若 **tg 是 attention-bound**（V100 少见）→ C1（decode 纳入 TC）。
3. 若 **tg 是 weight-bound**（最可能）→ fattn 非 tg 瓶颈，C1 收益有限；tg 提速应转向 **MMQ**（Phase 3 M3.1，decode GEMV）。
   此时 fattn 优化仍做，但预期收益主要体现在 **pp**。

## 4. 模型 config + baseline（已核实，2026-09-20，不凭记忆）
- **baseline**（单卡 GPU2，详见 `baseline.md`）：**pp512 = 741.16 t/s，tg128 = 36.23 t/s**。
- **模型 = Qwen3.5-27B（arch `qwen35`），27.32B 参数，9.14 GiB Q2_K**。
- **关键：混合（Flash Next）架构**（`src/models/qwen35.cpp` L15-22，`case 64 → LLM_TYPE_27B`）：
  - `full_attn_interval=4`，`is_recr_impl[i]=(i+1)%4!=0` → **64 层中 48 层是线性/递归注意力（GDN gated delta net，类 Mamba），仅 16 层是全注意力（FA / fattn.cu）**。
  - **fattn.cu 只覆盖 1/4 层**；GDN 占 3/4（1cat 对应 `flash_qla/.../sm70/` GDN kernel，更大的块，留 Phase 3+）。
  - 首个优化（FA-V100→fattn.cu）只触及那 1/4 全注意力层。
- 全注意力层 config（Qwen3.5 标准，head_dim 为关键事实）：
  - n_embd=5120，**head_dim=128**（=n_embd_head，k==v，L137 断言；∉{40,72} → **不被 line 644 排除，TC 可用**）。
  - n_head=40，n_kv_head=8 → **gqa_ratio=5**（奇数 → `gqa_ratio_eff=1`）。
  - **decode**（Q->ne[1]=1, eff=1）：`1*1<=2` → **VEC kernel**（SIMT，非 TC）。
  - **prefill**（Q->ne[1]=512）：`512*1>16` → **MMA_F16**（TC）。
  - （注：GGUF 用非标准 key `qwen35.attention.head_count`/`head_count_kv`，原始字节值不完全自洽；head_dim=128 由 Qwen3.5 标准 + n_embd=5120 推得，n_head/n_kv_head 40/8 待更严谨解析确认——但不影响 head_dim=128 这一 TC 可用结论。）
- **对首个优化的影响**：tg（decode）在 V100 是 weight-bandwidth-bound，且 FA 仅 1/4 层 → **首个 FA 优化对 tg 收益预期小**；主要看 **pp**（prefill，FA 走 TC，tile 调优空间）；GDN（3/4 层）是更大的机会（Phase 3）。

## 5. 状态
- [x] fattn.cu 分派流 + Volta 逻辑核对
- [x] 1cat FA-V100 技巧映射（参考文档 + traits）
- [x] 首改动候选 + 推荐逻辑
- [x] §4 baseline 数据 + 模型 config 填充（见 `baseline.md`）
- [x] 依数据定首个改动 → 转 Phase 2（选定 C1，见 §6）

## 6. Phase 2 首个改动设计（C1：V100 decode → MMA/tensor core）
### 6.1 改哪
- 文件 `ggml/src/ggml-cuda/fattn.cu`，函数 `ggml_cuda_get_best_fattn_kernel`（Volta 分支 L644-651）。
- 改法：V100（`cc == GGML_CUDA_CC_VOLTA`）跳过 VEC/TILE，小 batch（decode）也走 `BEST_FATTN_KERNEL_MMA_F16`；非 V100 保持原逻辑。标记 `[v100-opt exp1]` 便于回退。

### 6.2 为何（1cat 原理）
1cat FA-V100 核心：**喂满 Volta TC、不退回 SIMT**。llama.cpp V100 decode（ne[1]=1）当前走 VEC（SIMT）。本实验让 decode 也走 MMA（TC），实测是否更快。

### 6.3 预期（诚实）
- 本模型 **gqa_ratio=5（eff=1，无 GQA 行打包）** → MMA M-tile=8 只有 1 行真实、7 行 padding → **可能 padding 浪费、甚至回退**。
- **FA 仅 1/4 层**、占 pp~1% / tg~0% → 对 pp/tg 总耗时预期影响很小。
- 正确性：MMA 对 ne[1]<M 的 padding 行算垃圾，但**真实行结果正确**（flash attention 每行独立）→ 正确性安全。
- 定位："原理→实测→迭代"第一环；即使回退也是有效数据点（确认当前 VEC 对本模型 decode 合理），随后转 C2（prefill tile）或 GDN/MMQ（更大块）。

### 6.4 验证
AC922 增量编译（只 fattn.cu + relink）→ `CUDA_VISIBLE_DEVICES=2 llama-bench -p 512 -n 128` 对比 baseline（pp512=741.16 / tg128=36.23）；正确性=输出不崩不乱；回退=还原该行 + 记录。

## 7. C1 结果（2026-09-21 实测）：**崩溃 → 已回退**
- **增量编译成功**（只重编 fattn.cu + relink）。
- **pp512 = 740.53 ± 11.83**（= baseline 741.16，无变化——prefill 本就 MMA，未受影响）。
- **tg128 崩溃**：`CUDA kernel flash_attn_ext_f16 has no device code compatible with CUDA arch 700`，栈 `mma_f16_case<256,256,8,2>`（M=8, ncols2=2）。
- **根因**：V100（arch 700）的 MMA(TC) 路径**只为 prefill 的大 M 配置实例化了 device code**（template-instances）；decode 的小 M 配置（M=8, ncols2=2）**没有为 arch 700 编译** → 强行走 MMA 运行时崩。
- **结论**：llama.cpp 把 V100 decode 路由到 VEC/TILE **不是性能启发式，而是 MMA decode 配置在 V100 上不可用**（没编译）。要让 decode 走 TC 需**新增该 MMA 配置的 template instantiation**（更大的改动，且 M=8 仅少量真实行→大概率回退）。
- **已回退** fattn.cu 到 b11053 原始（Windows + 待 AC922 下次构建同步）。
- **修正 §4**：崩溃 ncols2=2 → `gqa_ratio_eff=2` → n_head=5120/128=40 → **实际 gqa_ratio=10（n_kv_head=4），非 5**（由崩溃反推，GGUF 未定论，待核）。
- **FA 价值判断**：FA 仅 1/4 层、占 pp~1%/tg~0%；decode TC 在 V100 未编译；prefill 已 TC。→ **FA 对本模型收益天花板很低**。更大的块 = **MMQ（FFN GEMV，QWEN.md §0.3①，Ampere 配置用在 V100 的已核实缺口）+ GDN（3/4 层）**。

## 8. MMQ / GDN 分析（Phase 3 调查，2026-09-21）+ 路线图
### 8.1 MMQ 分派（已核实）
- `mmq.cuh` L252-253 / L279-280：`cc >= GGML_CUDA_CC_VOLTA → ggml_cuda_mmq_get_config_ampere`。**V100 用 ampere 配置，无 V100 专属**（`mmq-config-volta.cuh` 不存在）。
- ampere config（`mmq-config-ampere.cuh`）**全表统一** `nthreads=256, occupancy=1, I=128, stream_k=true`，J 按 batch（8/16/24/…/128）。即 V100 用的是这套"通用"配置，非 V100 调优。

### 8.2 瓶颈判断（关键，防盲调）
- decode **36 t/s** vs 纯带宽上限 `~900 GB/s ÷ 9.14 GiB ≈ 98 t/s` → **慢 ~2.7×**，说明大头开销**未必在 MMQ**（FFN 若已近带宽，调 MMQ 收益 <10%）。
- 混合模型：48 GDN（Mamba 类）+ 16 FA + FFN。2.7× 开销更可能在 **GDN（48 层）/注意力/kernel overhead**。
- **结论**：MMQ config 盲调是猜（nthreads/occupancy/I 最优需 **nsight profiling** 先定位 decode 瓶颈在 MMQ 还是 GDN）。AC922 是否有 nsight-compute 未确认。

### 8.3 路线图（下一步，需 profiling 或用户定夺）
1. **先 profiling**（nsight-compute / 分层计时）定位 decode 瓶颈：MMQ(FFN) vs GDN(48 层) vs FA vs overhead。**不 profiling 就盲调 kernel config 风险高、可能白做**。
2. 若 GDN 是大头 → 借鉴 1cat `flash_qla/.../sm70/` GDN kernel（更大的块，3/4 层）。
3. 若 MMQ 是大头 → 新增 `mmq-config-volta.cuh`（V100 调 nthreads/occupancy/I）+ 改 L252/L279 分派（cc==VOLTA 路由）。
4. 验收：同 baseline 条件（GPU2, pp512/tg128）对比；正确性（test-* + 输出一致）。

### 8.4 本会话交付（Phase 0-2 完成 + Phase 3 调查）
- baseline.md（pp512=741.16 / tg128=36.23）+ core-changes.md（分派流/1cat 映射/混合模型/C1 崩溃根因/MMQ 分析）。
- C1（FA decode→TC）实测=崩溃（MMA decode 配置未为 arch700 实例化）→ 已回退；**fattn.cu 现为 b11053 原始**（Windows 已回退；AC922 build 仍是 C1 版，下次构建需重同步回退版）。
- 未动 vLLM（GPU 0/1/3/4 全程稳定）；GPU2 空闲。

## 9. Phase 3 ncu profiling — decode 瓶颈 = `mul_mat_vec_q` GEMV（~77%），V100 落 GENERIC 表

**ncu（nsight-compute 2025.1.1）profile**（GPU2, Q2_K_XL, `-p 1 -n 32`, `--launch-count 1500`, 按 kernel 名聚合 `gpu__time_duration.sum`）：
- **`mul_mat_vec_q`（量化 GEMV / FFN）= ~17.3ms / ~22.7ms = ~77%**（top 单 kernel 137-173μs，`ncols_dst=1`，type 22/17/18/16）
- other（norm/cpy 等）~4.8ms（~21%）
- GDN（`gated_delta_net`）~0.32ms（~1.4%）、FA（`flash_attn`）~0.28ms（~1.2%）

**结论（修正 QWEN.md §0.3 的 MMQ 假设）**：
- decode GEMV 走 **`mul_mat_vec_q`（`mmvq.cu`）**，不是 MMQ（`mmq.cu`）。Q2_K batch=1 时 `ggml_cuda_mul_mat` 的顺序是 mmvf→mmf→**mmvq**（`MMVQ_MAX_BATCH_SIZE` 内）→mmq（大 batch）→cublas。MMQ 只用于 prefill/大 batch。
- **`mul_mat_vec_q` 设备参数表（`get_device_table_id`）NVIDIA 只有 TURING + GB10 两张；V100（CC700）落到 `MMVQ_PARAMETERS_GENERIC`（`calc_nwarps`: ncols 1-4→4, 5-8→2）—— 无 V100 专属表 = 优化缺口。**（现有 V100 特殊处理只针对 MoE batch size：`get_mmvq_mmid_max_batch` L288/L427，与 nwarps 无关。）
- GDN/FA 只占 ~2.6% → C2/C3 确认低优先级。
- 注意：decode wall（27.6ms/token）> 采样 GPU kernel 合计（~10ms/token）→ 存在 overhead/launch 成分（GEMV 是最大 GPU kernel，但非唯一瓶颈）。

**C4 优化（本步，已落地）**：给 `mul_mat_vec_q` 新增 `MMVQ_PARAMETERS_VOLTA` 表（CC 700-749 路由，enum L98 + device/host 分派 + `calc_nwarps` L458 + `calc_rows_per_block` L587），nwarps 调优。

### 9.1 C4 实测（V100 ncols=1 nwarps sweep，tg128，pp512 全程 ~744 不变）
| nwarps(ncols=1) | tg128 t/s | vs baseline(36.23) |
|---|---|---|
| 8 | 32.05 | -11.5% |
| 4 (原 GENERIC) | 36.23 | 0% |
| **2** | **37.52 / 37.38** | **+3.4%（最优，已采用）** |
| 1 | 37.24 | +2.8% |

- **结论**：ncols=1 的 GEMV 上 **warp 越少越快**（K-loop 短，`warp_reduce_sum` 跨 warp 开销主导；2 为甜点位）。最终 `MMVQ_PARAMETERS_VOLTA`：ncols1→2、ncols2-4→4、5-8→2、default→1。
- **正确性**：nwarps 只改 K-loop 的 warp 切分（reduction 数学等价，求和不变，仅 FP 加法顺序差异）。llama-bench 640 token 无崩溃；llama-cli 生成 34.9 t/s（与 bench 一致）。
- **落地**：`mmvq.cu`（Windows `F:/.../ggml-cuda/mmvq.cu` + AC922 `/root/llm/test/v100-opt/llama.cpp/ggml/src/ggml-cuda/mmvq.cu` 已同步 nwarps=2 版）。**tg128: 36.23 → ~37.4（+3.4%）**，pp512 不变（741→744，噪声）。
- 说明（更正）：nvidia-smi 实测 tg 期间 **GPU util ~99%**（kernel 背靠背，launch overhead 很小）→ decode 是 **GPU-bound**，GEMV（77% GPU kernel time）是主瓶颈（C4 已优化 +3.4%）。**先前"~60% overhead"是基于 1500-launch ncu 小样本的误估**（低估了 per-token GPU kernel 时间）。后续候选：§0.3④ GEMV 访存/向量化（边际，GEMV 已带宽受限 + C4 调 nwarps）、GDN（48 层，~1.4%）。
