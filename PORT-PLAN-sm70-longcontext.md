# PORT-PLAN：SM70 长上下文 attention（把 1cat 的 V100 做法落到 llama.cpp）

> 状态：**待用户拍板**（属"大改动"，AGENTS.md §2 红线要求先问）。
> 依据：2026-09-21 DSH 会话实测（见 `SESSION-2026-09-20-measurements.md` §16），全部结论带 file:line 或实机数字。
> 代码基线：`llama.cpp` HEAD `c2d716519`（base b11053）。

## 1. 为什么必须动 attention

实测（本机，3xV100，Q8_0 权重 + DFlash2，tensor 模式）：

| 场景 | roofline（算出来的下限） | 实测 | 倍差 |
|---|---:|---:|---:|
| 256K **prefill** | attn 13.5 + GEMM 14 = 28 PFLOP => 约 230-290 s | 292 s（896 t/s） | **约 1.0x（已到顶）** |
| 256K **decode**（M=1） | KV 每卡 4.6 GB/token => 约 5.7 ms/token | **54 ms/token**（18.45 t/s） | **9.5x** |
| target 步（M=8/9，8K 上下文） | 权重 9.02 GiB/卡 => 12.1 ms | 32-34 ms | 2.7x |

关键事实（都已核对源码）：

1. **M=1 走 VEC，M=2..8 走 TILE，M>=9 才走 MMA**（`fattn.cu:611` `can_use_vector_kernel`、`:638-642` `gqa_ratio_eff=2`、`:644-652` Volta 分支）。
   => **DFlash2 投机解码（n_max=7 => M 约 8）走 TILE，我此前落地的 D=256 MMA 调参只作用于 prefill**（与实测一致：prefill +11%/+42%，decode 不变）。
2. **TILE 与 MMA 强制 `need_f16_K/V = true`**（`fattn.cu:706-709`）=> 量化 KV 每次调用要把**该层整段 K、V 反量化成 f16**
   （缓冲区在 KQV 的 extra 空间，`fattn-common.cuh:53`；反量化函数 `dequantize_V_q8_0`，`:588`）。
   256K 时每层每步约 1.64 GB => x16 层 **约 26 GB/步**（按头切分每卡约一半）。**VEC 没这个开销**（直读 q8_0）。
3. **上游已有 split-KV，但切分度不随上下文增长**：`fattn-common.cuh:1201-1203` 的 `blocks_num.y = parallel_blocks`（VEC/TILE 的 KV 切分索引；
   内核按 `gridDim.y` 跨步：`fattn-vec.cuh:250-256`、`fattn-tile.cuh:957-973`；归约 `flash_attn_combine_results`，`:917-972`、`:1288-1295`），
   而 `parallel_blocks` 由 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` + wave 效率扫描决定（`:1126-1132`、`:1179-1199`），`:1132` 的 `ntiles_KV` 只是上限。
   => V100 + 256K + M<=8 只有约 **6-26 路并行**，每块仍串行走 300-680 个 KV tile。上游 issue **#28734 "CUDA: decode slows linearly with context"** 仍 open。
4. **VEC 只实例化 `cols_per_block in {1,2}`**（`fattn-vec.cuh:544-572`）、**TILE 内核签名即 `const half2 * V_h2`**（`fattn-tile.cuh:563, 858`）
   => "让 TILE 直接读 q8_0"**不是有界改动**，必须改 staging。
5. **1cat-vLLM 的对标做法**（只读仓库实读）：SM70 默认后端 `FLASH_ATTN_V100`（`vllm/platforms/cuda.py:149-159`），
   decode 用 **partition split-KV**（`flash-attention-v100/kernel/flash_decode_paged.cu:1038, 1061-1076`，`blockIdx.z=partition_idx`，
   partition size 256/512/**1024（seq_len>=32768 时）**，`flash_attn_v100/flash_attn_interface.py:19-20, 599-608`）+ reduce kernel（`:2728`）；
   D256 有专用内核与 build-time 补丁（`cmake/patches/sm70_flash_attn_d256_{splitkv3,pipeline,k_pingpong,gqa_arch}.patch`），
   其中 `k_pingpong` 的注释直接写明"第二级 K 只在 P 生成前活着 => 可别名到 P 区"（smem 双缓冲不额外占内存）。
   KV 只做 **fp8_e4m3 存储**（算前展开 fp16），计算走 **fp16 HMMA**（NVFP4 权重 + marlin SM70）。

## 2. 目标与验收（与 goal 验收标准 2 对齐）

- 主指标：**256K decode** 从 33.97 t/s 提升；分阶段目标 **>= 2x**（=> 68 t/s 以上），理想接近 roofline 的一半（约 90 t/s）。
- 副指标：**32K/64K 投机解码每轮 ms**（用户真实 agent 深度段）下降 >= 20%。
- 不劣化：8K 主口径 tg（当前 96-100 t/s，AL 5.55）不得下降超过噪声（3%）；**greedy sha256 逐位一致**（同配置）。
- 纪律：同源 A/B（同一 build dir、四库 md5 + 标记串）、每配置 >=2 臂、正式数字带 drop_caches、同时报 AL 与 ms/轮。

## 3. 工作流 A（先做，收益最大且最小侵入）：让 KV 切分度随深度增长

**现状问题**：`parallel_blocks` 只看 occupancy/wave，不看 `n_kv`。长上下文时每块串行 300-680 个 tile，延迟无法被并行度掩盖。

**改法**（只动 host 侧策略，风险最低）：
1. 在 `fattn-common.cuh` 的 `launch_fattn` 中，对 VEC/TILE 增加**深度驱动**的候选：
   `parallel_blocks_kv = ceil(n_kv / (nbatch_fa * target_tiles_per_block))`，再取 `max(parallel_blocks_occupancy, parallel_blocks_kv)` 的**上限**（例如 `min(ntiles_KV, 4*sm_count)`）。
   hmm -> 注意不能简单取大：partition 越多，combine 的 `flash_attn_combine_results` 工作量与启动延迟越大。
   => 正确做法是先**扫一遍** `target_tiles_per_block in {64, 32, 16, 8}`，用长上下文标尺选最优点（不改默认行为，用 env 门控）。
2. 如果 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 给的 `max_blocks_per_sm` 已经是瓶颈（VEC 128 线程 + 0 smem => 通常 8-16/SM），
   则说明瓶颈在**每块内部**（串行 KV 循环的访存级并行度），此时工作流 A 无效，直接转 B。

**预期**：VEC(M=1) 54 ms/token -> 15-25 ms/token（2-3x）。判据：`[AR]`/事件无关，只看 `llama-bench -d 262144 -n 64` 的 tg 与每 token ms 斜率。

**验证顺序**：env 门控打开 -> `llama-bench -d {8192, 32768, 131072, 262144}` 对照（NODROP 快速迭代，最后正式口径复测）。

## 4. 工作流 B：消除 TILE/MMA 的"整段 f16 反量化"

**现状问题**：量化 KV 下每次 FA 调用把整层 K、V 展成 f16（256K 下约 26 GB/步），而 VEC 直读 q8_0 没有这笔开销。

**改法（两条，按侵入度排序）**：
- **B1（推荐，侵入中等）**：让 TILE 的 V 通路支持量化直读——把 `dequantize_V_q8_0` 从"预转换"搬到 **smem staging**
  （`fattn-tile.cuh:722` 附近的 `cpy_fattn_kv`/cp.async 之后立即反量化到 f16 smem tile），K 同理（K 本来就要进 smem 供 WMMA 用）。
  这样显存里只有量化 KV，f16 只存在于 smem tile 中，**每步只读一遍量化字节**。
- **B2（备用，更简单但收益小）**：把 `need_f16_K/V` 与 `parallel_blocks` 联动——只有当 `n_kv` 小于阈值时才预转换，超过阈值时**拒绝 FA 并回退到非 FA 路径**
  （`-fa 0` 走 `ggml_mul_mat` 直读量化 KV，无转换）。这是"配置级"的兜底（见工作流 C）。

**预期**：256K 下省掉约 26 GB/步的读写（按 800 GB/s 约 33 ms/步的量级；实际重叠后收益待测）；
32K 下约省 3.3 GB/步（投机路径每轮都要付）。

## 5. 工作流 C（零代码，先看数据）：配置层的三个旋钮

已排队的实测（结果回来填 §6）：
1. **KV dtype**：q8_0 vs f16 —— 短上下文 f16 快 4% 每轮但 AL 低 10%（=> tg 反而差 5%，Round 39 已定论）；
   **但长上下文下 f16 免掉整段转换**，可能反转。若反转 => 用户 256K 场景直接用 `-ctk f16 -ctv f16`（注意显存：256K 时每卡约 5.7 GB）。
2. **TP 度数**：KV 按头切分 => TP4/TP6 线性降低每卡 KV 流量（256K 时 TP3 每卡 4.6 GB/token，TP6 约 2.3 GB）。
   用户已明确"1-6 卡自由、跨岛不是瓶颈" => 长上下文加卡是**零代码**收益。
3. **FA 开关**：`-fa 0` 在长上下文下走 `ggml_mul_mat` 直读量化 KV（无转换、无 TILE 切分限制），代价是物化 KQ 矩阵（256K 时内存不可行，32K/64K 可行）。

## 6. 待填：排队中的实测（Round 40-41）

- `/root/lc-sweep.sh`：`llama-bench -d {8192,32768,131072,262144} x {q8_0,f16} KV x -fa 1`，加 `-fa 0` 诊断（8K/32K）。
- `/root/lc-spec.sh` + `lc-chain.sh`：长上下文**投机**标尺（3 prompt、固定 seed、报 tg/AL/中位数），32K/64K x {q8_0,f16}。
- `/root/lc-chain2.sh`：TP4(q8_0/f16) 与 TP6(q8_0) @64K。

## 7. 顺序与工作量（本机实测编译代价：全量 CUDA 模板重编 30-60 min；只改 host 代码 10-20 min）

> **2026-09-21 更新（Round 42）**：就 goal 验收标准 1（>=180 t/s）而言，**最优先的不是 attention，而是 allreduce 的主机介导往返**
> （§16.12：AR 不在图内，138 次/轮 x 68-95 us = 9.4-13.1 ms/轮，且下一段图依赖它 => GPU 空转）。
> 本文件的长上下文工作流仍然有效（对应验收标准 2 与用户真实场景），但**短上下文主指标的瓶颈排序是：AR(13ms) > draft 侧(13ms) > 图/调度残差(6-8ms)**。

1. 工作流 C 数据 -> **能拿多少先拿**（零代码）。
2. 工作流 A（host 策略 + env 门控，1 次编译）-> 用 `-d` 扫描判优劣；有效则固化默认 + 长上下文标尺复测。
3. 工作流 B1（kernel staging，改动集中在 `fattn-tile.cuh` V 通路与 `fattn-common.cuh` 的转换判定，2-3 次编译）。
4. 若 A/B 仍不够：才考虑自写 D256 splitd 内核（1cat 路线，属新子系统，需再拍板）。

## 8. 明确不做（已证伪，引用 §16 与 HANDOFF §5）

- MoE / 专家小 GEMM 方向（模型是稠密的，`feed_forward_length=17408`，无 `expert_count`）。
- 通过"改 NCCL 参数"或"加卡买带宽"改善短上下文（已实测更差 / 不成立）。
- 让 VEC 直接接管 M=8（`cols_per_block` 只到 2，寄存器与上游设计不允许）。
- `VLLM_SM70_USE_BREAKABLE_CUDAGRAPH`（1cat 实测 -13% ~ -29%，禁止）。
