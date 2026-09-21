# 查漏补缺：外部 V100 参考项目分析（2026-09-21）

> 来源：用户提供的 6 个仓库，克隆在 `F:\vllm+llama.cpp\v100-refs\`。
> 本节是**中期结论**，由子代理调研得出，主线已核对关键数字的出处。

## 1. ⚠️ 先纠正一个诱惑：「单卡 V100 219 tok/s」不是可比的数字

两个仓库各有一个 219，但口径完全不同：

| 来源 | 数字 | 真实口径 |
|---|---|---|
| `v100-skinny` README:34 | 219.1 tok/s | **4×V100-SXM2-16GB TP4 + k=7 投机验证**，26.9 ms/轮，5.89 tok/轮 |
| `ninfer-v100` README:23,54 / docs/v100.md:55 | 218.98 tok/s | **单卡 V100-PCIE-32GB**，但 K=1 投机跑在**合成续写语料**上、99.2% 接受率；
| | | README 自己标注 *synthetic-corpus ceiling, not a general-generation rate*；artifact 本体 23.72 GB，
| | | 按 900 GB/s 上限物理上不可能是普通 decode => 该语料触发了 context-lookup 长验证路径 |

**ninfer 真正可比的单卡数字**（`docs/v100.md:111-117`，varied-context sweep）：

| 配置 | tok/s |
|---|---:|
| 2K 无投机 | 29.28 |
| 2K + MTP K=3 | 67.66 |
| **8K + MTP K=3** | **92.74** |
| 32K | 55.04 |
| 150K | 43.63 |

⚠️ **但这一组数字本身就足够刺眼**：**单张 V100、8K、MTP K=3 = 92.74 tok/s**，
而我们 **3 张 V100、8K、DFlash2 = 99.97 tok/s**。按每卡效率算，我们落后约 2.8 倍。
（口径需再确认：模型同为 Qwen3.8-27B 系；MTP K=3 与 DFlash2 n_max=7 不是同一件事，但量级差摆在这里。）

## 2. 低比特权重：真实存在，但**不是那条大鱼**

两个仓库都**没有绕开 tensor core**，都用 `mma.sync.m8n8k4`：
- `v100-skinny kernels/skinny_kernels.cu:1145`：`MMA_8N8K4` -> `mma.sync.aligned.m8n8k4.row.col.f16.f16.f32`
  做法：**四个 quadpair 全部铺在 N 维、共享同一块 A 激活（QPN）**；4-bit 权重**保持压缩穿过 HBM**，
  在寄存器里直接解包成 mma 需要的 B fragment（nibble 预交织、无 shuffle、主循环无 smem、56 reg、零 spill）。
- `ninfer src/ops/common/volta_mma.cuh:111` `volta_mma_qp_n` 完整复刻了这一套；
  `nvfp4_volta_qpn_gemm.cuh` 头注释直接写 *This is v100-skinny's skinny_nvfp4_qpn2 ... Copy the decoder and scale-cadence choices*。

**实测有效带宽**（作者自测、同 harness 同一次 sitting）：

| 项 | 值 | 占天花板 |
|---|---:|---:|
| 读天花板 | 879 GB/s | - |
| NVFP4 (0.5625 B/权重) M=1 | 679.5 GB/s | 77% |
| NVFP4 M=8 | 619.8 GB/s | 71% |
| lm_head M=1 | 842.9 GB/s | 96% |
| FP8 (1 B/权重) M=1-4 | 718.6-720.5 GB/s | 82% |
| ninfer W8 GEMV T=1 | 752 / 该卡实测 ~794 GB/s | 95% |

### 对我们「HMMA 天花板 1.36x」结论的修正
那条是按 **Q8_0 自身 n=1 屋顶 757 GB/s 的 73%** 推的（`HANDOFF.md:443`，形状还是合成的 m=4096 k=14336，AUDIT E5 已标注）。
**它没错，但口径偏保守**：换成 4-bit 存储后，同一张卡的有效**权重**吞吐是 `679.5/0.5625 = 1208 G 权重/s`，
而 Q8_0 屋顶是 `757/1.0625 = 712 G 权重/s` => **同一张卡「换格式」的权重吞吐上限约 1.70x，不是 1.36x**。

**但端到端上限仍然很小**：权重流只占我们 12.1 ms / 55.5 ms => **低比特权重最多值 +10~13%，不是 2x**。
=> **不要把这条当成主线。**

## 3. ★ 真正的大鱼：两个参考实现都把**主机侧开销压到接近零**

| 参考 | 实测 |
|---|---|
| `v100-skinny`（4 卡） | target verify(G0->G1) **17.48 ms** + drafter 5.85 ms => **整轮 25 ms** |
| `ninfer`（**单卡**，2K 无投机） | **整轮 34.2 ms** |
| **我们（3 卡）** | target 整步 32.3-34.3 ms + draft 13.6 + selector 4.3 => **55.5 ms/轮** |

**我们 3 张卡的 target 整步，约等于 ninfer 1 张卡的整轮。** 低比特权重最多只能解释其中 22%。

两个实现的做法：
- vLLM 侧：**persistent-metadata 投机轮 + 固定形状 CUDA graph**；
- ninfer 侧（架构级规则，`docs/maintainer/engine-architecture.md:489`）：
  *CUDA Graph 按「合法的 exact-B topology」建立，**request identity 与 page IDs 是稳定输入数据，不是 graph key***。

=> **这正对着我们 53% 的 meta 后端主机循环**（每轮 29.4 ms），也印证了我们已闭合的那条链
（图重建 -> uid 重发 -> 属性抖动 -> direct）。**外部先例表明这条路是对的，而且有人已经走到位了。**

## 5. ★★ attention 与服务层线（中期，证据级）

### 5.1 最可能直接解决我们长上下文问题的一条：**GQA read-once**
`sglang-V100` fork 内 `python/sglang/srt/layers/attention/tilelang_fa_v100/`：
- **一个 CTA 吃掉一个 KV 头的全部 6 个 Q 头 => K/V 只读一次**
- K 轴 split-KV：decode 目标 **160 个 CTA**；prefill 尾块 64/32/16/8/4/2 路
- 1 字节 E5M2 KV + 移位转换
- 作者自测：把 **128K decode 从 30.04 tok/s 救回 200K 49.58 tok/s**

**为什么这条对我们特别关键**：我们的模型 **24 Q 头 / 4 KV 头（GQA=6）**，而我们的长上下文实测是
**有效 KV 带宽约 105 GB/s、roofline 800 GB/s**（256K decode 34 t/s）。
若我们的 FA 内核是「每个 Q 头各读一遍同一份 K/V」（llama.cpp 常见组织方式），则 KV 读流量是最小值的 **6 倍**。
**105 x 6 = 630 GB/s，正好接近 roofline** —— 这条线索可以解释我们长上下文的大部分差距。
对照：`flash-attention-v100` 自己**没有**做 GQA read-once（`template.h:58,73` 用 `kv_head_idx = head_idx/(H_Q/H_K)`，6 个 Q 头各读一遍），
而且它的 decode **明确不支持 split-KV**（`fused_mha_forward_kvcache.cu:462` 有硬 `TORCH_CHECK`：`num_splits > 1 not supported now`）。
=> **两个仓库正好形成对照：谁做了 GQA read-once 谁就把长上下文救回来了。**

### 5.2 与我们主机侧问题直接对应的配置纪律：**只捕获单形状 CUDA 图**
`sglang-V100` 的服务参数：`--max-running-requests 1` + **`--cuda-graph-bs 1 --cuda-graph-max-bs 1`**（只捕获单形状图）
+ `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`（规划流与计算流重叠）；KV 量化用 `--kv-cache-dtype fp8_e5m2`。
=> 这正是我们缺的**静态形状纪律**，也解释了为什么他们的图不会像我们这样每轮属性抖动 17,000 次。

### 5.3 D=256 在 Volta 上的两个硬约束（与我们已做的 FA 工作对照）
- `flashinfer-sm70.patch` 新增了 D=256 的 kernel spec：`head_size=256`、`warps_m=1`/`warps_n=2`、**`share_smem_k_v=True`**、`loop_step=16`；
  原文注释：*share_smem=True required - D=256 overflows Volta's 96KB SMEM without sharing*。
- 同 patch 把继承自 Turing 的 **`warps_n=8` 全部改成 `warps_n=2`**（针对 sm70 的 hmma884 重新调参）。
- `flash-attention-v100` 侧同样的结论：D=256 时 smem 合计约 **88.1 KB => V100 上只能 1 CTA/SM**，
  K/V 共用一块 smem 是能放下的**唯一原因**（`include/forward.h:42-61` 的 union 布局）。
=> **我们应核对自己的 FA 在 D=256 下的 `warps_n` 与 `share_smem_k_v` 是否为 sm70 调优值。**

### 5.4 其他可借的点
- `marlin-v100-*.patch`：外部仓库 `zhinianqin/marlin_v100` 的 MoE **W4A16** 内核；并新增 `csrc/sm70_bf16_compat.h`
  给 sm70 补 `__bfloat1622float2` / `__hfma2` 等在 CUDA 12.x 下对 sm_80 以下屏蔽的向量 intrinsic。
  （对我们意义有限：我们的模型是**稠密**的，不是 MoE。）
- `flash-attention-v100` 的一个重要否定性结论（`utils/docs/volta.md:139,143`）：
  **sm_70 上 `nvcuda::wmma` 只有 m16n16k16，且 `ldmatrix` 要 sm_75 => 拿 wmma 直接写 Volta attention 走不通**，必须手写 PTX。
  这与我们早前「自写 WMMA 原型只有 90 GB/s」的实测一致。

## 6. ★★★ sglang-V100 的关键发现表（证据级，全部作者自测）

| 技术点 | 文件:行 | 量化效果 | 可移植性 |
|---|---|---|---|
| **GQA read-once**：一个 CTA 覆盖一个 KV 头的全部 Q 头 | `tilelang_fa_v100/_kernels_paged_decode.py:1-7`（docstring 原文 *One CTA evaluates every GQA query head belonging to a KV head, so K/V pages are fetched once instead of once per query head*）；grid=`(heads_kv, max_splits, batch)` :73-77 | **128K 30.04 -> 200K 49.58 tok/s；TPOT 33.289 -> 20.170 ms** | **高**（思路+常量可抄，实现要为我们重写一版 ggml VEC 变体） |
| **E5M2 KV -> FP16 只左移 8 位、无查表** | `_kernels_paged_decode.py:131-137`（`bits = raw << 8`，注释 *avoids a dependent LUT load on Volta's K-panel critical path*） | 分组 attention 微内核 **0.065/0.186/0.444/0.766 ms @1K/25K/70K/128K**；旧 LUT 路线 128K 约 0.963 ms | 高（任何 1 字节 KV 都能用；前提是 KV 换成 E5M2 类） |
| **D=256 长上下文尾块精确 split-KV** | `_paged_adapter.py:33,95-115`（仅当 logical_dense_kv 且 max_seq_len>=32768 且 num_tokens<=2048；Q<=64/128/256/512/1024/2048 -> splits=64/32/16/8/4/2） | **Q=64, K=245,760, Hq/Hkv=6/1 => 4.306 ms vs 未切分 40.675 ms（9.45x）**，max abs diff 4.8e-7 | **高**（策略与阈值可直接照抄到我们的 prefill 尾块） |
| **反面证据：满 8K chunk 故意不切分** | `_kernels_dense_d256_splitkv.py:1-8`（*two through five splits were all slower than the dense one-CTA-per-query-tile path at Q=4096/K=32768*） | splits 2/3/4/5 全部慢于 dense | 高（**这解释了我们 A4 的负结果**，避免重犯） |
| decode split-KV 常量 | `_kernels_paged_decode.py:17-18`（DECODE_SM_TARGET=80、MIN_TOKENS_PER_SPLIT=64）、:334 `max_splits=ceil(80/(batch*heads_kv))`、:335 `block_n=32 if dim==256 else 64`、:340 `threads=block_m*4` | CTA 目标 80->120 **反而更慢** | 高 |
| D=256 dense prefill 几何 | `_kernels_dense_d256.py:21-23`（BLOCK_M=64, BLOCK_N=32, THREADS=256）、:28-34（QK=FullCol, PV=FullRow）、:15-19（D=256 单独开 fast-math） | GDN prefill 2048/4096/8192 = **1.455/2.199/3.725 ms** vs 旧 2.017/3.236/5.674（**-27.9/-32.0/-34.3%**） | 中（几何可借鉴，TileLang->CUDA 要重写） |
| 长前缀先 gather 成 dense 再算 | `_kernels_dense_d256.py:1-7`（*removing page-table lookup, integer divide, and scattered-page address resolution from every K/V element load*）；阈值 `_paged_adapter.py:30-31`（MIN_QUERY_TOKENS=3920、MIN_CONTEXT=8192） | D256 prefill 路线入口条件 | 中（我们无 paged KV，但「统一 logical 布局再算」可用） |
| **主机侧/小算子级提速（最便宜的一类）** | `benchmark/qwen38_nvfp4_v100_70tps_20260907/README.md:21-28`（逐轮加：63.206->67.646->69.662->70.036->70.108->71.977 tok/s，**+19.4%**）、:88-95（**QSA split-merge 23.69->4.84 us**，in-model 每 attention 层约 **34->5.80 us**）、:122-127（page-table metadata 8192 页 **23.26->2.11 us**）、:150-152（**one-shot push all-reduce 3.35 us**） | **纯主机侧+算子级优化拿到 25K decode +19.4%，attention 内核未动** | **高（对我们 29.4 ms 主机循环最直接）** |

### 6.1 ⚠️ 一条**反驳我们既有结论**的数据：push all-reduce 3.35 us
我们 R1 的结论是「设备侧 push AR 在役 **61.5 us** vs NCCL 53.0 us，更差」并据此**证伪**了这条路。
而 sglang-V100 记录的是 **one-shot push all-reduce 3.35 us** —— **差 18 倍**。
必须查清口径差异（几卡？消息多大？是否含同步？是内核时间还是端到端？）。
**若它成立，R1 的结论需要重审** —— 而不是当作已证伪永久排除。

### 6.2 对我们三条线的直接映射

| 我们的问题 | 对应外部手段 | 作者实测收益 |
|---|---|---|
| 256K decode 34 t/s（KV 带宽 105 GB/s vs roofline 800） | **GQA read-once**（GQA=6 最坏 6x DRAM 流量） | 128K 30.04 -> 200K **49.58** tok/s；TPOT 33.3 -> 20.2 ms |
| 256K prefill 895 t/s / TTFT 293 s | **D=256 尾块精确 split-KV**（按 Q 长度分 64/32/16/8/4/2 路） | Q=64/K=245760 **4.306 ms vs 40.675 ms（9.45x）** |
| 每轮 29.4 ms 主机循环（53%） | **只捕获单形状 CUDA 图** + 主机侧小算子优化 | 主机侧优化单独 **+19.4%**（内核未动） |

## 7. ★★★ 汇总：我们必须补的缺口 + 我们做错了的地方（三路调研终版）

### 7.1 ⚠️ 代码级发现：我们的 DFlash2 verify 路径每轮把整条 KV 反量化成 f16
链路（主线已核对我方行号）：
```
fattn.cu:644-650   Q->ne[1]*gqa_ratio_eff = 8*2 = 16 <= 16  => 8-token verify 走 TILE（不是 VEC）
fattn.cu:706-710   TILE 要求 need_f16_K/V = true
fattn-common.cuh:1026-1088  每次调用对整条 K 和 V 做 to_fp16
```
=> 在 256K 上这是 **读 1.06 B + 写 2 B + 读 2 B** 的 KV 往返，**而且每轮都做**（AL 5.55，每轮 8 token verify）。
另一条：`fattn-vec.cuh:106-111` 的 `head/gqa_ratio` 索引 + `blocks_num.z = ntiles_z_gqa * K->ne[2] * Q->ne[3]`
（VEC 走 ncols2=1 => ntiles_z_gqa = gqa_ratio = 6）=> **同一 KV 头的 6 个 Q 头各读一遍全量 KV**。
**动作（最小、只读诊断）**：在 `ggml_cuda_get_best_fattn_kernel` 里对 `D==256 且 Q->ne[1]>1` 打印选中的 kernel 与 K->ne[1]，
一次 A/B 即可证实 256K verify 上确实发生 TILE + f16 反量化。**这是本轮最便宜、可能最值钱的一条。**

### 7.2 缺口清单（按优先级）

| 优先级 | 缺口 | 证据 | 建议动作 |
|---|---|---|---|
| **P0** | 「HMMA 天花板 1.36x」口径错误，据此排除了整条低比特线 | 1.36x 是 Q8_0 **在自身格式内**的余量（`HANDOFF.md:443`，合成形状，AUDIT E5 已标注）。换 4-bit 后同卡**权重吞吐**上限 **1.70-2.11x**；但端到端仍只有 **+10~13%**（权重流仅占 22%） | 把 AGENTS §1 / HANDOFF §6.3 的表述就地改成「Q8_0 格式内余量；换 4~5bpw 后端到端 +10~13%」。**仍然不是主线** |
| **P1** | 主机侧：**metadata 缓存 + 状态指纹失效**（直接打我们 53% 的循环） | `v100-skinny gpu_model_runner.py:531-536,570-617,675-688`：缓存 metadata 对象（alias 同批 buffer），失效条件是 **`(req_id, mamba_state_idx, block_lens)`** —— 正是 GDN 递归状态视图；实测**稳态 75/97 字段逐字节不变** | ① 照抄 `_sm70_e5_diff_probe` 写 ggml 版（graph 构建对象树逐叶快照 + delta），先拿到我们自己的「哪些 ne/nb 在变、变成什么」清单（纯诊断）② 分两类处理：内容变但形状/地址不变 -> 预分配 + 原地 `copy_`/`fill_`；形状真变 -> 用状态指纹在**输入数据**里表达，不进图 key ③ **不要一上来做整轮单图** |
| **P2** | DFlash2 selector 留在 CPU（4.3 ms/轮 = 7.7%） | `ninfer src/ops/candidate_selector/bf16/dflash2_selector_volta.cu:29-112`（约 80 行有效代码，1 block/row、256 线程）+ `candidate_selector.h:11-45` 完整 K=1..15/B=1..8 契约与精确数学 | 按契约写 ggml 等价 device kernel；**RNG 必须换 counter-based**，否则图重放会漂 |
| **P3** | 没有 4-bit TC 路径；且误以为必须新增量化类型 + 加载时重排 | `nvfp4_volta_qpn_gemm.cuh:21-35,105-115`：**不重排权重**，而是对激活施加同一置换（两次 `__byte_perm`）——*the permutation is a relabeling of the reduction*；`q4_volta_qpn_gemm.cuh:36-40`：Q4 nibble 序**天然就是 B-fragment 序**，0 条 pack 指令 | 先做 q4_0/q4_K 的 QPN decode kernel（**复用现役 GGUF 布局，不动格式**），激活侧置换按 ggml 的 j/j+16 交织重推 `__byte_perm` 常量；geometry 逐形状扫，不要全局默认 |
| **P4** | 缺「每多一张卡到底买到多少」的体检 | ninfer 单卡（1-token prompt、varied 语料、INT8 KV）2K 无投机 29.28 / MTP-K3 67.66 / 8K 92.74 / 32K 55.04。我们 3 卡 99.18 | 同一份语料跑我们自己的 no-spec / DFlash2 / 单卡 三臂。**若远小于 3x，allreduce + 图管理就是净损耗，应先修 P1 而不是加卡** |
| **P5** | 缺「k 随 context 扫描」 | `v100-skinny results/ctx_depth_20260819.md`：4 卡 k=7 从 0.5k 的 127.4 掉到 65k 的 54.7（2.33x），同臂 no-spec 只掉 1.32x；**k=3 在 65k 达 76.26，比 no-spec 的 65.46 还快 16.5%** | k 是**每请求杠杆**，不是 boot 参数 |

### 7.3 ★ 我们可能做错了的地方（七条，按严重度）

1. **自写 WMMA 原型（90 GB/s、慢 6.2x）选错了 API 与映射，却用它否定了整条路线。**
   `flash-attention-v100 include/mma_m16n16k16.h:5-16` + `utils/docs/volta.md:139,143` 明说：sm_70 上 `nvcuda::wmma` **既没有 ldmatrix（sm_75 才有）也没有 swizzle**。
   `v100-skinny docs/qpn_race_notes.md:30-34` 记录他们的 v1/B_ring 也是因 **K 切分**而死；**赢法是 QP 切 N、A-stationary**。
   => **我们继续在 m16n16k16 + smem 暂存上做原型，等于重复他们的死路。**
2. **用 Q8_0（8.5 bpw）作唯一生产格式，把 1.73x 的字节差当成不存在。**
   每卡字节：我们 **9.67 GB/卡**（3 卡）；v100-skinny **4.21 GB/卡**（4 卡，close accounting）；ninfer 单卡 23.72 GB 全模型（MLP 是 NVFP4）。**per-rank 字节数是它们的 2.3 倍。**
3. **「图属性抖动只能靠静态形状/整轮单图根治」过于悲观。** v100-skinny 用「缓存 metadata + 状态指纹失效 + 原地改写」就修掉了，并量化到「稳态 75/97 字段逐字节不变」。
   我们连「哪些字段在变」的系统清单都还没有（有差异探针，但没有 per-field byte-diff 报告）。
4. **KV 只做精度选择，没做旋转。** ninfer 对 INT8 group-64 KV 的 K 与 Q 都做 **D256 归一化 Hadamard**（我们 head_dim 恰好 256）。
   反向证据同样有力：FP8 KV 在 SM70 上掉到 scalar paged，**代价 +4.82 ms/轮**；而权重 QPN2->QPN8 只 +1.08 ms/轮 => **KV 的 dtype 比权重的 dtype 更值钱。**
5. **归因工具是二等公民。** ninfer 把 `decode_host_exposed_seconds` / `decode_device_wait_exposed_seconds` 写进公开计时契约；我们在源码里留 7 处 env-gated `fprintf`。这直接导致要花几天才能定「21.8 ms 提交窗口」。
6. **selector 留在 CPU。** 我们并行化（23.45->4.3 ms）就收工；ninfer 有现成 GPU 版本，把 7.7% 主机时间变成 0，且顺带 graph-safe。
7. **「瓶颈在 VEC/TILE 的有效带宽只有 105 GB/s、远低于 roofline 800」可能把「冗余流量」当成了「低效」。**
   若同组 6 个 CTA 的重复读有一部分落到 DRAM，真实 DRAM 流量是 **105 x (2~6) = 210-630 GB/s**，已贴着 V100 实用 roofline。
   => **先测真实 DRAM 流量或做 grid.z=H_kv 的对照实验，再决定要不要重写内核。**

### 7.4 可复核性说明（子代理已标注举证缺口）
`skinny_kernels.cu:1390` 引用的 `results/qpn_matrix_20260817.csv`、ninfer 的 `bench/ops/nvfp4_qpn2_splitk_sweep.cu`（标注 deleted）等数字**无法在仓库内复核**。
可复核的是 `results/kernel_matched_20260819.csv` + `benchmarks/kernel_matched_bench.py` 这一对。**引用时请只用可复核的那部分。**

## 8. ★★★★ 最重要的一条：`-sm layer` vs `-sm tensor` —— 纯启动参数，作者自测 **+21.5%**

`jusko-llama-volta-qwen3flash` 的 `docs/examples/windows-dual-v100/bench-results.jsonl`（**同一 session、除 split 外参数逐字相同**，2xV100，Qwen3.8-Flash-Next Q5_K_M，F16 KV，`-c 65536 -b 2048 -ub 1024`）：

| 臂 | TG-64@2048 | PP-60k | GPU util |
|---|---:|---:|---|
| `V00-frozen`（**tensor**） | 28.11 t/s | 644 | 43.5 / 45 |
| `V01-layer64k`（**layer**） | **34.16 t/s（+21.5%）** | **887（+37.7%）** | **23.5 / 30（减半）** |

作者原话（`docs/fork-benchmarks.md:69-71`）：*the tensor-split AllReduce + Meta path was the tax*。
GPU util **反而减半**说明它不是 GPU-bound —— **与我们的诊断（meta 主机循环占 53%）是同一个病，但我们开错了药**。

### 为什么这条对我们特别重要
我们的实验清单里**全是「在 tensor 模式内部省时间」，没有一条是「换掉 split 模式」**。
且 AGENTS §4.8/E3 已记录：`pipeline_parallel` 只在 `split_mode==LAYER && !has_tensor_overrides()` 时为真，
而我们的 harness **永远传 `--tensor-split`** ⇒ **我们从未测过 layer split 下的同步/主机语义**。
harness 本身就支持（`SPLIT=${SPLIT:-tensor}`），所以这**是零代码、一臂就能出结果**的实验。

⚠️ 注意：这条路会改变同步语义（`pipeline_parallel` 变真），必须用 §4.2 的 md5 自检 + 轮计时探针看 `enqueue` 与 `decode+sync` 的分解。
若成立，**我们那 53% 的主机账本直接作废一半**（不是优化它，而是绕开它）。

## 9. 其他必须补的缺口（jusko / xllama，证据级）

| 优先级 | 缺口 | 证据与量化 | 可移植性 |
|---|---|---|---|
| **P0-1** | **把 q8_0 KV 的 decode/verify 注意力搬上 Volta 张量核** | `fattn-q8-volta.cuh`（336 行，独立 kernel）：把 q8_0 在 **smem 里反量化成 f16 再喂 `mma.sync.m8n8k4`**；HBM 只读 q8_0。实测 q8 子核 **2.674->1.419 ms @101k、6.879->3.363 ms @260k**；四件套同 fork A/B **TG 27.591->36.471 t/s（+32.19%）**，生成 token SHA 一致。旁证：xllama 明说 V100 上 `planar3/iso3 比 turbo3 慢，因为前者走 VEC（循环内反量化）而后者反量化成 f16 + MMA` | 中（kernel 硬编码 T=4/QH=24/KVH=4；我们是 DFlash2 T=8，需按 T=8 重写分块） |
| **P1-3** | D256 提示注意力**常量**差异 + 2-CTA 紧凑核 | 我们 `fattn-mma-f16.cuh:124-128`（combine=128, nstages=2）vs 他们 `:124-129`（combine=64, nstages=1 的 32 列版；V2=64,combine=128,nstages=2 的 64 列版）—— **只差 6 个常量**。他们的 2-CTA 紧凑核（`:2005-2291`，门控 `:2221-2251` 要求 `gqa_ratio==6 && D=256`，**正是我们的形状**）自测 **+13.11%** | 高（常量）/ 中（紧凑核要从零写） |
| **P1-4** | GDN x4 预填充路径（我们 48/65 层是 GDN） | `gated_delta_net.cu:172` 起 `gated_delta_net_cuda_128x4_volta`（S_v=128、每 warp 4 个独立状态列、共享 Q/K/g 载入）；孤立 kernel **1642->1067 us**；整模型 **+2.32% PP（8k）/ +2.03%（16k）**；V100 **40/40 正确** | 中高（新增 167 行、自包含，需按我们 H/S_v 复核） |
| **P2-5** | MMVQ「每列重复解码权重」（我们所有量化类型都有） | `mmvq.cu:562-609` 的 x4 版把权重解码提到 4 列循环外；上游是逐列调用 `vec_dot` | 中（他们没有我们量化类型的 x4 版） |
| **P2-6** | `RMS_NORM + SCALE` 融合 | `norm.cu:556` + `ggml-cuda.cu:4226`；自述 *removes 480 extra launches*；无速度声明 | **高**（小改动、与模型无关；host-bound 下任何 launch 削减都是纯赚） |
| **P3-7** | **KV 压缩：结论是现在不要做** | 我们 ctx 8192、16 层全注意力、4 KV 头、D=256 ⇒ 单序列 KV 约 256 MiB(f16)/136 MiB(q8_0)，而每轮权重流 29 GB/3 卡 ⇒ **KV 只占每轮流量约 1%**。xllama 自测 V100 上代价 −6%（turbo2）到 −22%（planar4/iso4）；其多卡长上下文实测 200k：上游 q8_0 **16.92** vs 本分支 **11.15** vs turbo3 **7.46 t/s**（decode 侧负收益）。且 `planar*/iso*` 被该仓库**自己的 PPL 表**判死（3.03-3.06 vs f16 1.0023 **+203%**） | 低（只留 turbo2/turbo3 备用） |

### 9.1 两条被这份报告纠正的既有结论
1. **「V100 上 HMMA 天花板 1.36x」被过度外推了。** 那是关于 **MMQ 权重矩阵乘**的结论。
   两个 fork 都证明**量化 KV 的注意力在 V100 上恰恰应该走张量核**。我们 `fattn.cu:704` 的注释
   （*On Volta tensor cores are only faster for sufficiently large matrices*）正是被它们反驳的那句。
   ⇒ 「不要重做 HMMA **权重**路线」仍成立；「不要在 V100 上用张量核做**注意力**」不成立。
2. **`--spec-draft-n-max 7`（T=8）让我们与整个 T=4 专用解码栈不兼容。**
   jusko 的全部解码加速（q8 TC kernel、Q5_X4、Q6_W4R4）都是为 n-max=3 / T=4 写的。
   ⇒ **kernel 级 T 特化是我们完全没利用的维度**，值得算一笔账：n=7 的 AL 增益 vs T=8 核比 T=4 核慢多少。

### 9.2 一个重要的否定性结论：图抖动这条线**只能我们自己解**
逐行核对：jusko 的 `ggml_cuda_graph_update_required`（`ggml-cuda.cu:2631-2671`）与我们 `llama.cpp:3036-3069` **逻辑完全一致**
（先比 `cgraph->uid`，再逐节点 memcmp `node_src_ne` / `node_src_nb` / `node_src_data_ptrs`）；
它的 `ggml-backend.cpp` 只有三处改动（moe-cache 失效、`n_pipeline_copies` 暴露、MoE 预取），**没有 split 缓存 / uid 复用的新招**；
xllama 用的是更老的上游。⇒ **不要再去外部找这条线的捷径。**

### 9.3 方法论副作用（记一笔）
这两个仓库都是 **depth=1 的浅克隆**，`git log`/`git diff` 在仓库内做不出 fork 层 delta。
本次是靠本地 llama.cpp 克隆里存在它们声明的 merge base（上游 `465e49b9c`），`git archive` + `git diff --no-index` 才得到可信的 24 文件 delta。**以后分析这类 fork 先确认能否拿到 merge base。**

## 10. ★★ P0-2 实测（Round 131）：**layer split 更慢，但它把奖金量化出来了**

`/root/layer-ab.sh`，官方口径（NPRED=512，`CARDS=0,1,2`，`L=libdir-instr`，`P2P=1`，交错四臂）：

| 臂 | MEDIAN_TG | AL(p1/p2/p3) | ms/轮(p1) | `splits` | `alloc_us` | `enqueue_us` | greedy sha256 |
|---|---:|---|---:|---:|---:|---:|---|
| **lyA (layer)** | **70.74** | 5.16/5.21/6.62 | **77.3** | **4** | **40,059** | **1,309,965** | `ccc284e4...` |
| **tsA (tensor)** | **100.61** | 5.55/4.22/6.38 | **55.2** | 2 | **620,946** | **5,431,093** | `f3edac19...` |

### 10.1 判决：**layer split 在本配置下慢 40%（77.3 vs 55.2 ms/轮），不采用**
jusko 的 +21.5% 在**我们没有复现** —— 那是 2xV100 + Qwen3.8-Flash-Next Q5_K_M 的口径。
原因清楚：layer split 下 3 张卡**串行**跑各自的层块（GPU 时间 ≈ x3），省下的主机时间补不回来。

### 10.2 ★ 但这次实验**独立证实了主机侧奖金的规模**（这是真正的收获）
把两条 `[RT]` 折算成每轮：

| 项 | tensor | layer | 差额（= meta 后端的税） |
|---|---:|---:|---:|
| `alloc` | 2.11 ms/轮 | 0.14 ms/轮 | **1.97 ms/轮** |
| `enqueue`（`graph_compute` 主机窗口） | 18.5 ms/轮 | 4.71 ms/轮 | **13.8 ms/轮** |
| **合计** | | | **约 15.8 ms/轮** |

**这与我们之前算的「meta 后端主机循环 29.4 ms/轮」是同一笔钱**，现在有了第二条独立路径的印证：
**绕开 meta 后端能省约 15.8 ms/轮**。而 layer split 因为付出约 22 ms/轮的 GPU 串行代价，净亏 6 ms。

=> **结论：奖金是真的（15.8 ms/轮），但 layer split 是错的收法。**
   正确的收法是**让 tensor-parallel 本身变便宜**（减少 meta 后端的设备调用次数 / 每次调用开销），
   也就是回到 **P0-1（q8_0 KV 张量核注意力）** 与主机侧 metadata 缓存那条线。

### 10.3 两个附带事实
- layer 模式下 greedy sha256 = `ccc284e4...`，与 tensor 的 `f3edac19...` **不同** => 两种 split 的数值路径确实不同（与早前 TP4 出现同一个 hash 一致），
  所以跨 split 模式**只能比 ms/轮，不能比 AL/tg**。
- layer 的 `splits=4`（tensor 是 2），且即便绕开 meta，`enqueue` 仍有 4.71 ms/轮 —— 说明还有非 meta 的主机成本。

## 11. ★★★★ 依赖关系分析（Round 136，用户提问触发）：**哪些结论是「有前提」的**

用户问：*有没有一种前后依赖 —— 有了前者后者才会提速？如果是这样，之前否掉的东西可能是被前提卡住了。*
**答案是：有，而且不止一条。** 下面把 layer A/B 的完整数据（四臂跑完）与依赖链一起给出。

### 11.1 layer A/B 完整结果（四臂，离散极小）

| 臂 | MEDIAN_TG | ms/轮 | AL(p1) | `alloc_us`/轮 | `enqueue_us`/轮 | **主机合计/轮** | greedy sha256 |
|---|---:|---:|---:|---:|---:|---:|---|
| lyA (layer) | 70.74 | 77.0 | 5.16 | 0.14 ms | 4.72 ms | **4.86 ms** | `ccc284e4...` |
| lyB (layer) | 70.78 | 76.9 | 5.16 | 0.14 ms | 4.72 ms | **4.86 ms** | `ccc284e4...` |
| tsA (tensor) | 100.61 | 55.1 | 5.55 | 2.11 ms | 18.48 ms | **20.59 ms** | `f3edac19...` |
| tsB (tensor) | 100.84 | 55.0 | 5.55 | 2.12 ms | 19.51 ms | **21.63 ms** | `f3edac19...` |

- 离散：layer 0.06%、tensor 0.23%；同 split 的 `sha256` 逐位一致。**判决稳固：layer 慢 40%。**
- **主机差额 = 21.6 - 4.9 = 约 16.8 ms/轮**；而轮时差额 = 77.0 - 55.1 = **21.9 ms（layer 更差）**。
  => 反推 **layer 的 GPU 串行时间 ≈ 72 ms vs tensor ≈ 33.5 ms（约 2.15x）**，与「3 卡串行」预期一致。

### 11.2 ★ 由这组数据得到的**可加性模型**
四条臂都满足 **轮时 ≈ 主机 + GPU串行**（55.1 ≈ 20.6 + 34.5；77.0 ≈ 4.9 + 72.1）。
**含义**：主机与 GPU **几乎没有重叠**，两者都在关键路径上、且近似**线性叠加**。
=> 推论：**任何一毫秒的主机节省或 GPU 节省，理论上都应该 1:1 地体现在轮时上。**

### 11.3 ⚠️ 那么这个模型**与 R1 的结论直接矛盾**
R1（设备侧 push AR）：**主机省了 4.8 ms/轮，但轮时零改善** —— 我们据此把「AR 的 us 级微优化/设备侧 push」永久排除了。
而按上面的可加性模型，省 4.8 ms 主机**应该**带来 4.8 ms 轮时改善。**两者只能有一个对。**

**这个矛盾必须解决，因为它决定我们所有『已证伪』清单的可信度：**
若可加性成立 => R1 的测量有问题（或那 4.8 ms 落在会重叠的部分），**R1 需要重审**；
若可加性不成立 => 主机节省不一定兑现，**那么所有『省了主机但没动轮时』的结论都要重新解释**。

**最便宜的判定实验**：重跑 R1 的 A/B，**同时开 `GGML_META_HOST_TIMING`**。
若 `[META] total` 确实下降约 4.8 ms/轮而轮时不动，则 R1 的异常为真；若 `[META] total` 根本没降，则 R1 当年的『省 4.8 ms』本身就不成立。
**无论哪个结果，都比现在这种『两个互相矛盾的结论并存』要好。**

### 11.4 真正的依赖链（用户问的那类）

**链 1（最重要）：q8_0 KV 的价值 ⇐ 需要 q8_0 的张量核注意力（P0-1）**
- 我们的结论「KV dtype 保持 q8_0（32K 中性、64K f16 仅 +4%）」是在**没有 q8_0 TC 内核**的世界里测的。
  而 jusko 同 session 实测：**无 TC 内核时 q8_0 KV 比 f16 慢 3.8%**（另一处 -4.9%）—— 因为它要付逐元素反量化税。
- 更糟的是**我方 verify 路径（T=8）走 TILE，每轮把整条 q8_0 KV 反量化成 f16**（`fattn.cu:644-650 -> 706-710 -> fattn-common.cuh:1026-1088`）。
- **=> P0-1 是前提：装上 q8_0 TC 内核后，「KV 用不用 q8_0」这个结论可能翻转**，长上下文带宽也会跟着变。**这条链是真的。**

**链 2：TP3 最优 / 加卡不买带宽 ⇐ 依赖当前的主机税**
- 「TP3 最优、TP4/TP6 更差」是**在 meta 后端主机税随卡数增长的前提下**测的（TP6 的 enqueue 高达 62 ms/轮！）。
- **主机税正是我们要消灭的东西。** 一旦消灭，『加卡』的收益函数就变了 —— 1cat 的 4 卡 + 整轮 fullgraph 正是这个 regime。
- **=> 「加卡没用」是条件结论，不能当成永久结论。**

**链 3：n_max=7 最优 ⇐ 依赖当前内核的 T 特化**
- jusko 的整套解码加速（q8 TC kernel、Q5_X4、Q6_W4R4）**全是为 T=4 / n_max=3 写的**。我们的 T=8 用的是为别的形状调的核。
- **=> 「n_max=7 最优」是在当前内核条件下的结论**；换了 T 特化内核，最优 n 可能移动。

**链 4：D256 MMA 常量（P1-3）⇐ 依赖 decode 是否真的走 MMA**
- 我们 verify（T=8）走 **TILE**，所以 P1-3 的常量只影响 **prefill**。
- 装上 P0-1 后 MMA 才在 decode 上变得相关 => **P1-3 的价值也由 P0-1 决定**。

### 11.5 结论：优先级要按「前提」重排
**P0-1（q8_0 KV 张量核注意力）不再只是「一条优化」，它是多条结论的前提。**
在它落地之前，关于 **KV dtype、长上下文带宽、n_max 最优值、D256 常量** 的判断都是**条件判断**，不应当作已定论。
同时，**R1 与可加性模型的矛盾必须先解决** —— 它决定我们整张「已证伪」清单是否可信。

## 4. 待深挖（子代理正在做）
- `v100-skinny fork_patches/gdn_attn.py`：**GDN 投机状态的快速元数据构建**，作者自测 **-1.4 ms/step**，
  消掉 **21 次 device sync + 约 70 次 copy** —— 我们的 direct 抖动源正是 **GDN 递归状态视图**，这条高度相关。
- `ninfer` 的 **ReplaySSM**：不做 T 份 state trajectory，只存 raw record 再 fold。
- 两者的图捕获 / 内存池细节。
