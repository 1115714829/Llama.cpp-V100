# SESSION 2026-09-20 (DSH) - 本轮实测账本与优化路线

> 本文件记录 DSH 会话 2026-09-20 深夜这一轮**亲自实测**的数字（全部在 AC922 上跑出来，命令见 §4）。
> 与旧文档冲突时以本文件为准（红线除外）。相关：AUDIT-2026-09-20-dsh.md（可信度索引）、INSTRUMENTATION.md（量具）、1CAT-PORT-BACKLOG.md（移植清单）。

## 0. 一句话结论

**每轮 58.9 ms 的构成已被逐项量出来**：target 步 39.2 ms（其中 allreduce **13.1 ms**）、draft 13.1 ms、CPU selector 4.2 ms。
target 步在 llama-bench 里**孤立跑 M=8 前向也要 30.8 ms**（瓶颈不是流水线开销，而是前向本身：9.5 GiB/卡 ÷ 30.8 ms = 309-391 GB/s ≈ V100 峰值的 33-43%）。
本轮净收益（已落地、已 A/B）：**D=256 FA 表 Q_in_reg=false -> 32K prefill +11.1%、8K +2.9%、解码不变、greedy sha256 逐位一致**。

---

## 1. 每轮账本（实测，非推算）

量具：LLAMA_ROUND_TIMING=1 + 本轮新增 GGML_CUDA_AR_TIMING=1（见 §5）。

| 成分 | 实测 | 口径 |
|---|---:|---|
| target 步（decode+sync） | **39.16-39.90 ms/轮** | [RT] target decode+sync，256 轮 |
| 其中 allreduce | **13.1 ms/轮** | 35,474 次 / 257 轮 = **138.0 次/轮**，**94.9 µs/次**（无偏） |
| draft（注入 + 块前向） | **13.12 ms/轮** | spec timing draft_decode |
| 其中注入 decode（主机侧） | 1.67 ms/次 | 新增 inject timing |
| CPU selector | **4.20-4.22 ms/轮** | spec timing |
| 合计 | 56.5-58.2 ms/轮 | 与官方标尺 58.9 ms 自洽 |

官方标尺（drop_caches，NPRED=512，TP3+NCCL+P2P）仍是 **tg 95.20 / 58.9 ms/轮 / AL 5.55**；
本轮诊断跑（NODROP，NPRED=384）tg 89.4-89.6（AL 5.18）= 58 ms/轮 ✓ 自洽。

### 1.1 allreduce 实测（本轮首次拿到真实数字）

- **138.0 次/轮**；每次 3 个张量（每卡一个）；平均 572 KB（计数器里的 bytes 是 3x 张量，即张量约 181 KB；M=8 时 5120x8x4 B = 164 KB）。
- 单次延迟 **94.9 µs 无偏**。带 CUDA graph 时环形采样丢约 50% 样本（GPU 落后于主机），均值偏低（59.6-84.0 µs）；**关图 arm skip=12（几乎全采到）-> 94.9 µs 才是无偏值**。
- 传输实测：NCCL 2.29.7 走 **P2P/direct pointer**（不是 SHM），8 个 coll 通道、ring 0->1->2、isAllDirectP2p 1；NET 插件 Socket（本机无 IB），NVLS 不可用。
- 链路能力：GPU0/1/2 组内 **NV2 = 2 x 25 GB/s**（nvidia-smi topo -m / nvlink -s）；
  自写微基准（/root/ar_bench.cu）实测**裸 cudaMemcpyPeerAsync 164 KB = 5.07-5.84 µs（边际约 47 GB/s，正好是 NV2 上限）**。
- **NCCL 调参全部无效或变差**（同机 A/B，ar_us_avg）：默认 74-84；NCCL_PROTO=LL128 -> 101；NCCL_MAX_NCHANNELS=1 -> 110；NCCL_ALGO=Tree -> 109。**不要再试这三个旋钮。**
- **CUDA graph 是刚需**：GGML_CUDA_DISABLE_GRAPHS=1 -> target 步 39.9 -> **62.2 ms**、enqueue 20.4 -> **57.8 ms/轮**、tg 89 -> 63。
- 自写 push 式一次性 allreduce（/root/ar_bench2.cu：push -> fence -> flag -> 自旋 -> reduce，融合单 kernel + CUDA graph）实测 **59-68 µs 且与尺寸无关**（32 KB 59.3 / 160 KB 59.6 / 640 KB 68.4）=> 固定开销约 57 µs，**不是主机启动开销**（加图不变），首要嫌疑是 __threadfence_system()；栅栏作用域对比实验因计数器复位竞态死锁，未完成（见 §6）。

### 1.2 target 前向标定（llama-bench，TP3，f16 KV，fa on，-ts 1/1/1）

| 批量 | 吞吐 | 单次前向 | 折算 |
|---|---:|---:|---|
| pp1 | 41.22 t/s | **24.3 ms** | 391 GB/s/卡（峰值 43%） |
| pp8 | 246.95-259.85 t/s | **30.8 ms** | 309 GB/s/卡 |
| pp64 | 592.54 t/s | 108 ms | 边际 1.4 ms/token |
| pp256 | 1566.48 t/s | 163 ms | 边际 0.64 ms/token |

=> M=8 的**固定成本（读权重）约 24 ms**，每多一个 token 只加约 0.93 ms。
=> 真实 target 步 39.2 ms = 孤立前向 30.8 ms（含 AR 13.1）+ 8K 上下文注意力/验证结构约 8 ms。
=> **目标 180 t/s（30.8 ms/轮）要求 target 步约 22 ms**，即"读权重"要从 24 ms 压到 13-15 ms。

### 1.3 draft 侧（本轮把结构定死了）

- 每轮 **2 次 draft 图计算**，都在同一批 CUDA 流上串行：
  1. **注入**：common/speculative.cpp:1383（process() 里的 llama_decode(ctx_dft, batch_inject)，每轮约 AL 个 token，把 target 层特征写进 draft KV）；主机侧仅 **1.67 ms**（异步）。
  2. **块前向**：common/speculative.cpp:1440（draft() 里，1 anchor + 7 mask = 8 token 一次并行前向）；**draft_decode 量到的 13.12 ms 覆盖两次前向 + logits D2H 同步**（CPU selector 需要 host 上的 logits）。
- --spec-draft-n-max 扫描（零编译）：

| n_max | 块 token | AL | tg | target 步 | 集合通信/轮 | AR 单次 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 1.81 | 41.2 | 22.4 ms | 74 | 39.6 µs |
| 7 | 8 | 4.90-6.13 | 76.9-95.1 | 39.8 ms | 148 | 72-95 µs |
| 15 | - | - | - | 该 arm 无输出，未复现 | - | - |

- **draft_decode 几乎不随块大小变化**（12.73 -> 13.96 ms）=> **权重读取/固定开销受限，不是计算受限**；理想约 2-3 ms => **头寸约 10 ms/轮**。
- 1cat 对照（RESEARCH-1cat-dflash2-path.md）：其 draft 是**整图 CUDA graph replay**，且"注入"是 _precompute_context_kv() **一次投影**（qwen3_dflash.py:707,725-726），而我们是**一次完整 draft 前向** => 结构差异，下一阶段主攻方向。
- draft 模型真实结构（src/models/dflash.cpp）：MLA（wq_a/wq_b、wo_a/wo_b、单 KV 头 attn_kv）+ **MoE**（ffn_*_exps、ffn_*_shexp）+ grouped conv（DFlashGroupedConv）+ 编码投影 fc: {target_layer_ids x n_embd, n_embd}。
  llama-bench **无法单独加载 draft GGUF**（failed to create context），无法用 bench 标定它的理想前向。

---

## 2. 本轮落地的改动（已 A/B 验证）

### 2.1 D=256 FA 表：Q_in_reg=false（M5 子代理发现，主线验证并落地）

文件：llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh，函数 ggml_cuda_fattn_mma_get_config_volta()

~~~cpp
    // D=256 GQA: Q resident in registers spills on Volta, stage it through shared memory.
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 128, 2,  64, 128, 128, 128, 2, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16,  64, 4,  32, 128, 128, 128, 2, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  32, 128, 128, 128, 2, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  32, 128, 128, 128, 2, false);
~~~

同源 A/B（/root/libdir-prefa = b0e1e72c vs /root/libdir-instr = 1d0d4204，唯一变量 = FA 表）：

| 口径 | pre | post | 变化 |
|---|---:|---:|---:|
| llama-bench pp8192（TP3+ub2048+f16KV） | 2457.50 | 2528.36 t/s | **+2.9%** |
| llama-bench pp32768 | 1952.69 | **2169.87** t/s | **+11.1%** |
| llama-bench pp131072（-r 1） | 999.11 | **1357.22** t/s | **+35.8%** |
| llama-bench pp131072（-r 2 复核） | 1054.20 ± 0.07 | **1355.65 ± 0.80** t/s | **+28.6%** |
| 标尺 tg（3 prompt，NPRED=384，NODROP） | 89.64/69.99/113.79 | 89.42/70.06/114.03 | 噪声内不变 |
| target 步 / draft | 39.242 / 13.12 ms | 39.163 / 13.12 ms | 不变 |
| **greedy sha256** | f3edac19...02ca34 | f3edac19...02ca34 | **逐位一致** |

微基准（M5，test-backend-ops perf -o FLASH_ATTN_EXT，真实几何 hsk=hsv=256/nh=4/nr23=[6,1]/causal/F16 KV）：
基线 19.38-19.77 TFLOPS -> **29.03-30.39 TFLOPS（+53%）**，kv=4096 与 65536 一致。

M5 终版结论（`M5-FA-VOLTA-TABLE.md` + `m5-fattn-volta.patch`）：

- **根因是寄存器溢出**：ptxas -v 显示 Q_in_reg=true 用满 **255 寄存器 + 2060 B spill stores / 1888 B spill loads**；false 后 254 regs、**0 spill**。
  这解释了为什么 nbatch_fa / combine / K2 / V2 / nthreads / occupancy 这些"分块"旋钮几乎都无效（瓶颈是寄存器，不是分块）。
- **解码（nb<=8）走 TILE kernel，不经 MMA 配置表**：判据 `Q->ne[1] * gqa_ratio_eff <= 16`（本模型 gqa_ratio_eff=2）
  => **8 token 验证口径既不受益也不受损**（与上面 A/B 完全一致）。收益全在 **ubatch >= 9 token（prefill）**：
  kv=8192 nb=16 -> **+113%**、nb=64 -> **+96%**、kv=65536 nb=512 -> **+53%**。
- **ncols 必须 8/16/32/64 四行都补**：缺哪个 ncols，那个 ncols 就**静默回落 Ampere 行**（Q_in_reg=true），同模型不同 batch 会出现性能断层。
  本树四行已统一为 `(128,2,32,128,128,128,2,false)`（M5 实测可编译/可链接/可运行；ncols=8 分支在 V100 永不进入，但模板仍会被实例化）。
- 残余风险：同一行也被 ncols2=1/4/8 的分裂共享（别的 gqa_ratio 才触发），这些 (ncols1,ncols2) 组合**没有运行时验证**（能编译 != 能跑）。

**量 FA 的铁律**（M5 踩的坑）：FA 的 device kernel 实例在 template-instances/fattn-mma-f16-instance-ncols1_*-ncols2_*.cu.o，
**只重编 fattn.cu.o 得到的测量完全无效**（配置怎么改都一样）。改 .cuh 后必须让这些实例 TU 一起重编（cmake --build 会做）。

**已排除的 FA 配置**：nthreads=256 对该 D=256 组合 -> CUDA error invalid argument（起不来）；nstages_target 在 Volta 无效（无 cp.async，恒为 0）；nbatch_combine=64（-25%）、nbatch_V2=64（-13%）都更慢。

---

## 3. 优化路线（按 头寸 x 可实施性 排序，全部带量化头寸）

| # | 目标 | 现状 | 理想 | 头寸/轮 | 备注 |
|---|---|---|---:|---:|---|
| 1 | **draft 侧**（注入 + 块） | 13.1 ms（2 次前向，权重/固定开销受限） | 约 2-3 ms | **约 10 ms** | 1cat 注入是"一次投影"；块前向是整图 replay。先查 draft 图是否被 CUDA graph 捕获（关图对 draft 无影响 = 可疑） |
| 2 | **allreduce** | 94.9 µs x 138 = 13.1 ms | 20-30 µs/次 | **约 9-10 ms** | 自写 push 式已做到 59 µs（含约 57 µs 固定开销，疑为 __threadfence_system）；NCCL 旋钮无效 |
| 3 | **target 前向本体** | 24 ms 固定（391 GB/s/卡）+ 0.93 ms/token | 约 13-15 ms | **约 9 ms** | M=8 的 MMVQ/GDN/MoE；旧结论"量化 matmul 头寸 1.36x"需按 M=8 重估 |
| 4 | CPU selector | 4.2 ms | 约 0 | 约 4 ms | 已 8 线程；每轮 D2H 4.9 MB logits |
| 5 | 长上下文注意力/FFN | 32K prefill 2169 t/s | 1cat 3567-4069 | 大 | FA 已 +11%；256K 需实测 |

**达到 180 t/s 需要 1+2+3 同时落地**（58.9 -> 30 ms）；单靠 1 或 2 大约到 115-125 t/s。

---

## 4. 复现命令

~~~sh
# AR 计时（本轮新增量具）
GGML_CUDA_AR_TIMING=1 CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=art NPRED=192 NODROP=1 \
  bash /root/p60-ab-harness.sh
grep -a AR. /tmp/p60-art-server.log | tail -2      # calls/timed/us_avg

# 无偏 AR 单次延迟（关图 => 采样不再丢）
env NOGRAPH=1 GGML_CUDA_AR_TIMING=1 CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=dd2 NPRED=384 NODROP=1 \
  bash /root/p60-ab-harness.sh

# target 前向标定
LD_LIBRARY_PATH=/root/libdir-instr CUDA_VISIBLE_DEVICES=0,1,2 /root/libdir-instr/llama-bench \
  -m /root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf -ngl 999 -sm tensor -ts 1/1/1 \
  -p 1,8,64,256 -n 0 -r 5 -ctk f16 -ctv f16 -fa 1 -o md

# 链路/P2P 微基准
/usr/local/cuda-12.4/bin/nvcc -O3 -arch=sm_70 -o /root/ar_bench /root/ar_bench.cu && timeout 240 /root/ar_bench
~~~

服务器上本轮新增的脚本：/root/nccl-arms.sh、/root/nmax-arms.sh、/root/draftdiag.sh、/root/fa-ab.sh、/root/fa-128k.sh、/root/ar_bench.cu、/root/ar_bench2.cu；
库快照：/root/libdir-prefa（FA 改动前，b0e1e72c）、/root/libdir-instr（FA 改动后，1d0d4204）。

---

## 5. 本轮新增/改动的量具（env 开关，均为诊断用，可随时移除）

| env | 位置 | 作用 |
|---|---|---|
| GGML_CUDA_AR_TIMING | ggml-cuda.cu 的 ggml_cuda_ar_count（环形事件对，延迟读取不阻塞流水线） | 每次 allreduce 的 calls/tensors/bytes/timed/skip/us_sum/us_avg |
| LLAMA_SPEC_TIMING（扩展） | common/speculative.cpp 的 process() / draft() | 新增 inject timing（注入 decode 主机耗时）与 draft ctx: n_eval/n_reused |
| GGML_CUDA_OP_TIMING | 同上 | 现在只保留 calls 计数（逐算子事件表在多卡不可用，勿开） |
| NOGRAPH=1（harness 参数） | 映射到 GGML_CUDA_DISABLE_GRAPHS=1 | 暴露主机提交成本（会显著变慢，仅诊断） |

注意：本地工作区相对 HEAD 的未提交改动 = ggml-cuda.cu（AR 计时）、common/speculative.cpp（draft 计时）、src/llama-context.{h,cpp}（[RT] 探针）、ggml/src/ggml-cuda/fattn-mma-f16.cuh（**FA 表，这是要保留的功能改动**）。

---

## 6. 待办（下轮直接接）

1. **draft 图是否被 CUDA graph 捕获**：关图对 draft_decode 无影响（13.47 -> 13.58）而 target 翻倍 => 要么 draft 没被捕获、要么它真的 GPU 受限。查法：给 draft ctx 打 [RT] 式 reuse/rebuild 计数，或搜 ggml_backend_cuda_graph_compute 的禁用日志（llama.cpp 只在禁用时打日志，本轮未见）。
2. **栅栏作用域实验**：ar_bench2.cu 的 memset-in-graph 与对端 push 有竞态（对端先累加后被本方 memset 清零 => 自旋死锁）。改成"累积 epoch、不做 memset"的版本重测 __threadfence vs __threadfence_system，拿到 push-AR 真正下限，再决定是否移植进 ggml-cuda/allreduce.cu（移植要求：CUDA graph 可捕获、无主机同步、固定归约顺序；正确性门走 AL 不劣化 + ppl <= 0.1%）。
3. **FA 表第二/三批**：M5 正在扫 occupancy 3/4、nbatch_fa 64/128、combine 64；落地前同样 A/B + greedy sha256。
4. **draft 结构**：评估"注入与前向合并为一次图"或"按 1cat 方式改成一次投影"，两者都是**大改动 => 先问用户**。
5. **长上下文献**：128K 已完成（§2.1：+28.6% ~ +35.8%）；**256K 仍未测** —— f16 KV 在 TP3 下 256K 需约 5.7 GB/卡 KV + 9.5 GB/卡权重，可能 OOM，建议先用 q8_0 KV 跑 A/B。
---

## 7. 2026-09-21 凌晨：图碎片化发现（本轮最重要的结构性结论）

新增量具：`GGML_CUDA_GRAPH_DEBUG`（ggml-cuda.cu，绕过 Release 下被 NDEBUG 编掉的日志），打印
① 哪个 `mul_mat_id` 形状触发 needs-sync；② 哪个 ctx 的图被判 incompatible；③ 每 256 次 `graph_compute` 的 capture/replay/direct 计数与节点数；④ 属性变更（warmup 重置）样本。

### 7.1 实测（NODROP，NPRED=192，TP3+NCCL+P2P）

| 配置 | graph_compute/轮 | 重捕获/轮 | direct/轮 | enqueue_us/轮 | target 步 |
|---|---:|---:|---:|---:|---:|
| 只有 target（`--spec-type none`，M=1，713 轮） | 387 | 4.3 | 10.9 | 11.8 ms | 20.9 ms |
| target + draft（n_max=7，153 轮） | 422 | **17.9** | **92.7** | 24.1 ms | 39.0 ms |
| **draft 增量** | +35 | **+13.6** | **+82** | **+12.3 ms** | +18 ms |

- 每个子图只有 **7-79 个节点**（`last_nodes` 采样值），capture 出来的图基本没有摊销价值。
- 图 compatibility 检查**从未失败**（`mul_mat_id needs sync` 与 `incompatible graph` 零命中）⇒ 与 MoE 无关；
  真正的开销是**属性每次变化导致 warmup 重置**（`props changed`，且 `warmup=0` 反复出现）。
- 碎片数量 ≈ 3 × 128 = 384：**tensor 模式下 meta 后端在每个跨设备 allreduce 边界切一刀**，每个碎片再按 3 个设备各提交一次。

### 7.2 结论（含对旧假设的证伪）

1. **P2 的原始假设"draft 在 tensor 下白付 ~9 ms allreduce"被证伪**：draft 只让集合通信从 **128.1 → 138.0 次/轮**（+9.9 次 ≈ 0.9 ms）。
   （更正：此前把 n_max=1 臂算成"74 次/轮"是错的——我把 timed 当成了总数；实际 89,560/649 = **138.0 次/轮**，与 n_max=7 相同。）
2. **draft 的 13 ms ≈ 主机侧图管理**：+13.6 次重捕获 + 82 次逐节点直提（每轮 12.3 ms 的 enqueue 增量），
   与 `draft_decode` 13.0-14.0 ms、与"关图不变"（graph 本来就没生效）完全自洽。
3. target 步的 **11.8 ms enqueue**（387 次碎片提交）就是账本里长期"未归因余量"的真身。

### 7.3 可选修法（均属结构性改动，需用户批准后再动）

| 方案 | 预期 | 代价/风险 |
|---|---|---|
| **A. 静态形状/填充**：把 verify 批次（1+n_max）与注入批次（AL）都固定成常量形状，消除"属性变化 ⇒ 重捕获" | 可回收 draft 的 ~10 ms + target 的一部分 | 需要改 spec 编排与 mask 语义；正确性门必须过 |
| **B. 跨设备整轮图捕获**（1cat 的 fullgraph 路线）：让 meta 后端把一整轮的多设备工作捕获成**一张**图 | 理论上把 500 次提交压成 1-2 次 | 大改动（meta 后端 + 事件编排）；1cat 的 `VLLM_SM70_USE_BREAKABLE_CUDAGRAPH` 实测 −13%~−29%，**这条路要先小样验证** |
| **C. 降低每次 graph_compute 的固定开销**（锁、图映射查找、属性 memcmp、cudaSetDevice） | 387 次 × 20 µs ≈ 5-8 ms 里的一部分 | 低风险小改动，但要先量出固定开销到底多少 |
| **D. 减少碎片数**：把 138 个 allreduce 边界合并（例如相邻 allreduce 合并成一次 NCCL group） | 碎片数线性下降 | 需要改 meta 后端的分段逻辑 |

**下一步建议**：先做 **C 的量化**（在 graph_compute 入口/出口加计时，得出每次调用的固定开销），再决定 A/B/D 的投资顺序。
### 7.4 固定开销已量化：方案 C 出局

在 ggml_backend_cuda_graph_compute 入口到判定结束之间加计时（同一 GGML_CUDA_GRAPH_DEBUG 开关）：

~~~
[GRAPH] calls=64512 capture=2742 replay=47583 direct=14187 decision_us=212094 per_call=3.3 avg_nodes=40.0
~~~

- **每次调用的判定开销只有 3.3 µs**（含 compatibility 扫描 + 属性 memcmp；平均 40 节点/片）
  => 422 次 x 3.3 µs = **1.4 ms/轮**，**方案 C（优化判定路径）没有价值，出局**。
- 真正的开销是 **约 372 次/轮的图重放** + **约 109 次/轮的"直提"**（属性一变，llama.cpp 就整段退回逐节点提交，
  见 ggml-cuda.cu:4694-4699：warmup_complete = false => 该次不走图）。draft 增量（+82 直提 +13.6 重捕获）正好等于它的 12.3 ms。
- cudaGraphExecUpdate 帮不上忙：它需要一张**新捕获**的图作为 update 源，而重新捕获 40 节点的图与直接提交 40 个节点代价相当
  （这也是 llama.cpp 选择直提的原因）=> **必须在源头消灭"属性变化"或"碎片数量"**。

### 7.5 结论与下一轮顺序（修订版）

| 优先 | 动作 | 预期 | 说明 |
|---|---|---|---|
| 1 | **消灭 draft 的碎片直提**：让 draft 每轮的图形状固定（注入批次 AL 与块批次都不变） | **约 11 ms/轮** | 需改 spec 编排（填充/静态形状）=> **大改动，需用户批准** |
| 2 | **跨设备整轮图捕获**（meta 后端把一轮的多设备工作 + NCCL 集合通信捕获成一张图，1cat 的 fullgraph 路线） | 把 400+ 次提交压成 3 次 | 大改动；1cat 的 breakable 变体实测 -13~-29%，**必须先小样验证** |
| 3 | 减少碎片数（相邻 allreduce 合并成一次 NCCL group 等） | 线性下降 | 改 meta 后端分段逻辑 |
| ~~4~~ | ~~优化 graph_compute 判定路径~~ | ~~1.4 ms~~ | **已量化，出局** |
### 7.6 根因链闭环，以及"填充注入批次"为何不可行（本轮否决）

1. **碎片**：tensor 模式每轮被切成 387（仅 target）/ 422（含 draft）个 graph_compute，平均 40 节点。
2. **开销**：其中约 372 次是图重放，约 109 次是"直提"（属性一变，llama.cpp 就把该片退回逐节点提交，见 ggml-cuda.cu:4694-4699）。
3. **谁的属性在变**：属性差异探针（GGML_CUDA_GRAPH_DEBUG）打印出的首批样本显示，变化集中在
   norm-0 / attn_norm-0 / **cache_r_l0** / **conv_states-0** / node_9 这类张量上，形状如 [9600,8]、[30720,0]。
   结合"注入批次 = AL 每轮都不同"，可判定：**注入 decode 的形状每轮变化 ⇒ sched 缓冲区分配随之变化 ⇒ 与其共享缓冲池的块 decode 图也一起失效** ⇒ 整轮碎片全部退回直提。
4. **"把注入批次填充到固定长度"不可行（本轮否决）**：draft 模型里有 **递归状态**（conv_states / cache_r_l0，DFlash2 的 grouped-conv + GDN 结构）。
   多喂重复 token 会**多推进一次递归状态**，而递归状态不像 KV 那样会被下一轮覆盖 ⇒ 会静默污染 draft 的后续输出（表现为 AL 下降，且不可恢复）。
   所以固定形状**不能靠填充**实现。
5. **可行方向（=1cat 的做法）**：把"注入"从**一次完整 draft 前向**改成**一次投影**（1cat 的 _precompute_context_kv，
   qwen3_dflash.py:707,725-726）：只把 target 隐状态投影成 draft 的 K/V 写入缓存，**不推进递归状态**。
   这样：(a) 少掉一次 draft 前向的权重读取；(b) 投影的批次可以安全地填充成常量形状（无状态语义）⇒ 图稳定 ⇒ 重放代替直提。
   **预期回收：draft 侧约 10-12 ms/轮**（当前 13 ms 里大部分）。
   这是**结构性改动**（改 DFlash 图的构建方式），按红线需用户批准后再动。



