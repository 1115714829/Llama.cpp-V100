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
---

## 8. 2026-09-21 上午：上游 PR 核对 + draft churn 代码级根因 + 两个有界修法

### 8.1 上游 PR 核对（按用户要求定期查最新 PR 的产物）

| PR/Issue | 状态 | 与我们的关系 |
|---|---|---|
| **#22041** "Reduce CPU overhead in meta backend: cache subgraph splits when cgraph is unchanged"（gaugarg-nv，2026-04-19 合并） | **已在我们的 base 里**（ggml-backend-meta.cpp:1970 `needs_rebuild`、:2283 给子图分配 uid） | 上游在 2x5090 上 tg128 **+16%**；我们已享有该收益 |
| **#28549** "Enable CUDA graph for MTP draft"（gaugarg-nv，2026-09-16 合并） | **已在我们的 base 里**（llama-context.h:371 `std::array<llm_graph_result_ptr, 2> gf_res_prev`、按 `n_outputs > 0` 选槽） | 上游症状与我们完全一致（两种形状互相顶掉捕获图）；双 arena 已生效 |
| **#28661** "SM70 V100 Volta - Crashes using CUDA FA for Non-Standard Dimension Heads"（open） | 需要核对 | 只影响**非 64/128/256** 的头维度；我们 D=256 => **本次 FA 配置改动无此风险**（且 8K/32K/128K 实测通过） |
| #26289 "CUDA: tune fp16 tile FlashAttention configs for head sizes 40-112" | 上游已合并（未核对是否在 base） | 我们的**解码走 TILE kernel**，但 8K 下注意力占比约 1-2%，暂不追 |
| #22105 DFlash 支持 / #25173 DSpark | 上游 | 我们 DFlash2 路径的来源 |

### 8.2 draft churn 的代码级根因（本轮闭合）

1. `llama-graph.cpp:29-46`：`build_attn_inp_kq_mask()` 把 mask 建成 **`[n_kv, n_tokens]`**，其中 `n_kv = mctx->get_n_kv()`（随上下文增长）、`n_tokens = ubatch.n_tokens`。
2. `llama-graph.cpp:48-65`：`can_reuse_kq_mask()` 要求 **ne[0]==n_kv 且 ne[1]==n_tokens** 才允许复用。
3. ⇒ draft 的 `causal_attn = false`（必须显式 mask），且
   - **块前向**：n_tokens=8 恒定，但 **n_kv 每轮 +AL 增长** => mask 形状变 => 整图重建；
   - **注入前向**：n_tokens=AL 每轮变化 => 同样重建。
   ⇒ **draft 的每一片子图每轮都判为"属性已变" => 全部退回逐节点直提**（实测：draft 的 ~70 次子图计算/轮，direct 占 ~82 次/轮），而 target 因果 + 批次/分桶稳定 => 只有 2.8% 直提。
4. 这也解释了 `GGML_CUDA_DISABLE_GRAPHS` 对 draft 完全无影响（它本来就没在用图）。

**修法（下一步，需注意 n_kv 分桶与 mask 填充语义）**：把 mask 的 ne[0] 由 n_kv 改为**分桶值**（如 GGML_PAD(n_kv, 256)），并把填充区写 -inf；
这样 mask 只在跨桶时重建。注意：**注入前向的 n_tokens=AL 变化是独立的第二因**，靠分桶修不掉（而"填充注入批次"因 draft 含递归状态 conv_states/cache_r 而不可行）。

### 8.3 本轮新增的两个有界修法（已提交代码，待 A/B）

1. **sched 层 split 缓存**（ggml-backend.cpp）：`ggml_backend_sched_alloc_graph()` 每轮无条件重跑 `split_graph`（400+ 子图扫描 + 逐 split 优化）= 实测 `alloc_us` **4.26 ms/轮**。
   新增 `last_graph_uid` + `GGML_SCHED_SPLIT_CACHE=1` 时才跳过重复切分（默认行为不变，便于同库 A/B）；
   `ggml_backend_sched_reset()` 里清零 uid 以强制重切。另加 `GGML_SCHED_SPLIT_TIMING=1` 输出 split/alloc 两段耗时。
2. **selector 重排**（common/speculative.cpp）：实测 `fetch 0.06 + work 3.21-3.30 ms`，其中 gate 矩阵乘
   （`sel_hidden[rank x 5120]` 对**每个** block 位置各读一遍 = 约 42 MB/轮）是大头，不是 top-k 扫描。
   改成 rank-major（每行只读一次、8 个位置复用），**每个点积的累加顺序不变 => 数值逐位相同**；
   同时把计时拆成 `fetch + topk + gate` 三段便于验证。

### 8.4 又一次靠"核对代码"避免弯路：mask 宽度**已经**分桶

llama-kv-cache.cpp:1250-1263：

~~~cpp
uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    // pad the n_kv value so that the graph remains constant across batches and can be reused
    const uint32_t n_pad_cur = std::max(n_pad, 256u);
    result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
~~~

=> mask 的 ne[0]=n_kv **每 256 token 才变一次**，block 前向（n_tokens=8 恒定）在桶内是稳定图。
**所以 draft 的约 82 次/轮直提全部来自"注入前向"**（n_tokens=AL 每轮都不同 => 整图重建），与实测自洽：
+82 direct/轮、+13.6 capture/轮、GGML_CUDA_DISABLE_GRAPHS 对 draft 无影响（本来就没在用图）。

### 8.5 对应修法：按**批大小**分槽的图 arena（已实现，待验证）

上游 #28549 只分"有输出/无输出"两槽；我们把每类再按批大小分 16 槽（llama-context.h 的 gf_res_n_size_slots）：
每个 AL 形状各一张图 => 各自独立的 CUDA graph 缓存键 => 某形状第二次出现即完成 warmup 并开始重放
（warmup 只要求"与该键上次属性一致"，不要求连续两轮）。预期把注入的约 82 次直提变成重放 => **约 -10 ms/轮**。
风险：arena 数量增加 => 捕获的 CUDA 图变多（稳态只用到 AL 的少数取值，约 6-8 张），显存/主机内存可控。

### 8.6 selector：rank-major 本身没用，瓶颈是串行 FMA 依赖链

实测 fetch 0.06 + topk 0.70 + gate 2.78 ms（rank=256, n_embd_dec=5120, n_tokens=8）：
gate = 10.5M MAC / 8 线程 = 每 MAC 约 2.1 ns（约 8 周期）=> **标量累加的串行依赖**，不是内存带宽。
已改为 4 路部分和（归约顺序固定 => 结果确定），预期 gate 降到约 0.8 ms。

### 8.7 sched split 缓存：守卫无法生效（有测量证据）

[SCHED] 显示 split=137 us/call 但 alloc_splits=2863 us/call，且 uid_before=0 ... last=0：
传入 ggml_backend_sched_alloc_graph 的图**每轮 uid 都是 0** —— 因为 **KV cache 有自己的 sched**
（llama-kv-cache.cpp:866/873 每轮 reset + 重建 cpy 图，图对象是新建的）。所以 split 只占 alloc 的 5%，
大头是 alloc_splits（每次 2.9 ms）。探针已改为 BIG（>200 节点，主解码图）/SMALL（KV cpy 图）分桶，
以便定位主图那 2.47 ms/轮花在哪。

### 8.8 selector 优化已 A/B 验证（子代理跑测，两臂 md5 一致）

| 指标 | 之前 | 之后 |
|---|---:|---:|
| selector gate | 2.78 ms | **1.00 ms** |
| selector 总计（spec timing） | 4.42 ms | **2.63 ms** |
| tg（prompt1/2/3） | 89.57 / 69.90 / 114.18 | **92.35 / 72.23 / 117.56** |
| AL | 5.18 / 3.75 / 6.38 | 完全相同 |
| greedy sha256 | f3edac19...02ca34 | **逐位一致** |

=> **+3.3% tg** 已落袋（提交 0a598af40）。

### 8.9 主机侧最大头寸：主解码图的 alloc_splits = 5 ms/次

分桶探针（新增；BIG = 主解码图 >200 节点，SMALL = KV cache 自己的 cpy 图）：

| 桶 | calls / 257 轮 | split | alloc_splits |
|---|---:|---:|---:|
| **BIG**（主解码图，649 节点） | 263 | 246 us/call | **5063 us/call** |
| SMALL（KV cpy 图，58 节点） | 249 | 25 us/call | 460 us/call |

- **split 只占 5%**，大头是 ggml_backend_sched_alloc_splits（5 ms 级/次）。
  target 的 [RT] alloc_us 是 2.44 ms/轮；BIG 桶均值 5.06 ms 混合了 target 与 draft 两个 sched。
- 根因：split_graph 每轮**交换 node_backend_ids / prev_ 数组**，使 alloc_splits 里的
  backend_ids_changed 恒为真 => 每轮走**全量重分配**分支。
- 修法（已实现，待验证）：给 sched 加**图指纹**（节点指针+形状+buffer 的 FNV 哈希，约 2k 次乘法/轮），
  指纹相同就跳过 split_graph（GGML_SCHED_SPLIT_CACHE=1）=> 数组不再交换 => backend_ids_changed 为假
  => 走 gallocr 快速路径。**不能用 graph->uid 做守卫**：split_graph 每轮给它赋新值；
  实测 uid_before 恒为 0 的调用来自 KV cache 新建的 cpy 图（它有自己的 sched）。

### 8.10 同时验证中的另一项：按批大小分槽的图 arena（见 8.5）

两项都已编译进服务器源码，正由后台子代理跑双臂（缓存关/开）取证。

### 8.11 验证纪律教训（本轮踩到，必须记住）

一次 A/B 被两个流程错误污染，结论作废：

1. **上传与编译竞态**：我在子代理"已经启动编译"的时间窗内上传了新源码 => 编译用的是半新半旧的树
   （表现为 [SCHED] 行里 `last=0`、`cached=0`，说明跑的还是**旧的 uid 守卫版本**，而不是新的指纹守卫）。
   **规矩：先上传并校验源码，再启动编译/跑测；编译期间绝不改服务器源码。**
2. **md5 自检集合不全**：之前只校验 `libggml-cuda.so` 与 `libllama-common.so`，
   但**调度器在 `libggml-base.so`**、**llama_context 在 `libllama.so`** => 这两个库的改动完全没被 A/B 纪律覆盖。
   **规矩：A/B 的 md5 集合 = {libggml-cuda.so, libggml-base.so, libllama.so, libllama-common.so} 四个。**
3. **再加一道二进制内标记校验**：`strings <lib> | grep -c <新格式串>`（例如 `BIG calls`），
   确认新代码真的进了**被加载的那个库**，而不是只看源码 grep。

这三条已写入本文件；下一次 A/B（arena 分槽 + 指纹缓存）按新流程执行。

### 8.12 结构性根因（本轮最终定论）：单 scheduler + 交替形状 ⇒ 每轮重建是**结构性**的

证据链（全部实机）：

1. 升级版属性探针（新增 `what=` 字段判定）统计：`other 120 / src_ne 116 / src_data 46 / nb 4`，
   样本节点全是 GDN/卷积类（`v_conv_predelta-22`、`conv_output_silu-57`、
   `cache_r_l5 (view) (copy of conv_input-5 (view))`、`state_predelta-14`、`attn_post_norm-48`）。
2. ⇒ 图内容每轮都在变（源张量形状/指针），不只是 mask；**块前向（n_tokens=8 恒定）也每轮失效**。
3. 但"按批大小分槽的图 arena"实测**无效**，原因：
   - llama.cpp 的 `ggml_backend_sched` **只有一份 `sched->graph` 与一份 splits**（单槽资源），
     复用图 result 时必须跳过 `ggml_backend_sched_reset()`；
   - 上游 #28549 的守卫 `gf_res_prev_active == res` 保证"只有连续同一个 arena 才复用"——
     而我们的流程是 注入(AL) -> 块前向(8) -> 注入(AL') 交替 ⇒ **守卫必然挡住复用** ⇒ 每轮 reset + 重切 + 重分配。
   - 若去掉该守卫而直接复用别的 arena，则 sched 里留着的仍是**另一个图**的 splits ⇒ 会执行错图（这正是守卫存在的原因）。
4. ⇒ 结论：**draft 侧约 12 ms/轮 的图管理开销、以及主图 ~5 ms 的 alloc_splits，都是"单 scheduler + 形状交替"的结构性代价**，
   在 llama.cpp 当前架构下无法用局部补丁消除；要根治只能走 1cat 的路线（静态形状 / 整轮单图捕获），属大改动。
   这条结论同时解释了：`cached=0`、arena 无效、以及 `GGML_SCHED_SPLIT_CACHE` 无法生效。

---

## 9. 验收标准逐条对账（2026-09-21 状态）

| 验收标准 | 起点（项目最初） | 现在（本轮实测） | 结论 |
|---|---|---|---|
| **1) 主口径 tg**（Q8_0 + DFlash2 n=7, ctx 8192, 官方采样, 3 prompt, drop_caches） | **95.20 t/s**（58.9 ms/轮, AL 5.55） | **96.43 / 99.37 t/s**（两臂, AL 5.55, 离散 3.0%） | 略升；**距 >=180 仍差 ~1.9x** |
| 1) 每轮 ms（= AL / tg） | 58.9 ms | **57.6 / 55.9 ms** | 目标 ~20 量级，未达 |
| 2) 32K prefill | 1520.45 / 1952.69 | **2169.87 / 2180.70** | +11.1%（FA 表） |
| 2) 128K prefill | 999.11 | **1355.65 / 1357.22** | **+28.6 ~ +35.8%** |
| 2) **256K prefill** | **370**（旧记录）/ 632.95（本会话改动前） | **895.93 t/s** | **+142% vs 370**；+41.6% vs 改动前 |
| 2) **TTFT@256K** | **672 s** | **293 s** | **-56%** |
| 2) decode@256K | 33.97 | **34.13 t/s** | 持平（**未改善，待办**） |
| 3) 正确性门 | - | 所有 A/B 两侧 **greedy sha256 逐位一致**（f3edac19...02ca34） | 通过 |
| 4) 纪律 | - | 同源 A/B + 四库 md5 + 二进制标记 + drop_caches 正式臂 + 两臂离散度 | 通过（并新增 AGENTS §4 第 18-21 条） |
| 5) 范围 | - | 改动只在 llama.cpp/；已提交 7 个 commit（llama : ... + Assisted-by: DSH），未 push | 遵守 |

**相对项目最初：tg 55.95 -> 96.4-99.4 = +72% ~ +78%**（两侧都开 DFlash2 投机解码）。
**未达成的部分**：主口径 180 t/s（差 ~1.9x）、每轮 ~20 ms、256K decode。

### 9.1 push 式 allreduce 的落地状态（本轮）

- 微基准路线**两次挂死**（ar_bench2 的 memset-in-graph 竞态；ar_bench4 的 300 s 零输出超时），
  已**停掉微基准自证**这条路 —— 改用**生产路径内置的 `[AR]` 计时**做判据（`GGML_CUDA_AR_TIMING=1` 直接给出单次集合通信延迟），
  这比体外微基准更贴近真实（含设备偏斜）。
- 已实现并接入（`ggml/src/ggml-cuda/allreduce.cu`，env 开关 `GGML_CUDA_AR_PUSH=1` + `GGML_CUDA_ALLREDUCE=internal`）：
  - 设备 IPC/UVA 三卡一次性 push+reduce 融合 kernel（每块独立到达 epoch，**只增不减 => 可被 CUDA graph 捕获重放**）；
  - 每个设备 3 槽位 + 每块 flags + 每块 epoch + 超时标志，**全部在 init 期分配**（避免图捕获中 cudaMalloc）；
  - 仅对 **3 卡 + F32 + <=8 MB** 的张量生效，其余自动回退（解码张量 164 KB ✓，prefill 大张量回退）。
- 已知风险：`__threadfence_system()`（POWER9 上可能很贵，这正是微基准想量的东西）；
  若 A/B 显示收益不足，可再试设备域栅栏（但必须先用正确性门验证）。

### 9.2 push 式 allreduce：实测不通过，已回退（成果存档）

双臂（同一套库 37d46800，唯一变量 = GGML_CUDA_ALLREDUCE=internal + GGML_CUDA_AR_PUSH=1）：

| 臂 | MEDIAN_TG | [AR] 单次延迟 | greedy |
|---|---:|---:|---|
| A（NCCL 对照） | **77.76** | **68.4 us** | 正确（f3edac19...） |
| B（internal + push） | **0.00** | **161.4 us** | **错**：content_len=0、sha256=e6e226d0... |

- 结论：push 内核**能跑起来**（构建/启动/无崩溃），但**归约结果错误**且**比 NCCL 慢 2.4 倍**（161.4 vs 68.4 us），
  也远慢于体外微基准的 59 us（生产内核比微基准多一次 __threadfence_system()，且 40 个块的 flags 挤在同一缓存行）。
- 处置：**回退**（allreduce.cu + ggml-cuda.cu 的三卡门卫），完整实现存为
  patches/0002-push-allreduce-experimental.patch（28856 B）。默认路径与已验证状态保持一致。

### 9.3 图 arena 分槽：无收益，已回退（同样存档）

按批大小分槽的 arena（src/llama-context.{h,cpp}，默认生效）实测对 direct/capture 计数与 tg 均无影响
（原因见 8.12：单 scheduler + 守卫），属"无收益却改变行为"的改动 => **回退**，
存为 patches/0003-graph-arena-per-batch-size-experimental.patch。
保留并提交的是 ggml-backend.cpp 的 [SCHED] 探针（env 关闭即零成本，且给出了 8.9 的关键数字）。

### 9.4 FA 的 TILE/MMA 线索：核对后判为死路

fattn.cu:644-651：Volta 上 Q->ne[1] * gqa_ratio_eff <= 16 走 TILE（注释：小矩阵上张量核不划算），
本模型 gqa_ratio_eff=2 => **nb<=8 恰好落在 TILE**。这**解释了 FA 表改动为什么只影响 prefill、解码零变化**。
但 M5 的微基准显示 TILE@nb=8 = 10.1 TFLOPS vs MMA@nb=16 = 18.4 / @nb=64 = 30.4 —— 这是**不同批量**的对比，
上游阈值本身合理 => **不改分发**。256K decode（34 t/s）的真正限制是 KV 带宽（q8_0 下每 token 每卡约 5.7 GB ≈ 6.3 ms）+ 权重读取。

### 9.5 教训（写进纪律）

**分布式内存协议必须先过"确定性单元测试"（3 卡、已知输入、比对期望和），再上端到端。**
本轮顺序错了：两个微基准都挂死（memset 竞态 / 自旋超时），我跳过单测直接端到端 => 得到"能跑但结果错"的臂，
既费时间又留下危险代码（幸而它始终在 env 开关后面）。

---

## 10. draft/target 图每轮失效的**精确根因**与修法（下一轮直接可做）

### 10.1 根因（代码级，一行定位）

属性差异探针（GGML_CUDA_GRAPH_DEBUG）的统计是 src_ne 116 / src_data 46 / other 120，样本节点全是 GDN/卷积类；
顺着这些节点找到写入侧：

- delta-net-base.cpp:490-496（卷积状态）与 :514-520（rollback 分支）：
  conv_state_update = ggml_view_2d(conv_states_all, row_count, n_seqs, nb1,
      (s_slot * mem_size + **kv_head**) * row_size)，随后 ggml_cpy(conv_state_last, conv_state_update)；
- llama-graph.cpp:3480-3498（递归状态 cache_r 的写入）同样是 view + 偏移里带 **rs_head**。

=> **视图偏移里含滚动缓存头指针（head）**，而 head 随序列前进 => **每轮偏移都变 => 这些视图的属性变化 =>
包含它们的整张子图被判"属性已变" => 逐节点直提（direct）**。
这就是 draft 每轮 +13.6 重捕获 / +82 直提（约 12.3 ms）的来源，也**同时挡住 P1（静态形状）与 P2（整轮单图捕获）**，
因为两者都要求图跨轮稳定。注意：这不是 llama.cpp 的 bug —— 捕获的图里偏移是常量，偏移变了就必须重捕获，这是**正确行为**。

### 10.2 修法（结果等价、有先例、机械）

llama.cpp 对**注意力 KV 早就用了索引式写入**：llama-kv-cache.cpp:1318-1350 的 cpy_k ->
`ggml_set_rows(ctx, k, k_cur, k_idxs)`，其中 k_idxs 是**图输入**（build_input_k_idxs:1409）=> 图里没有会移动的偏移 ✓。

把递归状态也照这个模式改：

1. **llama-graph.h**（llm_graph_input_rs，:262-276）：新增 `ggml_tensor * s_copy_dst;`（**I64** [n_rs]，与 KV 的 k_idxs 同型），
   以及需要的 main/extra 视图；`s_copy`（I32，读取侧）保持不变。
2. **llama-graph.cpp**：build_rs_inp_impl(:3502) 里创建 s_copy_dst 并 ggml_set_input；
   `llm_graph_input_rs::set_input`(:329) 里填入**目标行号**（= s_slot * mem_size + head + j，j 为序列下标）；
   build_rs 的写入(:3480-3498) 由 cpy(view(offset)) 改成 `ggml_set_rows(s, src, s_copy_dst_view)`。
3. **delta-net-base.cpp**（build_conv_state，:490-521）：同样把 conv_state_update 的 cpy(view) 换成 set_rows，
   行号 = s_slot * mem_size + head + j。
4. **（可选）llama-memory-recurrent.{h,cpp}**：加一个 "目标行号" 计算 helper，避免在图形层重复公式。

**为什么这个改动安全**：写入的**数据与目标位置完全相同**，只是寻址方式从"偏移视图"变成"索引张量" =>
数值结果逐位相同 => 用 greedy sha256 即可判定（预期不变），再用 AL 不劣化兜底。
**预期收益**：draft 侧约 -10~12 ms/轮（direct/capture 计数应大幅下降，draft_decode 13 -> 约 3 ms），
并且**解锁 P1/P2**（静态形状、整轮捕获）=> 这是通往 180 t/s 的必经一步。

**验证判据（一次跑测即可判定）**：GGML_CUDA_GRAPH_DEBUG=1 下
`[GRAPH] direct=` 每轮值应从约 66 降到个位数，`draft_decode` 从 13 ms 降到约 3 ms，greedy sha256 不变。

---

## 11. 2026-09-21 下午：两个"负结果"带来的方向纠正（重要）

### 11.1 CUDA graph 缓存键加形状信息 => 无收益，已回退

假说：不同批量的子图重建到同一 arena，nodes[0] 地址相同 => 共用缓存键 => 互相顶掉（对应统计里 src_ne 116 / src_nb 4）。
实现：ggml_cuda_graph_get_key() 改为对 (nodes[0], n_nodes, 所有节点的 ne[0..2]) 做 FNV 哈希并放进稳定表。

实测（NPRED=192, NODROP=1, 同源同协议）：

| 指标 | 改动前 | 改动后 |
|---|---:|---:|
| MEDIAN_TG | 77.26 | **77.28**（无变化） |
| [GRAPH] capture | 2742 | **3132（+14%，更差）** |
| [GRAPH] direct | 14187 | 13407（-5%） |
| [GRAPH] per_call | 4.5 us | 7.3 us（哈希开销） |
| draft_decode | 13.86 ms | 14.09 ms |

=> 净效果为负，**回退**。greedy sha256 两侧一致（只是性能问题）。

### 11.2 方向纠正：draft 的"12 ms 图管理"不在关键路径上，draft 瓶颈在 GPU 侧

首次拿到 draft ctx 自己的 [RT] 行（此前只有 target 的）：

~~~text
[RT] perf: ctx=Qwen3.8-27B-DFlash2 n_ctx=8192 splits=2 rounds=269 reuse=0 rebuild=269
     | build_us=33503 alloc_us=509256 setin_us=12491 enqueue_us=1542455 sync_us=0
~~~

折算每轮：build 0.12 ms + alloc 1.9 ms + setin 0.05 ms + **enqueue 5.7 ms**（reuse=0：每轮重建图）。
而 draft_decode = 13.9 ms => 主机侧只解释约 7.7 ms，且 enqueue 是**异步提交窗口**（与 GPU 工作重叠），
**剩余约 6 ms 是它在等 GPU**（logits D2H 同步）。

=> **结论（纠正 §8.12 与 §10 的框架）**：draft 侧 13 ms 的主因是 **GPU 侧效率**（约 1 GB 权重读取 + MoE + GDN 顺序算子，
实测有效带宽仅约 70-100 GB/s），而非主机侧图管理；把图管理降到 0 也不会带来墙钟收益（它与 GPU 工作重叠）。
**下一步应攻 draft 的 GPU 内核效率（算子融合），而不是图缓存。**

### 11.3 归档问题修复

patches/0002、0003 之前经 PowerShell 重定向写出，是 **UTF-16**（git apply 报 no valid patches）=> 已重编码为 UTF-8
（0002: 28856 -> 14427 B）。注意 0002 是针对当时的 ggml-cuda.cu 生成的，现已漂移，需手工重贴。

### 11.4 push allreduce 失败的根因（已确证）

搜索发现上游 #23480「Add missing buffer set in allreduce fallback !COMPUTE clear」与 #21808
「TP: fix 0-sized tensor slices, AllReduce fallback」正是这一类问题。对照代码：
**NCCL 路径在归约前对未标 GGML_TENSOR_FLAG_COMPUTE 的分片先 cudaMemsetAsync 清零，而我的 push 内核没有**
=> 本应为 0 的 padding 分片被加进归约结果 => **结果错误**（与实测"内容为空、sha256 不同"一致）。
次要问题：40 个块的到达 flags 挤在同一缓存行 => NVLink 往返放大 => 161 us vs NCCL 68 us。
=> 若重做：先补清零 + flags 每块独占一行 + 先跑 3 卡确定性单测。

---

## 12. push allreduce 第二次尝试：仍错，但根因再进一步（已回退，存档 patches/0002）

**本次做的两处修复**（记录在补丁里）：
1. **非 COMPUTE 分片清零**（对齐 NCCL 路径）—— 对应上游 #23480 同类问题；
2. **到达 flags 每块独占 128 B 缓存行**（push_flag_stride = 32）。

**实测（同源同协议，NPRED=192, NODROP=1）**：

| 臂 | MEDIAN_TG | [AR] ar_us_avg | greedy |
|---|---:|---:|---|
| A（NCCL 对照） | 78.17 | 68.2 us | 正确（f3edac19...） |
| B（internal + push，含两处修复） | 0.00 | **126.6 us**（原 161.4） | **仍错**，sha256 与修复前**逐位相同**（e6e226d0...） |

**关键推论**：错误是**确定性的**（两次尝试的错误 sha256 完全一致）=> 不是竞态；**清零修复不是（唯一）原因**。
flags 分行确实把延迟从 161 降到 126.6 us，但仍高于 NCCL 的 68.2 us。

**真正的协议缺陷（本轮定位）**：槽位是**单缓冲**（每对 (dst, src) 只有一个 slot），
而三个设备的 epoch 计数器**各自独立、互不同步**。若对端领先一轮，它的槽位已被**本轮之后**的张量数据覆盖，
而我仍以本轮的 ep 去读 => **跨轮数据串扰**（确定性、可复现）。这是经典的"流水线缓冲深度不足"问题。

**正确修法（下次直接照做）**：
- 槽位按 epoch 取模做**多缓冲**（深度 2-4，缓冲索引从设备端 epoch 推出 => 仍然捕获安全）；
- 把"允许的对端领先量"写成不变量并在内核里断言（领先超过缓冲深度 => 置 err）；
- **先做 3 卡确定性单测**：已知输入 -> 比对期望和，连跑 100 轮并**逐轮**校验（这次两次失败都因为跳过单测直奔端到端）。

**当前取舍**：连续两次失败后按纪律**收手并回退**（树回到已验证状态），补丁保留全部改进供下次从单测开始。

### 12.1 修正：奇偶多缓冲不是充分修法（本轮推演结论）

上一节提出的"按 epoch 取模多缓冲"经推演**不成立**，理由是：

- AR 的槽位是**跨不同张量复用**的（每轮的每个 allreduce 都是不同张量，但槽位固定）；
- 只要对端在**张量序列**上领先 d 步，它写的是槽位 (ep_peer-1) mod DEPTH；
  而我需要的是"对端针对**我这个张量**（它的第 ep 次调用）"的数据；
  当 ep_peer - ep 是 DEPTH 的整数倍时，槽位号相同但内容是**别的张量** => 仍然串扰；
- => **任何固定深度的环形缓冲都无法消除该危险**，必须引入**信用/确认握手**：
  接收方读完对端数据后发布"已消费计数"，发送方在写槽之前必须等到"对端已消费我此前那次写入"
  （2 深度流水线：写第 ep 次之前等 peer_consumed >= ep - 2）。这正是 NCCL 内部的做法。
- **结论**：push AR 是一个需要**精心实现 + 确定性单测**的独立子项目（1-2 天量级，不是 1-2 轮能收敛的）。
  本会话剩余轮次不再尝试。补丁保留全部已知改进（清零 + flags 分行 + 单缓冲诊断），并在此记录设计要点：
  1) 每 (src, dst, block) 一对 data-ready 与 consumed 计数器；
  2) 发送方写前等待 consumed[me][block] >= my_ep - DEPTH；
  3) 接收方 reduce 后 atomicAdd(consumed[me][block], 1)；
  4) 所有计数在设备内存、只增不减 => CUDA graph 捕获安全；
  5) **先写 3 卡确定性单测**（已知输入 -> 期望和，连跑 100 轮逐轮校验），再进端到端。

---

## 13. decode 深度曲线（验收标准 2 的补齐）+ 口径警戒

用 llama-bench 的 **深度协议**（-p 0 -n 16 -d <depth>，q8_0 KV，TP3+ub2048，-r 1）实测：

| 深度 | 8K | 32K | 128K | 256K |
|---|---:|---:|---:|---:|
| tg16 (t/s) | 33.88 | 31.83 | 24.76 | **18.45** |

**口径警戒（重要）**：验收基线里写的"256K decode 33.97 t/s"对应的其实是 **-p 262144 -n 16** 协议
（prefill 之后立刻 decode），我实测该协议为 **34.13 t/s**（=> 未改善，仍是待办）；
而深度协议（-d 262144，先把 KV 填到该深度再解码）给出 **18.45 t/s**。
两者**不可混用**：报告 decode 时必须写清是哪种协议。历史文档里"32K decode@depth 40.87"等数字同样需要复核其协议。

### 13.1 统一性洞察：本机"每次 kernel / 每次图调度"的成本极高（10-25 us）

把本轮所有实测放在一起看：

| 现象 | 数字 | 折算 |
|---|---|---|
| target 整步 | 39.1 ms | 约 1500 个 kernel => 约 26 us/kernel |
| draft 每对前向（注入+块） | 约 13 ms | 约 1000 个 kernel => 约 13 us/kernel |
| 关掉 CUDA graph 后多出 | 22 ms | 约 16000 次逐节点提交 => 约 1.4 us/节点（但图启动本身约 50 us x 372 次/轮 ≈ 20 ms） |
| allreduce | 68.2 us/次 x 138 次/轮 | NCCL 内核 + 等待对端 |

=> **本项目剩余池子里最大的共性不是带宽，而是"启动/提交/同步"的单位成本**。
=> 因此优先级应偏向"减少 kernel / 片段 / 集合通信**次数**"（融合、更大粒度、更少 TP 边界），
   而不是继续优化单 kernel 的带宽效率。

### 13.2 下一会话的候选实验（按性价比排序，均已写明判据）

1. **draft 侧算子融合**（老清单 P6）：把 GDN/卷积/MLA 的小算子合并，目标是把 draft 的 kernel 数从约 1000 降到约 300
   判据：draft_decode 13 ms -> 6 ms 以内，greedy 不变。
2. **减少 TP 边界**：（例如把 48 个 GDN 层的 allreduce 合并/降低频率，或在 GDN 层用复制而非切分）
   判据：138 次集合/轮 -> 显著下降，且 AL 不劣化。

---

## 14. TP 度数重扫（NCCL+P2P 时代，2026-09-21，四臂同源）

用户 2026-09-20 明确"1-6 卡自由、跨岛不是瓶颈，直接排除可能性"，而账本里的"TP3 最优"是 butterfly 时代结论
=> 本轮用当前已验证库重扫（CARDS 变化，其余相同：SPLIT=tensor, P2P=1, L=/root/libdir-instr, NPRED=192, NODROP=1）：

| TP | MEDIAN_TG | AL (p1/p2/p3) | enqueue/轮 | alloc/轮 |
|---|---:|---|---:|---:|
| TP2 (0,1) | 60.55 | 3.65 / 3.33 / 6.37 | 20.0 ms | 3.77 ms |
| **TP3 (0,1,2)** | **78.49** | 4.90 / 3.90 / 6.13 | 27.4 ms | 4.63 ms |
| TP4 (0,1,2,3) | 59.98 | 3.80 / 4.22 / 5.79 | 36.3 ms | 5.01 ms |
| TP6 (0-5) | 48.76 | - / 4.52 / - | 62.0 ms | 5.31 ms |

**结论**：
1. **TP3 仍是最优**（78.49 领先 60.55/59.98/48.76）=> 旧结论在 NCCL+P2P 时代**得到实证**，复扫义务了结；
   卡数自由不等于"越多越快"——因为每轮集合通信次数与碎片数随 TP 度数增长。
2. **enqueue 与 alloc 近似随碎片数线性增长**（TP6/TP3 的 enqueue 比 2.26 ≈ 卡数比 2）
   => 直接印证 §13.1 的"启动/提交成本主导"洞察。
3. ⚠️ **各 TP 度数的 greedy sha256 不同**：TP3 = f3edac19...（我们的参考），TP2/TP6 = 69207026...，TP4 = ccc284e4...。
   原因是张量并行改变行并行矩阵乘的**归约顺序** => 浮点结果有极微差异 => argmax 偶尔不同。
   **纪律**：验收的 sha256 门必须**按配置记录**，跨 TP 度数不可比；正确性 A/B 必须在同一 TP 度数下比较。

---

## 15. KV dtype 对**投机解码标尺**的影响（两臂复现，2026-09-21）

P0 阶段用 llama-bench 得到"f16 KV 优于 q8_0"（32K prefill +3.2% / decode +4.0%），本轮把它放到**主口径标尺**上复验
（两臂，正式协议：NPRED=512 + drop_caches，同源同库，唯一变量 = --cache-type-k/v）：

| 口径 | q8_0（今日两臂） | f16（今日两臂） |
|---|---:|---:|
| MEDIAN_TG | 96.43 / 99.37 | 92.92 / 91.28 |
| AL（3 prompt） | 5.55 / 4.22 / 6.38（两臂逐位相同） | 4.96 / 4.69 / 6.67（两臂逐位相同） |
| ms/轮（AL/tg） | 57.6 / 55.9 | **53.4 / 54.1（快约 4%）** |
| greedy sha256 | f3edac19...（稳定） | 69207026...（稳定，与 q8_0 不同） |

**结论**：
1. **标尺保持 q8_0 KV**：f16 每轮确实快约 4%（与 P0 的原始解码结论一致），但 **AL 低约 10%**，
   而 tg = AL / 每轮时间 => **tg 反而低约 5%**。=> **P0 的结论不能外推到投机解码标尺**。
2. **AL 对 KV dtype 高度敏感且可复现**（两臂 AL 完全相同）=> 任何影响 target 数值的改动（KV dtype、TP 度数、
   归约顺序、attention 配置）都可能改变 AL，因此**接受率必须作为一等指标**随每次改动测量（本项目已有此纪律，这里再补一个实证）。
3. **greedy 输出也随 KV dtype 改变**（f3edac19 -> 69207026，稳定复现）=> 验收的 sha256 门必须**冻结完整配置**
   （TP 度数 + KV dtype + attention 配置）；跨配置比较 sha256 无意义，A/B 必须同配置。
4. 顺带：f16 臂的 enqueue 更高（5.17-5.60 s / 286 轮 = 18.1-19.6 ms/轮，q8_0 为 5.86 s / 294 轮 = 19.9 ms/轮，
   基本同量级），说明差异主要来自 AL 而非提交开销。


3. **256K decode**：KV 带宽受限（q8_0 下每 token 每卡约 5.7 GB ≈ 6.3 ms）；若要改善需 KV 压缩或 attention 重构。
4. **push AR（独立子项目）**：需要信用/确认握手 + 3 卡确定性单测（见 §12.1），不要再无单测直上端到端。

---

## §16 Round 40（2026-09-21）：长上下文差距的机制定位 + 树一致性修复

### 16.1 卫生事故（已修，教训入档）
- 服务器 `/root/llm/test/v100-opt/llama.cpp` **根本不是 git 仓库**：命令 `git status --porcelain 2>/dev/null` 静默失败，
  我据此误判"树干净"。**远程非 git 树不能用 git 判断状态**。
- 实测漂移：服务器 `allreduce.cu` 仍含 push 实验码（`grep -c push_flag_stride` = **7**），
  且 `/root/libdir-instr/libggml-cuda.so` md5 = `7d4069d9...`、`strings | grep -c "push allreduce enabled"` = 1
  => **当时在案的库带实验代码，与文档记录的 HEAD 不一致**。
- 修法（权威同步）：本地 `git archive HEAD`（176 MB tar）-> scp -> 服务器解包覆盖源码 -> 重建。
  校验：`allreduce.cu` md5 = `c663d3b8...`、`push_flag_stride` 计数 = **0**。
- 重建后按正式口径跑 1 臂（`CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 NPRED=512`）：
  **MEDIAN_TG = 99.76 t/s，AL 5.55 / 4.22 / 6.38，贪心 sha256 = `f3edac19...`**
  => 与记录（96.43 / 99.37、逐位相同 AL、同一 hash）一致 => **记录数字有效、可复现**；
  同时证明 push 码（默认不激活）不改变性能与数值。

### 16.2 模型几何补全：这是**稠密**模型（MoE 假设全部作废）
自写 GGUF 元数据 dump（`/root/ggufdump.py`，纯 stdlib）：
`feed_forward_length = 17408`、**无 `expert_count`** => 稠密。
=> 27.05 GiB Q8_0 权重 / TP3 = **9.02 GiB 每卡每步**，按有效 ~800 GB/s 计 => **步下限 12.1 ms**。
（实测 target 步 32-34 ms，其中含 138 次集合通信的串行延迟 ~9.4-13 ms。）

### 16.3 roofline 三算（把"还有多少空间"钉死）
| 项 | 计算 | roofline | 实测 | 倍差 |
|---|---|---|---:|---:|
| 256K prefill | attn ~13.5 PFLOP + GEMM ~14 PFLOP = 28 PFLOP | ~230-290 s | **292 s（896 t/s）** | **约 1.0x => 已到顶** |
| 256K decode (M=1) | KV 每卡 ~4.6 GB/token（4 个 KV 头按 2+1+1 切） | ~5.7 ms/token | **54 ms/token（18.45 t/s）** | **9.5x** |
| target 步 (M=8/9) | 权重 9.02 GiB/卡 | 12.1 ms | 32-34 ms | 2.7x |
结论：**256K prefill 不再是问题**（1cat 的 3567-4069 t/s 是 32K/64K 深度，不可直接比）；
**长上下文 decode 是真正的洼地**；短上下文步的 2.7x 余量主要被集合通信延迟吃掉。

### 16.4 FA 在 Volta 上按形状派发（纠正"FA 改动影响解码"的误解）
- `fattn.cu:611` `can_use_vector_kernel`（D=256 为真）、`:638-642` `gqa_ratio_eff = 2`、`:644-652` Volta 分支。
- **M=1 -> VEC**（`:645`，原生读 q8_0，不做反量化）；**M=2..8 -> TILE**（`:648`）；**M>=9 -> MMA_F16**（`:651`）。
- => DFlash2 投机解码（n_max=7 => M 约 8）走 **TILE**；我们落地的 `Q_in_reg=false` 只作用在 **prefill**（M>=9）
  => 与实测一致（32K/256K prefill +11%/+42%，decode 无变化）。
- `fattn.cu:706-709`：**TILE 与 MMA 都强制 `need_f16_K/V = true`** => 量化 KV 每次调用都要**整段反量化成 f16**
  （缓冲区在 KQV 的 extra 空间，`fattn-common.cuh:53`；反量化函数 `dequantize_V_q8_0` 等，`:588`）。
  256K 时每层每步：K 约 285 MB 读 + 537 MB 写，V 同理 => 1.64 GB/层 => x16 层 ≈ **26 GB/步**（按头切分每卡约一半）。
  **这是投机解码在长上下文下的额外固定流量，VEC（M=1）没有。**

### 16.5 上游 split-KV 的真实状态（子代理取证 + 主线抽查）
- VEC/TILE 确实切分 KV：`fattn-common.cuh:1201-1203` 的 `blocks_num.y = parallel_blocks` 即 KV 切分索引，
  内核按 `gridDim.y` 跨步（`fattn-vec.cuh:250-256`、`fattn-tile.cuh:957-973`），归约在 `flash_attn_combine_results`（`:917-972`、`:1288-1295`）。
- 但 **`parallel_blocks` 由 occupancy / wave 效率决定，不随 n_kv 增长**（`:1126-1132`、`:1179-1199`；`:1132` 的 `ntiles_KV` 只是上限）。
- MMA 走 stream-K（`fattn-mma-f16.cuh:1870-1871`，`kbc` 空间均分 + `flash_attn_stream_k_fixup_*`）。
- => V100 + 256K + M<=8 只有约 **6-26 路 KV 并行**，每块仍要串行走 300-680 个 KV tile。
- 上游 issue **#28734 "CUDA: decode slows linearly with context"（2026-09-11，仍 open）** 与我们实测同源。

### 16.6 1cat 的对标做法（"提取 V100 专项优化"的直接答案）
- `vllm/platforms/cuda.py:149-159`：SM70 默认后端 **`FLASH_ATTN_V100`**（`VLLM_SM70_FLASH_ATTN_V100` 默认 1，`vllm/envs.py:358`）。
- **decode 明确 split-KV**：`flash-attention-v100/kernel/flash_decode_paged.cu:1038, 1061-1076`
  （`blockIdx.z = partition_idx`、`start_token_idx = partition_idx * PARTITION_SIZE`）+ 归约 `:2728`；
  partition size 256/512/**1024（seq_len >= 32768）**（`flash_attn_v100/flash_attn_interface.py:19-20, 599-608`）。
- D256 专项：`csrc/attention/sm70_v37/` + `cmake/patches/sm70_flash_attn_d256_{splitkv3,pipeline,k_pingpong,gqa_arch}.patch`。
  - 四个补丁的**具体战术**（本地只读仓库实读）：
    - `splitkv3`：给 dense decode 加 `sm70_d256_splitd_dense_splitkv3_fwd`，带 `partial_out/partial_max/partial_sum` => **显式 3 路 KV 切分**。
    - `pipeline`：把 FA2 的 `HEADDIM_SWITCH` 全替换成**只编 D=256 causal prefill** 的窄路径
      （`flash_fwd_d256_splitd_sm70.cu`、`flash_fwd_hdim256_causal_prefill_sm70.cu`；断言 `d==256 && causal && seqlen_q>=1024 && num_splits<=1`）=> **为单一形状极致特化**。
    - `gqa_arch`：新增 `flash_fwd_d256_gqa_arch_sm70.cu`，编入 tile 常量 `QK_TB_M=128 QK_TB_N=512 QK_WARP_M=64 QK_WARP_N=128`、
      `PV_TB_M=64 PV_TB_N=256 PV_WARP_M=32 PV_WARP_N=64`、`PREFIX_QK_FULL_STATS`、`PREFIX_QK_SKIP_APPLY` => GQA 专用的 D256 分块。
    - `k_pingpong`：注释原文 "The second K stage is live only before P is materialized, so it may alias the beginning of the later P region"
      => **用 smem 别名把 K 双缓冲塞进同样的共享内存**（`kKStageElements = 2240`）。
  => 他们的路线 = **为 D256/GQA/causal 单一形状写专用内核 + 显式 split-KV(3) + smem ping-pong**，而不是通用模板调参。
- 生产（`scripts/serve_qwen38_27b_nvfp4_v100.sh:81-86`）：
  `--attention-backend FLASH_ATTN_V100 --dtype half --kv-cache-dtype fp8_e4m3 --max-model-len 262144 --tensor-parallel-size 4 --block-size 2048`；
  权重 NVFP4，计算是 **fp16 HMMA**（`csrc/attention/sm70_v37/tail.cu:30-31`，`SM70_8x8x4_F32F16F16F32`）。
- => **KV 只做 fp8 存储（1 B/元素），算前展开 fp16，且带 partition 并行**；
  我们 = q8_0 存储 + **整段** f16 展开 + 切分度不足。注意 V100 无 FP8 张量核，1cat 的 FP8 也只是 KV 存储。

### 16.7 量化权重在 V100 上的算力天花板（新假设，**未验证**）
- V100 无 INT8 张量核；MMQ 依赖 `VOLTA_MMA_AVAILABLE`（`mma.cuh:152/838`）的**模拟 int8 mma**。
- 1cat 用 NVFP4 权重 + fp16 HMMA（marlin SM70：`csrc/quantization/marlin/sm70_marlin_*.cu{h}`，cutlass `default_mma_core_sm70.h`）。
- 与实测吻合：32K prefill 我们 2170 t/s vs 1cat 公开 3567-4069 t/s（1.6-1.9x）。
- **量化核算（本轮算出，把"假设"升级为"有量级支持"）**：
  - 32K prefill：GEMM 2 x 27e9 x 32768 = **1.77 PFLOP**，attention 约 0.21 PFLOP（12%）=> 合计约 **1.98 PFLOP**。
  - 实测 2170 t/s => 15.1 s => **131 TFLOPS / 3 卡 = 43.7 TFLOPS/卡**。
  - V100 的 dp4a（INT8）峰值约 4 x FP32 = **62.8 TOPS/卡** => **我们已到自身天花板的约 70%**。
  - 1cat 公开 32K/64K 3567-4069 t/s（4 卡）=> 约 **55-61 TFLOPS/卡**，与 **fp16 HMMA 峰值 125 TFLOPS 的 ~50% 效率**吻合。
  => **prefill 的 ~1.65x 差距是"天花板差距"（int8 模拟 vs fp16 张量核），不是调参问题**；
     要拿只能改用 fp16/低比特权重 + HMMA（1cat 的 marlin 路线），即换量化格式/自写 GEMM，属大改动。
  - 隔离实验（仍未做）：单卡 `test-backend-ops perf -o MUL_MAT` 比 q8_0 与 f16 权重的 prefill 形状吞吐；或用 f16 GGUF 跑 32K prefill 对照。

### 16.8 本轮判据与下一步
1. 长上下文 decode 的 **KV dtype x 深度曲线**（§16.9 正在跑）：若 f16 KV 在深上下文反超 q8_0 => **零代码可交付配置**（用户 256K 场景）。
2. 若要动内核：把 TILE/MMA 的"整段 f16 反量化"改成按 tile 反量化（或 TILE 支持 q8_0 直读），即 1cat partition 路线的等价物；**属大改动，先问用户**。
   - **补充实测（Round 40 源码核查）**：`fattn-vec.cuh:544-572` 只实例化 `cols_per_block = 1`（`Q->ne[1]==1`）与 `= 2`（其余），
     => **VEC 只适合 M<=2**，无法直接接管 M=8；TILE 内核签名即 `const half2 * V_h2`（`fattn-tile.cuh:563, 858`）=> **必须 f16 V**。
     => "让 TILE 直读 q8_0"**不是有界改动**（要在 TILE 的共享内存 staging 里加反量化，等于重写 V 通路）。
3. 量化权重 prefill 天花板：用 f16 GGUF 做 32K prefill 隔离实验。
4. push AR（信用/确认握手）仍是独立子项目，优先级低于 1 与 2。

## §17 长上下文实测（Round 44-45，卡 3/4/5，投机路径 DFlash2，3 prompt 固定 seed）

### 17.1 数据（`/root/lc-par.sh`，日志 `/tmp/lc-spec-lc32-*.txt`）

### 17.1.0 深度曲线（同一合成 prompt 家族，投机路径，卡 3/4/5，q8_0 KV，3 prompt 固定 seed）

| CTX | prompt tokens | prefill t/s | AL（中位） | tg（中位） | **ms/轮** | 图 rebuild | alloc_us/轮 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8192（正式标尺，卡 0/1/2，正式 prompt） | 短 | 2170 | 5.55 | 96.4-99.8 | **57.6** | 8% | 2.44 ms |
| 32768 | 26719 | 1502 | 2.81 | 46.35 | **60.6** | 48% | 12.2 ms |
| 65536 | 53043 | 1352 | 2.79 | 42.38 | **65.8** | **62%** | **15.1 ms** |

=> **每轮随上下文只涨 14%（8K->64K）**；prefill 从 1502 降到 1352 t/s；**AL 在 32K->64K 持平（2.81 -> 2.79）**。
=> **推论（重要，纠正上文的暂定说法）**：AL 对**深度不敏感**（至少 32K->64K），此前的"长上下文 AL 下降"是
   **合成重复源码 prompt 的内容效应**，不是上下文效应；正式标尺的 5.55 与这里的 2.8 **不可直接比较**。
=> **真正随上下文急剧恶化的是 host 侧图管理**：alloc 2.44 -> 12.2 -> 15.1 ms/轮，rebuild 比例 8% -> 48% -> 62%。
   这是长上下文下**独立于 attention 的第二个抓手**（根因同 rule 20：单 scheduler + 形状/视图交替），256K 时会更大。
=> 对照 §16.3：256K 实测 34.13 t/s（AL 约 5，投机）=> 每轮约 146 ms，说明 **128K 之后 attention/KV 通量才成为主导**（与线性 KV 增长的算式一致）。

### 17.1 KV dtype 对照（CTX=32768）
|---|---:|---:|---:|
| MEDIAN_TG | 46.35 t/s | **49.98 t/s** | 96.4-99.8 |
| AL（中位） | 2.81 | 3.06 | 5.55 |
| **每轮 ms（AL/tg）** | **60.6** | **61.2** | **57.6** |
| prompt1/2/3 tg | 51.68 / 46.35 / 38.48 | 50.08 / 49.98 / 46.75 | - |
| prefill（26699 tokens 均值） | 1502 t/s（24100/17791/17804 ms） | **1519 t/s**（18266/17583/17568 ms） | 2170（llama-bench 口径） |
| draft_decode / selector | 14.62 / 2.79 ms/轮 | 14.47 / 2.67 ms/轮 | 13.0 / 2.63 |
| 图 rebuild 比例 | 178/367 = 48% | 179/357 = 50% | 24/294 = 8% |
| alloc_us（含 prefill） | 4472 ms / 367 轮 | 4246 ms / 357 轮 | 2.44 ms/轮（纯解码） |

## §19 A2/A3 共同根因已定位（一个修法打两个目标）—— 2026-09-21

### 19.1 证据（源码核实）

1. **图的复用被 head 卡住**：`src/llama-graph.cpp:345-361` `llm_graph_input_rs::can_reuse()` 里
   `res &= head == mctx->get_head();`（`:357`）=> **只要滚动 head 变了，递归状态输入就不能复用** =>
   上层 `alloc_graph` 只能重建图（`rebuild` 计数上升、`alloc_splits` 变贵）。
2. **head 被编进图的写入路径**：`src/models/delta-net-base.cpp:491-493` 与 `:515-518`
   用 `ggml_view_2d(ctx0, conv_states_all, row_count, n_seqs, nb[1], (s_slot*mem_size + kv_head) * row_size)`
   作为 conv/state 的写入目标 => **视图的数据指针随 head 变化** => 图指纹（节点指针/视图）每次都不同 => 重建。
3. **修法所需的机制已存在**：`llm_graph_input_rs::set_input()`（`:329-343`）已经在做"**每次调用把值写进设备张量**"
   （它现在只用于 `s_copy` 的行索引）=> 把写入改成**索引式**（`ggml_set_rows` 路线，KV cache 已有同样用法
   `src/llama-kv-cache.cpp:1318-1350`）后，图里只留下**固定的张量**，head 只在 `set_input` 里改**索引值** =>
   图可跨轮复用（`can_reuse` 去掉 head 依赖，改为比较索引张量的形状）。

### 19.2 为什么值得做（两个目标一次拿下）

| 目标 | 现状（实测） | 预期 |
|---|---|---|
| **prefill / TTFT**（长上下文，§17.1.0） | 64K prefill 39.25 s，其中 `alloc_us` 合计 8.0 s（约 20%）；rebuild 与分块数吻合（312 块 -> 329 次） | **-15~20% prefill 时间**（256K 时 TTFT 293 s 量级同比例） |
| **draft 侧每轮图管理**（主指标） | draft 13.0-14.6 ms/轮，`reuse=0 / rebuild≈269`（rule 20 的结构性事实） | 若 draft 的递归状态同样依赖 head，则**每轮可省 8-12 ms**（=> 57.6 -> 约 46-50 ms/轮） |

=> **A2 与 A3 是同一个修法的两面**，优先级仅次于 A1（AR）。

### 19.3 改动面（估计，开工前先小步验证）

- `src/models/delta-net-base.cpp`：conv/state 写入改索引式（新增一个小索引张量，或在现有 rs 输入里加一路）。
- `src/llama-graph.cpp`：`llm_graph_input_rs` 增索引张量、`can_reuse` 去掉 head 依赖、`set_input` 写入索引值。
- `src/llama-memory-recurrent.*`：暴露 `s_copy`/`head` 的索引语义（让主机侧能填索引值）。
- 验证判据：**同配置 greedy sha256 逐位不变**（本改动不应改变数值，只改变图的结构复用）+
  `rebuild`/`alloc_us` 显著下降 + ms/轮下降 + AL 不劣化。

## §18 A1 工作流（AR 结构性改造）已启动：设备侧 push AR 小样

**工件**（2026-09-21，Round 46）：
- 源码：`/root/ar_push_test.cu`（服务器）与 `.dsh/tmp/ar_push_test.cu`（本地，可复现）；
  编译：`nvcc -O3 -arch=sm_70 -DUSE_NCCL -I$NCCL/include -o /root/ar_push_test /root/ar_push_test.cu -L$NCCL/lib -l:libnccl.so.2`
  （NCCL 路径 = `/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime`）。
- 设计（针对上次失败的根因）：
  1. **主机侧零轮询**：每次调用只发 1 个 kernel/卡（NCCL 路径是 3 次 set_device + group 调用）。**这是与上次 126 us 原型的关键差别**（上次在主机侧轮询 flag）。
  2. **slot + epoch 方案**：SLOTS=8，每次调用的 slot = call % SLOTS，flag 写入对端并比较 `>= epoch`；
     `pub_cnt[slot]` 单调累加、目标值 = use_count * gridDim（**不需要复位**）=> 天然避免跨迭代覆盖（上次的正确性根因）。
  3. 固定求和顺序（w=0,1,2）=> 结果**逐位可复现**。
  4. **自旋有上限**（`spin_cap = 1<<22`）+ 超时计数 `err`，超时会打印 "spin timeouts per rank" =>
     **不会把 GPU 挂死**（实验代码必须的安全网）。
- 测量口径：同一 harness 内并联 **NCCL 基线**（同样的 host-loop、同样的 event 口径），分别报
  **host us/call** 与 **GPU 可见 us/call（rank0 的 ev0..ev1）**；三种规模：164 KB（在役尺寸）/ 1 MB / 10 MB。
- **判据**：164 KB 下 1000 轮 **0 mismatch**、**spin timeout 0**、GPU 可见 **<=15 us**（NCCL 在役 68-95 us）。

### 18.2 A1 小样结果（v2 协议：逐块 flag、无全局屏障、slot+epoch）

命令：`LD_LIBRARY_PATH=$NCCL/lib CUDA_VISIBLE_DEVICES=0,1,2 /root/ar_push_test <n> <iters>`
（v1 = 有 pub_cnt/go_flag 两道网格屏障；v2 = 逐块 flag，块 k 只等对端块 k）

| 规模 | 指标 | **push v1** | **push v2** | **NCCL**（同 harness、同预热） |
|---|---|---:|---:|---:|
| 160 KB/rank | 主机 us/次 | 10.12 | **9.5-10.1** | 36.6-38.6 |
| 160 KB/rank | GPU 可见 us/次 | 69.08 | **37.4 / 47.6** | **36.7** |
| 1 MB/rank | GPU 可见 us/次 | 102.5 | 93.0 / 93.2 | **55.2** |
| 正确性 | - | 1000 轮 0 错 | **1000 轮 0 错、0 自旋超时** | - |

**结论**：
1. **正确性问题已解决**（这是上次失败并回退的根因）：`slot = call % SLOTS` + flag 比较 `>= epoch` + `pub_cnt` 单调累加（v1）/ 逐块 flag（v2）
   => 1000 轮跨迭代 slot 复用**零错误**；自旋有上限（`SPIN_CAP`）且有超时计数，**不会挂死 GPU**。
2. **主机侧便宜 3.7x**（10 us vs 37 us）=> 这是 R1 能确定拿到的部分。
3. **GPU 侧在役尺寸（160 KB）只是打平**（37-48 vs 37 us），1 MB 时 NCCL 更快（55 vs 93）=> **R1 单独不够**。
4. 1cat 的 18 us/次是**图内**数字（设备端等待，无主机 gap；`custom_all_reduce.cuh:1944-1955` 只在 capture 时启用 push kernel）。
   => **R1（主机侧更便宜）预期 +5~12%；要拿满必须 R2（入图）**，而 R2 需要元后端不在 AR 处切图（上游架构级，风险高，需用户批准）。

### 18.3 A1 集成已落地（env 门控，默认行为不变）

**改动（三个文件，全部在 `llama.cpp/` 内，ASCII）**：
- `ggml/src/ggml-cuda/allreduce.cuh`：新增不透明类型与三个接口
  `ggml_cuda_ar_device_{init,free,allreduce}`（与既有 `ggml_cuda_ar_pipeline_*` 同风格）。
- `ggml/src/ggml-cuda/allreduce.cu`：新增**设备侧 push AllReduce**（小样 v2 协议原样搬入）
  - `ggml_cuda_ar_dev_push_kernel`：块 k 把本块 chunk 发布到每个对端（peer access），逐块 flag（`atomicExch(write) + volatile read(spin)`），
    块 k 只等对端块 k；**无网格级屏障**；自旋上限 `1<<22` 且超时计入 `err`。
  - slot/epoch：`SLOTS=4`，`slot = (epoch-1) % 4`，flag 比较 `>= epoch`，epoch 单调递增 => **跨迭代复用安全**（小样已 1000 轮验证）。
  - 容量 `cap = 262144` floats/分片（1 MiB）：**超过容量的调用返回 false**，交给常规实现（=> prefill 的大张量仍走 NCCL/BF16 路径）。
  - 只接受 **F32 + 连续 + 同元素数** 的张量，非 `GGML_TENSOR_FLAG_COMPUTE` 的分片发布 0（与 NCCL 路径的 memset 语义等价）。
- `ggml/src/ggml-cuda/ggml-cuda.cu`：`comm_context` 加 `ar_device` 字段、析构释放、
  `ggml_backend_cuda_comm_init_device()`（**只在 `GGML_CUDA_AR_DEVICE` 存在时启用**）+ `try_allreduce_device`，
  并在 `comm_init_nccl` 开头尝试；失败则照旧回退 NCCL => **未设 env 时行为与上游逐位相同**。

**验收口径（重要）**：本改动**改变归约顺序** => greedy sha256 会变（与 §16 的 KV dtype 同理），
因此**不能用逐位 sha256 做门**；按 goal 验收标准 3 对本类改动用：**AL 不劣化 + 每配置 >=2 臂 + 报 ms/轮**，
外加"sha256 在同配置内可复现"作为稳定性检查。

**待跑**：`build-instr` 编译（`/tmp/ardev-build.log`）-> 冒烟（`GGML_CUDA_AR_DEVICE=1` 起服务、看 `[AR] ar_us_avg` 是否从 68-95 us 下降）
-> 正式 A/B（关/开各 >=2 臂，drop_caches，报 AL 与 ms/轮）。

### 18.1 子代理取证：1cat 的 allreduce **确实在捕获图内**（回答 R1/R2 取舍）

- 机制：**CUDA-IPC 共享内存 push，每次调用一个 kernel，不走 NCCL**
  （`csrc/custom_all_reduce.cu:492` -> `csrc/custom_all_reduce.cuh:1905`；SM70/V100 TP4 push 变体 `sm70_cross_device_reduce_1stage_push<4>` 定义在 `.cuh:762`、launch 在 `.cuh:1950-1953`；
  IPC 缓冲在 `__init__` 建一次：`custom_all_reduce.py:296-302`、`:341-347`，并在 `capture()` 里集群级注册 `:386-418`）。
- **关键证据（在不在图内）**：`custom_all_reduce.cuh:1944-1955` 的 push kernel **只在 `status == cudaStreamCaptureStatusActive` 时才 launch**；
  注释原文："The push protocol amortizes peer polling across captured collective chains. A lone eager call stays on the ordinary registered-buffer pull path."
  => 该实现是**为"活在捕获图里"而设计的**。
- 量化：NSYS 图节点桶 `docs/design/sm70_dflash2_target_graph_20ms.md:308-311`：**TP4 all-reduce 128 节点 / 2.313 ms**（在 1257 节点的 target 图内）
  => **约 18 us/次**；另有 `:454` "1.277 ms TP4 push all-reduce"。轮时锚点 `sm70_quasar_nvfp4_dflash2_acceptance.md:125-128` = **17.552 ms 均值**。
- allreduce **不在任何 split 列表**（`vllm/config/compilation.py:756-779` 只含 attention/GDN；`:1143` `splitting_ops`）=> **不打断图**。
- 生产是 TP4 + push AR **默认开**（`vllm/envs.py:277`、`:280`，默认值 "1"）。
- **对我们的含义**：
  1. 目标是"**每次 AR 约 18-25 us 且主机侧零轮询**"。我们现状 68-95 us => 138 次/轮可省 **约 9-13 ms/轮**。
  2. 完整复刻（AR 进图）在 llama.cpp 里需要 meta 后端不再在 AR 处切图（上游架构级，风险高）；
     **但先做到"主机侧零轮询的自定义 AR"即可拿到大部分收益**（AR 仍是主机调用，但每次只发 1 个 kernel）=> 这正是 §18 小样在测的东西。
  3. 若小样达到 ~20 us，则预期：57.6 ms/轮 -> **约 46-48 ms/轮**（tg 96 -> 约 115-120）。

### 17.1.1 反推纠正：`rebuild` 的增长来自 **prefill 分块**，不是解码轮退化

用新数据算：`rebuild` 与 prefill 分块数**精确吻合**
- 64K：3 prompt x ceil(53043/512) = **312** 块，实测 rebuild = **329**（差 17，即少数解码轮）
- 32K：3 x ceil(26719/512) = **156** 块，实测 rebuild = **178**（差 22）
- 8K（正式标尺的短 prompt）：几乎无 prefill 分块，实测 rebuild = 24/294

=> **结论修正**：`rebuild` 比例上升与 `alloc_us` 变大，主因是**长 prompt 的分块 prefill**（每块一次 `alloc_graph`），
   **不是**"解码轮随上下文变慢"。=> 工作流 A2 的作用域应重新定位为 **TTFT/prefill 抓手**（64K prefill 39.25 s 里估计有数秒是图分配），
   解码轮的主机侧开销仍以 ~2.4-5 ms/轮计（与 8K 同量级）。
   => A2 的改法不变（索引式写 + 让连续同形状分块复用图），但**预期收益记在 prefill/TTFT 上，不要记在 decode 上**。

### 17.1.2 TP 度数 @64K：**TP4 反而更差 => 保持 TP3**

| CTX=65536，q8_0 | TP3（卡 0/1/2） | TP4（卡 0/1/2/3） |
|---|---:|---:|
| MEDIAN_TG | **42.38 t/s** | 37.55 t/s（-11%） |
| AL（中位） | 2.79 | 2.68 |
| **ms/轮** | **65.8** | **71.4（+8.5%）** |
| prefill | 1352 t/s | 1318 t/s（-2.5%） |
| AR 均值 | **431.6 us** | 527.2 us（**+22%**） |
| rebuild / alloc | 329/530、8.0 s | 333/530、9.8 s |

=> 即使每卡权重从 9.02 GiB 降到 6.76 GiB、每卡 KV 通量减少，**仍更慢**：因为 AR 从 3 卡变 4 卡后每次更贵（+96 us x 138 次/轮）
   => **AR 主导的结论再次被独立证实**（与 §16.11 的账本一致）。
=> **配置结论：短/中上下文保持 TP3**。保留意见：256K 时 KV 项比 64K 大 4 倍，平衡点可能移动，未测（成本高）。

### 17.2 结论

1. **KV dtype 在 32K 无实质差别**：每轮 60.6 vs 61.2 ms、prefill 1502 vs 1519 t/s（均在离散度内）。
   => 工作流 C 的"f16 免掉整段 f16 反量化"预期**在 32K 不兑现**（该开销按 roofline 估算约 3.3 GB/步 ≈ 4 ms，且与 GPU 计算重叠）。
   => **继续用 q8_0 KV**（显存减半 => 可换更长上下文或更少卡）。
   => tg 的 7.8% 差异来自 AL（3.06 vs 2.81），而 q8_0 臂的 prompt1 与已作废的并发臂重叠（wall 28 s vs 22 s）=> **q8_0 臂被低估**，差异不可信。
2. **32K 的长上下文代价很小**：每轮 60.6 ms vs 8K 的 57.6 ms => **仅 +5%**（+3 ms/轮）。
   => 本模型 16 层全注意力 + 4 KV 头上，32K 上下文的 attention（含 TILE 路径的整段反量化）**只值约 3 ms/轮**；
      真正的长上下文墙在 128K-256K（KV 线性增长 + 每卡 KV 通量），见 §16.3。
3. **⚠️ 长上下文吞吐下降的主因是接受率（AL），不是速度**：tg 从 96 掉到 46-50，其中每轮只涨 5%，
   而 **AL 从 5.55 掉到 2.81-3.06（-45%）**。
   **重要保留**：本臂用的是**合成重复源码 prompt**（91 KB 前缀），AL 的绝对值不可与正式标尺（3 个通用短 prompt）相比 =>
   "AL 随上下文下降"这一条**受 prompt 内容混淆，未定论**；**已定论的是每轮时间只涨 5%**（这一项与 prompt 内容无关）。
   => 若后续要定论 AL，必须用**同一批 prompt 在不同深度**（例如把正式 3 prompt 前面接一个固定长前缀），本次没做。
4. **新发现（host 侧随上下文恶化）**：32K 时 **约 50% 的轮会重建图**（8K 只有 8%），`alloc_us` 也显著变大。
   => 与 §16 rule 20 的"单 scheduler + 形状交替"是同一个根因，且在长上下文下更贵（KV cpy 图更大）。
   => 这是长上下文下**除 attention 之外的第二个抓手**（属结构性改动）。

### 17.3 待测（已在服务器排队，接手直接看日志）

- `lc64-q8` / `lc64-f16`（CTX=65536，正在跑）
- `lc64-tp4-q8` / `lc64-tp4-f16` / `lc64-tp6-q8`（TP 度数 @64K，等 PAR_DONE）
- `/root/fa-split-probe.sh`：编译 + 扫 `GGML_CUDA_FA_SPLIT_FLOOR`（等 CHAIN2_DONE）
- M=1 的 32K/128K 深度点：**必须独占机器重测**（§16.14 的并发教训）

### 16.14 已作废的测量：并发导致的 d32768 点（纪律教训）

Round 44 我为了让"投机长上下文标尺"早点出数，把它并行放到 3/4/5 卡（与 0/1/2 上的 `llama-bench` 深度扫描同时跑）。
结果：`tg64 @ d32768 = 13.45 +- 6.62 t/s`（**离散 49%**）=> 反推 `2.06 us/KV-token`，与既有 256K 实测（54 ms/token）**自相矛盾** => **该点作废，不入账**。
原因：本机每轮瓶颈在**主机侧**（元后端 417 次提交 + 采样 + selector 线程）与**页缓存/NFS**（两个 29 GB 模型），这两者都是跨卡共享的，与"卡不重叠"无关。
=> 纪律化：**同一台机器同一时刻只跑一个测量任务**（已加入 AGENTS.md §4.24）。
=> 有效数据：`tg64 @ d8192 = 40.19 +- 3.15 t/s`（24.9 ms/token，**该臂在并发开始前完成，有效**）。
=> 待办：M=1 的 32K/128K 深度点需要**独占重测**（下一会话，用 `/root/lc-sweep.sh` 的单臂即可）。

### 16.13 元后端的图切分结构（源码核实，解释"每轮 417 次 graph_compute"）

`ggml/src/ggml-backend-meta.cpp:2434-2462`：图被切成 **n_subgraphs 段**，每段对**所有后端各调用一次** `ggml_backend_graph_compute_async`（`:`2437），
段与段之间做一次 allreduce（`:`2453）；末段之后不归约（`:`2443）。
=> 每轮 target = **139 段 x 3 后端 = 417 次 graph_compute** + **138 次 AR**，与实测"每轮 387-422 次、平均 40 节点"完全吻合。
=> 每段的 GPU 工作量很小（约 40 节点），而每段的**主机侧固定成本**（dispatch + buffer 映射 + CUDA graph launch）与 AR 的主机成本才是每轮的主导 —— 这就是 1cat 用整轮 fullgraph 拉开差距的地方。

**NCCL AR 单次成本分解（`ggml-cuda.cu:1000-1034`，我们的参数：ne 约 40960、3 后端）**：
- 命中"小张量走 FP32"分支（`:`1019：3 后端且 ne < 131072）
- 每次调用 = 3 x `ggml_cuda_set_device` + （仅非 COMPUTE 张量）`cudaMemsetAsync` + `ncclGroupStart` + 3 x `ncclAllReduce` + `ncclGroupEnd`
- => 95 us 里绝大部分是**主机侧 NCCL 入队与 3 次设备切换**，而不是 164 KB 的传输（NV2 理论约 3.5 us）。
- **推论**：任何"微优化"（去 memset、去 set_device、换 ring/P2P 拷贝）都只有 us 级收益 —— **必须走结构性修法（R1 设备侧 AR / R2 入图捕获）**。

### 16.12 allreduce 延迟的本质：**主机介导、未被图捕获**（源码核实）

- 源码：`ggml/src/ggml-backend-meta.cpp:2453` `backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());`
  —— **元后端在 `graph_compute` 之间调用 allreduce**；CUDA 后端只是把它注册成接口函数（`ggml-cuda.cu:5995-5996`）。
- => 每轮 138 次 AR **都不在 CUDA graph 内**：每次都是一次主机发起的独立操作（NCCL enqueue + 内核 + 依赖 gap），
  而**下一段图依赖它的结果** => GPU 在这段时间**空转**（这正是 §16.11 里"12.1 + 13.1 = 24.9"能闭合的原因）。
- => 与 1cat 的真正差别不是"NCCL vs push"，而是 **1cat 的整轮 fullgraph 让 AR 也进了图**（其 6.6-11 us 是 GPU 侧、无主机往返）；
  我们的 push 原型之所以更慢（126-161 us）：**它在每次调用里加了主机侧 flag 轮询**，等于把主机往返放大。
- **两条修法（都属结构性改动，需按 §2 走审批）**：
  - **R1（设备侧、无主机往返的 AR）**：把 push/credit 机制全部做成设备端（生产者/消费者都在 GPU 上自旋），主机只发一次 launch。
    难点：跨迭代的 slot 复用需要 credit/ack（上次失败根因），且必须有 3 卡确定性单测。
  - **R2（把 AR 纳入图捕获）**：让 AR 边界不切断图（或让 CUDA 后端把 AR 作为图内节点）。这正是 goal 里的 P2（"跨设备整轮图捕获"，预期 -9 ms/轮）。
- **判据（无论走哪条）**：in-situ `[AR] ar_us_avg` 从 68-95 us 降到 <=15 us，且 greedy sha256 不变、AL 不劣化。

### 16.11 判决定论：普通解码 = 权重流 + 集合通信（两条独立测量互相印证）

Round 42 实测（canonical 构建，`llama-bench -m Qwen3.8-27B-Q8_0 -d 8192 -n 64 -r 2 -sm tensor -ts 1/1/1 -fa on -ctk/-ctv q8_0`）：

```
tg64 @ d8192 = 40.19 +- 3.15 t/s   =>  24.9 ms/token（普通解码，M=1，无投机）
```

对照本会话独立算出的两个下限：
| 成分 | 数值 | 来源 |
|---|---:|---|
| 权重流（9.02 GiB/卡 @ ~800 GB/s） | **12.1 ms** | 27.05 GiB Q8_0 / 3 卡（GGUF 实测参数） |
| 138 次集合通信 x 94.9 us（无偏） | **13.1 ms** | GGML_CUDA_AR_TIMING + 关图对照臂 |
| **合计** | **25.2 ms** | 实测 **24.9 ms**（误差 1%） |

=> **结论 1**：普通解码已经**正好等于"权重流下限 + 集合通信"**，说明
（a）权重流已接近硬件 roofline，没有"读得慢"的余地；
（b）**target 步里约 40-50% 是 allreduce 延迟**，这正是 1cat（17.463 ms/轮、其自研 push 式设备 IPC allreduce 约 6.6-11 us）与我们（94.9 us）差 2.3x 的主因；
（c）投机解码轮 57.6 ms = target 32-34 + draft 13 + selector 2.6 + 残差 6-8，而 target 32-34 = 12.1（权重）+ 13.1（AR）+ 其余（M=8 的额外激活与尾部）=> **账本闭合**。

**推论（量化路线图，用于排优先级）**：
| 改动 | 假设效果 | 每轮 ms | tg |
|---|---|---:|---:|
| 现状 | - | 57.6 | 96 |
| AR 降到 10 us（1cat 式 push/设备 IPC） | -13.1 -> -1.4 ms | **~46** | ~121 |
| 再 + draft 侧 13 -> 5 ms（融合/整轮图） | -8 ms | **~38** | ~146 |
| 再 + M=8 前向本体（图捕获/调度残差） | -8 ms | **~30** | ~185 |

=> **验收标准 1（>=180 t/s、约 20 ms/轮）在数学上要求三件事同时成立**（AR 重写 + draft 侧减半 + 图/调度残差消除），
单靠任何一项都到不了；而 AR 重写是**唯一能一次拿 13 ms** 的一项。

### 16.10 已落地的探针（Round 41，env 门控 + 默认行为不变）
`ggml/src/ggml-cuda/fattn-common.cuh`（launch_fattn，非 stream-K 分支，插入点在 `blocks_num.y` 与 `dst_tmp.alloc` 之前）：
```
// The scan above keeps the split at one wave, so for long contexts each block still
// walks a long serial KV loop. GGML_CUDA_FA_SPLIT_FLOOR raises the split to probe this.
if (const char * env = getenv("GGML_CUDA_FA_SPLIT_FLOOR")) {
    parallel_blocks = std::max(parallel_blocks, std::min(atoi(env), ntiles_KV));
}
```
- 理由：上游把 `parallel_blocks` 限制在**一波**（wave 效率扫描，`fattn-common.cuh:1179-1199`），长上下文时每块仍串行走 300-680 个 KV tile；
  提高切分度是"用并行度换每块串行长度"的直接探针（代价：combine 流量 `O(parallel_blocks x M x heads x D)` 与启动延迟）。
- **默认零影响**（env 未设时行为与上游逐位相同）；用 `/root/fa-split-probe.sh` 在 `-d {32768,131072,262144}` 上扫 floor={0,64,256,1024}（NODROP 诊断口径）。
- 若有效：改默认需重新过验收（**切分度会改变归约顺序 => greedy sha256 会变**，必须按 §16 的纪律冻结配置后重测）。

### 16.9 正在跑的实测（脚本 `/root/lc-sweep.sh`，日志 `/tmp/lc-sweep.log`）
canonical 重建 + `llama-bench -sm tensor -ts 1/1/1 -fa 1 -p 0 -n 64 -r 2`，
`-d {8192, 32768, 131072, 262144}` x `-ctk/-ctv {q8_0, f16}`，另加 `-fa 0` 诊断两臂（8K/32K）。
判据：每 KV-token 的 ms 斜率 vs §16.3 的 roofline 倍差（结果回来后补在 §17）。
















