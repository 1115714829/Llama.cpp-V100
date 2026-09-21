# PLAN：把 DFlash2 每轮 57.6 ms 收到 ~20 ms（= tg 96 -> 180+）

> 依据：2026-09-21 DSH 会话全部实机实测（`SESSION-2026-09-20-measurements.md` §16-§20、`HANDOFF.md` 顶部）。
> 所有数字都可复核；已证伪项已剔除（见文末"不做清单"）。

## 1. 当前账本（8K 投机标尺，AL 5.55，56.6 ms/轮，四次实测离散 <0.5%）

| 成分 | ms/轮 | 依据 | 可动性 |
|---|---:|---|---|
| target 权重流（M=8） | 12.1 | 9.02 GiB/卡 @ ~800 GB/s = roofline | 已到顶 |
| **AR（138 次，事件计时 53.0 us）** | **7.3** | `[AR] ar_us_avg=53.0` | **可降到 ~2.5（R2）** |
| M=8 相对 M=1 的增量（激活/GDN） | ~6.5 | pp8 30.8 - pp1 24.3 ms | 需内核/融合工作 |
| M=1 其余固定成本 | ~5.5 | 普通解码 24.9 - 12.1 - 7.3 | 部分可动 |
| 图 alloc/复用（target） | ~2.2 | `[RT] alloc_us` | 分桶后可降 |
| **draft 侧（注入 + 块前向）** | **13.6** | `draft_decode=13.61`；权重仅 1.14 GB => 非带宽受限 | **可降到 ~5（结构性）** |
| CPU selector | 2.7 | 已并行化（8 线程） | 已优化 |
| **未归因余量** | **~6.7** | 差额；已定位到采样器/接受记账/服务器簿记 | 待仪表化 |

**目标 ~20 ms 需要四项同时成立**：AR 7.3->2.5、draft 13.6->5、余量 6.7->2、以及 M8 增量/alloc 各削一部分。

## 2. 四项结构性工作（按"收益/风险"排序）

### P-A：AR 入图捕获（R2）—— 唯一能一次拿约 4.8 ms/轮的一项
- **现状**：AR 由 meta 后端在 `graph_compute` 之间**主机侧**调用，不在 CUDA 图内（`ggml-backend-meta.cpp:2453`）；
  每轮 139 段 x 3 后端 = 417 次图提交 + 138 次 AR。事件计时 53.0 us/次。
- **1cat 的做法**：CUDA-IPC push AR，**只在 `cudaStreamCaptureStatusActive` 时启动**（`csrc/custom_all_reduce.cuh:1944-1955`），
  NSYS 图节点桶里是 **18 us/次**（128 节点 / 2.313 ms，在 1257 节点 target 图内）。
- **为什么必须入图**：本会话已证明"主机侧更便宜"换不来轮时（R1 证伪：enqueue 省 4.8 ms/轮，轮时零改善）=> 只有把 AR 变成图内节点才能省 GPU 可见时间。
- **改法（两条）**：① 让 meta 后端不在 AR 处切图（把 AR 表达为图节点/自定义 op）；② 或让 CUDA 后端把 AR 作为图内节点捕获。
  两条都要动上游架构级代码 => **需批准**。
- **验收**：`[AR] ar_us_avg` <= 25 us、每轮 <= 50 ms、greedy sha256（同配置可复现）、AL 不劣化。

### P-B：draft 侧整轮单图 / 静态形状 —— 最大单项（13.6 -> ~5 ms/轮）
- **现状**：draft 每轮 `reuse=0 / rebuild=556`（100% 重建）；每轮两个不同形状的图（注入 + 块前向）交替，
  而 `ggml_backend_sched` 只有单份 splits/graph => 每轮必然 reset+重切+重分配（`alloc_splits` ~5 ms/次）。
- **已排除的局部补丁**：`GGML_SCHED_SPLIT_CACHE`、按批大小分槽的 arena（均实测无效）。
- **改法**：把 draft 的两个图合并为一次提交（1cat 的"注入即投影"路线）或让 draft 上下文有独立 scheduler；
  属新子系统 => **需批准**。
- **验收**：draft 侧 `rebuild` 显著下降、`spec timing: draft_decode` <= 6 ms、每轮 <= 50 ms、AL 不劣化。

### P-C：mask/KV 宽度分桶 —— TTFT 专项（prefill 的约 20%）
- **现状**：`llama-graph.cpp:48-65` `can_reuse_kq_mask` 要求 `kq_mask->ne[0] == n_kv`，K/V 也是按 n_kv 切出的视图
  => 每个 prefill 块形状都变 => 每块重建图。64K 时 `alloc_us` 合计 **8.0 s / 39.25 s prefill = 约 20%**（312 块）。
- **改法**：mask 与 K/V 视图按桶对齐（如 512/1024），桶内未用区间填 `-inf`；**会改变 attention 读数范围** => 必须重验 sha256/AL/perplexity。
- **验收**：64K/256K prefill 时间下降 >=10%、`rebuild` 比例大幅下降、greedy sha256 逐位一致（若纯 padding 则应一致）。

### P-D：长上下文 decode 的内核吞吐（256K：34 -> ?）
- **现状**：256K decode 每 token 54 ms，而 KV 带宽 roofline 约 5.7 ms（9.5x）；**A4 已证伪"并行度不足"**
  （更深切分单调变差 26.85 -> 22.43 t/s）=> 是 **VEC/TILE 内核在长 KV 下的有效带宽只有约 105 GB/s**。
- **改法**：写 D256/GQA 专用长上下文 attention 内核（1cat 的 `FLASH_ATTN_V100` 路线：partition split-KV + 向量化 KV 加载 + 原生量化读数；
  其 `cmake/patches/sm70_flash_attn_d256_*.patch` 与 `csrc/attention/sm70_v37/` 是现成参考）=> 新内核，**需批准**。
- **验收**：256K decode >= 60 t/s（先 2x），greedy sha256 逐位一致（纯 kernel 重写应保持一致）、长上下文标尺复测。

## 3. 不需要批准、可立即做的两项（低风险）

1. **6.7 ms 未归因余量仪表化**（已定位区间：采样器、投机接受/校验记账、服务器每轮簿记）：
   各加一个 `LLAMA_SPEC_TIMING` 门控计时即可定量；若确认为软件开销 => 直接优化，无需架构改动。
2. **调试探针规整**：树里 7 处 `[RT]` `fprintf` 收拢为统一的 env 门控探针（代码卫生 + 交接友好）。

## 4. 明确不做（已证伪，勿重试）

MoE 方向（模型稠密）｜NCCL 调参｜**设备侧 push AR 的"省主机"路线**（R1，轮时无收益）｜
**递归状态索引式写入**（A2，逐位中性但 prefill 真因是 mask/KV 宽度）｜**更深的 FA KV 切分**（A4，单调变差）｜
让 VEC 接管 M=8｜`VLLM_SM70_USE_BREAKABLE_CUDAGRAPH`｜AR 的 us 级微优化｜并发测量（会污染结论）。

## 5. 复现与状态

- 正式数字：`CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 NPRED=512 bash /root/p60-ab-harness.sh`（带 drop_caches）。
- 长上下文标尺：`/root/lc-spec.sh`（3 prompt、固定 seed、报 tg/AL/中位数）；深度/ dtype 扫描：`/root/lc-sweep.sh`。
- 代码状态：`llama.cpp` 干净 at `c2d716519`；实验补丁在 `patches/0001..0006`；判决与预算见 `SESSION` §16-§20 与 `HANDOFF` 顶部。
