# PLAN-EXEC：DFlash2 on V100 —— 55.5 -> <=37.0 ms/轮（tg >= 150）

> 状态：**用户已批准**（2026-09-21，plan mode -> approved）。这是执行计划，不是账本；账本在 `PLAN-to-180ts.md`，依赖图在 `PLAN-GRAPH.md`。
> 每次改动落地后，本文件顶部的「进度」段必须更新。

## 进度
| 步骤 | 状态 | 证据 |
|---|---|---|
| P0 计划落盘 | **完成** | 本文件 + `PLAN-GRAPH.md` 指针 |
| M1 draft 差分臂 | 待跑 | — |
| M2 派发数普查 | 待跑 | — |
| M3 注入图定位 | 待做 | — |
| 线 A 注入图定长化 | 待做 | — |
| 线 B draft 13.8 -> 2.6 ms | 待做 | — |
| 线 C alloc/派发 early-out | 待做 | — |
| 线 D selector 上 GPU | 备选 | — |

## 0. 目标与验收
- **硬指标**：3 卡 TP + NCCL + P2P，DFlash2 n_max=7 + Qwen3.8-27B-Q8_0，**tg >= 150 t/s**，即 AL ~ 5.55 下 **ms/轮 <= 37.0**。
  **基线（本会话权威 512 口径实测）**：**97.03 t/s**，ms/轮 **53.8 / 55.8 / 56.3**，AL **5.58 / 4.25 / 6.38**（p1/p2/p3）。
- **验收口径**：同源 A/B（同 build dir、同 CMake 参数、四库 md5 + 二进制标记串）；每配置 >= 2 臂并报 `+-`；
  **ms/轮 与 AL 并列报**（tg = AL / ms/轮）；正确性门 = **同配置** greedy sha256 `f3edac19446ef641447f8391c71cb1b25a74055d662e9887890fa4a98602ca34`；
  正式数字带 `drop_caches`，诊断用 `NODROP=1`；**同一比较内探针集不得改变**。
- **不改模型/精度**：仍是 Q8_0 target + DFlash2-Q4_K_M draft。

## 1. 已确证的事实（设计约束，全部本会话实测）

### 1.1 每轮账本（3 卡，权威 512 口径）
| 成分 | ms/轮 | 来源 |
|---|---|---|
| target 整步 | ~34 | 权重流 22.1 + AR ~8 + 注意力 ~2 |
| **draft 前向（DFlash2 独有）** | **13.8-14.4** | `LLAMA_SPEC_TIMING`；权重仅 1.14 GB |
| **CPU selector** | **4.3** | 并行化后实测 |
| 合计 | 53.8-57.4 | = tg 97.03 |

### 1.2 关键实测
1. **draft 是「一次前向出整块」**（`common/speculative.h:52,61` block draft）=> 1.14 GB 在 438 GB/s 下应 **2.6 ms**，实测 **13.8 ms（83 GB/s，差 5 倍）**。
2. **主机侧逐节点启动是主导项**：`[META] calls=832 sub/call=47.6 ar/call=46.6 | total=10.902 loop=9.648 dev=6.890 ar=2.754 ms/call | prologue=1.254 (11.5%)`；
   `[GRAPH] calls=82688 capture=2742 replay=64875 (78.5%) direct=15071 (18.2%) per_call=5.9us avg_nodes=40`；每轮约 2 万次节点启动。
3. **加卡单调变差**（同口径三臂）：TP3 **97.03** / TP4 68.66 / TP6 67.31 t/s；ms/轮 55.8 / 61.5 / 74.5；
   机制 = `dev` 6.890 -> 9.032 -> **15.208** ms/call（tensor 切分下每节点在每卡各启一次）+ 单次 AR 53.8 -> 85.1 -> 125.2 us => **固定 3 卡**。
4. **KV 线在 8K 封顶 ~2 ms/轮**：密集阶梯（256->131072，r=8，升序，一次加载）斜率恒为 **0.1028 us/ctx-token**；8K 深度项 0.84 ms/token，128K 为 13.4 ms/token。
5. **direct 的成因是属性真的在变**：`[FPD]` 抓到注入图 `ne1 = 4/8/2/1`（每轮被接受的 token 数）。N6a（rebuild 复用，commit `92a1566c9`）机制生效
   （`[MKEY]` 的 n0 开始循环复用）但**不省时间**，且 `[GRAPH]` 计数逐位不变 => 只跳过 rebuild 不够。
6. `ggml-alloc.c:1052` `ggml_gallocr_alloc_graph` **无快路径**，实测 **alloc = 5437 us/call**（split 的 21 倍）。
7. **MTP 对照（量尺，非路线）**：同机同库 MTP n=4 每轮 **39.7-43.2 ms**（AL 2.93-4.39），证明一轮 40 ms 在这台机器上可达。

## 2. 阶段 0
### P0 计划落盘 —— 完成（本文件）
### M1 draft 差分臂（1 次独占机器，约 15 分钟）
- 两臂，其余全同（`CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 NPRED=512 NODROP=1`，`GGML_META_HOST_TIMING=1 GGML_CUDA_GRAPH_DEBUG=1`）：
  `SPEC=--spec-type none` vs 默认 DFlash2（`--spec-type draft-dflash --spec-draft-n-max 7`）。
- 产出：两臂 `[META] calls / sub/call / dev / ar`、`[GRAPH] calls / avg_nodes`、`ar_us_avg`；差分得到 **draft 每轮的 meta 调用数、子图派发数、节点启动数**。
- **判据**：draft 每轮启动数 >> 100 => 13.8 ms 主要为主机提交（走 **B1**）；≈ 数十次 => GPU 真在算（走 **B2**）。

### M2 派发数普查（同一机器时段，纯读日志）
- 算 target 与 draft 各自的「每轮派发数 / 每轮节点数 / 每轮 dev ms」，写成表入 `PLAN-GRAPH.md`。

### M3 注入图定位（只读代码）
- 定位接受 token 注入的确切调用点（`common/speculative.cpp` accept/draft -> `llama_decode`，与 `src/llama-context.cpp` batch 构造），确认 `ne1` 变化来源。

## 3. 阶段 1：三条改动线（按实测毫秒排序）

### 线 A —— 注入图定长化（把 direct 变 replay）
- **改动**：注入批固定为 `n_max+1`（8）个 token，多余行用 attention mask 屏蔽，使注入图与 verify 图形状恒定。
- **落点**：M3 定位到的 batch 构造点。
- **门控**：新 env `LLAMA_SPEC_FIXED_BATCH`，默认关。
- **验收**：`[GRAPH] direct` 自 18.2% 明显下降；`[META] dev` 不升；同配置 sha256 `f3edac19...`；报 ms/轮 与 AL。

### 线 B —— draft 13.8 -> 2.6 ms（DFlash2 独有，最大单笔 ~11 ms）
- **B1（主机提交为主）**：合并 draft 图小算子（优先 `RMS_NORM`+`SCALE`，量级约 480 次启动；其次 GDN conv/state 小算子）。
- **B2（GPU 为主）**：查 draft 在 M=8/block 下的 GEMM 形状与 MMVQ/MMQ 选择（draft 为 Q4_K_M，用既有 C5=4 交叉点结论），核对 block draft 的 ubatch 是否被放大。
- **验收**：draft 段 ms/轮 或 `[META] dev` 下降；sha256 不变；>= 2 臂。

### 线 C —— 主机派发/分配开销（`alloc 5437 us/call` 是最大单笔已测主机项）
- **改动**：把 N6a 的「内容未变则跳过」从 meta rebuild 扩展到分配路径：`ggml_gallocr_alloc_graph` 在图内容 + 容器状态均未变时 early-out，
  并让 meta 的 `init_tensor` 幂等（与跳过联动）。
- **约束**：只在属性确实未变时生效；env 门控、默认关；**先做 2 臂预判再投入**（N6a 教训：机制生效 != 省时间）。
- **验收**：`[SCHED] alloc` 与 `[META] total` 同步下降；正确性门通过。

### 线 D（小、备选）—— selector 上 GPU（4.3 -> ~0.5 ms）
- 前置：RNG 必须 counter-based；语义需自写（R159：ninfer 的 op 不是 drop-in）。优先级低于 A/B/C。

## 4. 阶段 2：验证与入账（每次改动同一套）
1. 构建三查（`BUILD_RC=0` / errors=0 且有 `Built target` / 二进制标记串非 0）。
2. 同源 A/B，>= 2 臂，报 `MEDIAN_TG` + `timings.py` 的 ms/轮 与 AL。
3. 正确性门：同配置 greedy sha256 逐位一致。
4. 结果写入 `PLAN-GRAPH.md`（新节点/边 + 维护日志）与 `AGENTS.md §1`；`HANDOFF.md` 顶部同步。
5. 每完成一件回图复读接线。

## 5. 明确不做（已证伪或已关闭）
加卡（TP4/TP6 实测更慢）、KV dtype / 张量核 KV（8K 封顶 ~2 ms）、N8T/N8-vec、layer split（慢 40%）、多形状 decode graph 缓存、
CPU 采样迁移、`--spec-draft-device`、照抄 jusko D256 FA 常量、**切换到 MTP 交付**（仅作量尺）。

## 6. 假设与风险
- **假设 1**：M1 会显示 draft 每轮只发一次前向；若多次，B 改为「减少前向次数」。
- **假设 2**：注入图定长化不改变数值（mask 行不参与 attention/采样）；若 sha256 变化，以新配置 sha256 作新门并如实标注。
- **风险 1**：线 C 触碰分配器（结构性）=> env 门控、默认关、先预判。
- **风险 2**：机器独占 + `/tmp/LLAMA_BUILD_LOCK`（构建与测量互斥）；探针集在同一比较内不得改变。

## 7. 交付物
- 本文件（P0）；`llama.cpp` 内线 A/B/C 的独立门控 commit（`llama : <描述>` + `Assisted-by: DSH (DeepSeek Harness)`，只 push 私有 fork `v100`）；
- `PLAN-GRAPH.md` / `AGENTS.md` / `HANDOFF.md` 同步；一份数字表：每轮 `dispatches / node launches / dev / ar / ms/轮 / AL`（改动前后并列）。

**执行顺序**：P0 落盘 -> M1/M2/M3（一次机器时段）-> 按 M1 判据进 B -> 并行准备 A -> A/B 落地后评估 C -> D 备选。
