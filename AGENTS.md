# AGENTS.md — llama.cpp V100 / SM70 专项优化（DSH 迁移版）

> 本工作区原先由 **Qwen Code** 驱动（规则在 `QWEN.md` + `.qwen/`），2026-09-20 迁移到 **DSH**。
> DSH 每次会话**自动加载本文件**，所以这里放"必须先知道"的内容；细节一律指向下列权威文档。
>
> **阅读顺序（接手/新会话）**：
> 1. **本文件**（自动加载）
> 1.5 **`1cat-vllm-v100-study/AUDIT-2026-09-20-dsh.md`** —— **可信度索引**：哪些旧结论可信、哪些是弯路（16 条更正 E1–E16 + 修正后事实 F1–F7）。**冲突时以它为准（红线除外）**。
> 2. `1cat-vllm-v100-study/HANDOFF.md` —— **当下实况**（进度、已落地提速、已排除清单、服务状态；其中被审计更正处已就地标注）
> 3. `olddoc/qwen-private/QWEN.md`（原工作区根 `QWEN.md`，2026-09-23 迁入 olddoc）—— **持久规则全文**（含 §0.12 决定性发现、§0.13 外部先例、§0.15 Goal 实测、度量纪律、服务器档案；**含口令，禁止外传**）
> 4. `llama.cpp/AGENTS.md` —— **上游代码约定**，动 `llama.cpp/` 代码前必读
>
> 冲突优先级：**HANDOFF（实况）> QWEN.md（持久规则）> 本文件**；但 `## 2 红线` 与 QWEN.md §1 **不可被任何文档覆盖**。

## 0. ★★★ 最高优先级工作方式：维护 `PLAN-GRAPH.md`（用户 2026-09-21 明确要求）

**`1cat-vllm-v100-study/PLAN-GRAPH.md` 的 Mermaid 依赖图是本项目的最高优先级产物，必须持续维护。**

规矩（逐条照做，不得省略）：
1. **图是入口**：任何接手（新会话/子代理）**先读它**；我向用户汇报时**用节点 id 指代**（N0/N1/X5/K4...），不重复解释。
2. **试过的每一样东西都要画上去**，包括**走错的坑**。证伪的节点**只改灰、不删除** —— 因为「为什么被否」本身就是信息。
3. **每完成一件事，必须回来复读这张图**，检查：
   - 新结论是否需要**新节点**？（我现在就漏了 A2、低比特权重等节点）
   - 新结论是否依赖别的节点？若依赖，**必须画成粉框『条件结论』并连到依赖方**。
   - **某个环节是否还需要接线？** —— 这次复读就发现：A2 没在图上、低比特权重路线没有节点、X11 少一条指向 N1 的边、D4 与 N3 是同一张配置表却未相连、N8 少了指向目标的边。
4. **外部想法随时接进来**：在 `PLAN-GRAPH.md` §3 登记来源，并连到对应节点。
5. **接线是双向义务**：不只加节点，还要检查已有节点之间的边是否该补/该改。

> 这条规矩存在的理由：用户要的是「我分析问题、他看问题都更直观，外来想法随时能接进来」。
> 而**只放在我记忆里的规矩会随上下文消失** —— 所以它必须写在本文件（每次会话自动加载）与 `PLAN-GRAPH.md` 两处。
> ⚠️ `FINAL-REPORT.md` 是较早的交付报告，其中"差距主因是 draft 质量"的结论**已被 HANDOFF §6 修正**（真因是每轮延迟）。

## 1. 北极星与当前位置

**目标（不随迭代改变）**：让 `llama.cpp` 在 V100 上**追平 1cat-vLLM 的速度与效果**。交付物只落在 `llama.cpp/` 内；
**★ 硬指标（用户 2026-09-21 重申）：tg 至少 150 T/s 以上** ⇒ 在 AL 5.55 下 **ms/轮 ≤ 37.0**（现 55.5，需再 −33%；基线 99.97 t/s 还要 +50%）。原验收 180 t/s / ~20 ms/轮 为目标上界，对标 1cat 17.463 ms/轮、221–263 t/s。**报数必须 tg 与 ms/轮 并列**（tg = AL ÷ ms/轮）。第二大缺口是**预填充**：32K prefill 2162.64 t/s vs 1cat 公开 3567–4069 —— ⚠️ **那是 4 卡、我们 3 卡，按卡归一后约 1.40×，不是 1.65–1.88×**（R215）。
**★★ R143-R158 的四个决定性数字（先看这里，再去看旧账）**
1. **有效带宽 = 438 GB/s，不是 800**（Z1 实测：无投机 8K = 45.31 t/s = 22.07 ms/token，每卡权重 9.68 GB ⇒ 9.68/0.02207）。
   ⇒ **一切基于带宽的旧估算都要 ÷1.82**；旧结论「权重流 12.1 ms 已到 roofline」**撤销**（按 438 算是 **22.1 ms**，正好等于实测 token 时间）。
   ⇒ 这只是 V100 HBM2 实用值的 **55%**（权重就在显存里，所以这是**达成率**问题，不是 IO 问题）。
2. **权重流占 8K 轮时的 40%**（22.1 / 55.5），不是 22% ⇒ **LOWBIT 从「非主线」升为与 N6 同级**（砍一半字节 = 省 10.4~13.8 ms = tg +23~33%）；
   并新增主线 **BW1**：q8_0 有效带宽 438 -> 700 就值 +16%（抓手 N12 / v100-skinny QPN）。
3. **两阶段策略（`PLAN-to-180ts.md` §2.7）**：轮时 = 主机 20.6 + GPU 串行 34.5（可加性，N0 已证）。
   **阶段一 = 主机侧三件套（N6a+N6b / P-B / N7）⇒ 落到 ~34.5 ms 的 GPU 地板 = tg ~161（已达 150）**；
   **阶段二 = 再动地板（地板 22.1 ms 是权重流）⇒ 只有 LOWBIT 或 BW1 能到 180**。⚠️ 没有 N6b，阶段一只有 tg 128。
4. WARNING - **Z1 的 ctx 扫描（8K/32K/128K）已作废：测量设计错误（R160 自查）**。
   harness 的三个 prompt 是**一句话**（约 100-200 字符），`--ctx-size` 只**分配** KV 而**不填充**它，
   ⇒ 三个臂实际都在 n_kv 只有几百的同一条件下 decode ⇒ **零斜率是假象**。
   ⇒ 之前由此推出的「KV 被隐藏」「N1/N8T 是 256K+ 项」**作废，不得引用**。
   ⇒ 正确的深度测试：`llama-bench -d <depth>`，或先用长 prompt（`/tmp/prompt256k.txt`）填满 KV 再测 decode。
   **⚠️ R168 重要更正：上面那三点是 `-r 1`，被「第 1 次重复」的瞬态偏低 20-27% ⇒ 绝对值与斜率都不可直接引用**（原始数字与 `S/F 反解` 见 §4.36）。
   - ⇒ **H5（浅上下文离散台阶）已撤回**；**R170 密集阶梯已定论**（一次加载 + 逗号列表 + 升序 + `-r 8`，11 个点 4 分钟）：
     256->8192 只从 21.35 涨到 22.18 ms/token（**+0.83 ms，3.8%**），稳态浅段斜率 **0.10-0.14 us/ctx-token**，
     而**全段漂移 3.8% 小于臂内离散 4.6-6.5%** ⇒ 曲线基本是平的。
     ⇒ **8K 上整条注意力只值约 1.1 ms/token（5%）**；反推权重流 9.68 GB / 21.35 ms = **453 GB/s**（与 438 一致）。
     ⇒ **KV 线（N1 / N8T / N8-vec / N13）在 8K 上封顶约 2 ms/轮**（含 f16 转换 0.8），**不是通往 150 的路**。
     ⇒ **R175 把深段也量出来了**（同一次加载、升序、`-r 8`，到 131072）：**256->131072 跨三个数量级是一条直线，斜率恒为 0.1028 us/ctx-token**
       （256->8192 = 0.1054 / 8192->131072 = 0.1026 / 32768->131072 = 0.1054；四个深段配对**首次可分辨**，0.090-0.109）。
       **8K 时深度项 0.84 ms/token；128K 时 13.4 ms/token（占该步的 39%）** => KV 是**用户 256K 场景的项**，不是 8K 的项。
     ⇒ **`[FAK]` 给出机制**：内核在 `n_q=1` 处切换 —— `n_q=1` 走 `VEC`（`need_f16_K/V=0`），**所有 `n_q>1` 走 `MMA_F16`/`TILE` 且 `need_f16_K/V=1`**。
       => **f16 KV 转换由「`n_q>1`」触发，与 `n_kv` 无关，且不在纯 decode 路径上** —— 这正是 N1 只值那 ~0.8 ms/轮的原因。
     ⇒ 账被逼到只剩两处：**主机侧（N6a/N6b/P-B/N7）与权重流（LOWBIT/BW1）** —— 正是两阶段策略的全部。
   - ✅ **N8T / N8-vec 改灰这条仍然成立**：它只需要「斜率 << P=6 对应值」——P=6 要 2-3 TB/s（任何测量都不可能出现），
     瞬态偏 20-27% 不足以翻转它。**只是引用的斜率数字要换成 r>=8 的。**
   - ✅ 测量纪律细节（`-r >= 8`、跨工具 4% 之谜）见 §4.36/§4.38。
5. **前三条仍然有效**（438 GB/s、权重流占 40%、两阶段）—— 它们只用到 8K 那一个点。

**本次会话新增的判断**：见 `PLAN-GRAPH.md` §3（外部项目/上游 PR 侦察）与 `BASELINE-LEDGER.md` R218-R254（最新：R253 = sched 慢路径根因）。
`vllm/`、`1cat-vllm/`、`vllm-forkpoint/` 只是**只读参考**（抄作业对象）。

| 项 | 数字 | 来源 |
|---|---|---|
| 对标（1cat 生产 `vllm-1cat.service`，4×V100，FP8+DFlash2） | **221.6 / 230.8 / 263.2 tok/s**，AL 4.06-5.21，**17.463 ms/轮** | `premise-check-1cat-vs-llamacpp.md` |
| 我们（Q8_0 + DFlash2 n=7，TP3 tensor + NCCL，正式口径带 drop_caches） | **95.20 tok/s**（NODROP 最好 96.60） | `HANDOFF.md` §4 |
| 起点基线（原版 b11053，TP tensor） | 55.95 | `/root/goal-baseline.txt` |

⇒ **已 +70%**；与 1cat 差 **~2.3×**，差距在**每轮延迟**（我们的 AL 4.11-6.08 **不输**对方），不在草稿质量。

**三项已落地提速（均验证过正确性）**
1. 并行化 CPU selector（`common/speculative.cpp::build_dflash2_selector_cpu()`，8 线程）：23.45 → 4.3 ms/轮，**+26.8%**
2. `GGML_CUDA_P2P=1`（`ggml-cuda.cu:391` 默认未设）：~~+10.6%~~ ⚠️ **R276 复测：当前边际 0%**（收益已被 NCCL+FGC+T8 吃尽；保留开启无害）
3. **把 NCCL 编进 `libggml-cuda.so`**（本机 1cat venv 里已有 NCCL 2.29.7，无需下载）：**+18.4%**（单次最大）

**已排除，勿重做（详见 HANDOFF §5/§6）**
- **HMMA / Volta FP16 Tensor Core 路线** —— ⚠️ **2026-09-21 更正口径（见 `GAP-ANALYSIS-2026-09-21.md` §7.3）**：
  - 旧「天花板 ~1.36×」是 **Q8_0 格式内**的余量（合成形状 m=4096 k=14336，AUDIT E5 已标注），**不是「换格式」的收益**；而 **E14 已用同源实测否决 Q4_K_M**（17.11 GB 的 ms/轮反而 58.6 vs 29.05 GB 的 55.8、AL 更低）⇒ **低比特权重这条方向关闭**，别再引用旧 22% 口径。
  - **该结论不适用于「注意力」**：两个独立 V100 fork 都证明**量化 KV 的注意力在 V100 上恰恰应该走张量核**
    （`fattn-q8-volta.cuh`：smem 内反量化 q8_0 -> f16 -> `mma.sync.m8n8k4`；我们 `fattn.cu:704` 的注释 *On Volta tensor cores are only faster for sufficiently large matrices* 正是被反驳的那句）。
  - **「自写 WMMA 原型仅 90 GB/s」这条推断的样本可能是坏的**：sm_70 上 `nvcuda::wmma` **既没有 ldmatrix（sm_75 才有）也没有 swizzle**
    （`flash-attention-v100 utils/docs/volta.md:139,143`）；v100-skinny 记录其 v1/B_ring 两个版本**都因 K 切分而死**，赢法是 **QP 切 N、A-stationary**、56 寄存器、主循环无 smem、全程 1 barrier。
  - ⚠️⚠️ **R207/R213 反转此条，而且比原先想的更彻底：预填充早就在 FP16 张量核（cuBLAS）上** —— `ne11>=64` 时 `mmf`(对量化直接 false)/`mmvq`(<=8)/`mmq`(<64) 全不过 ⇒ 走 `mul_mat_cublas`，而它 `ggml-cuda.cu:1585-1608` **每次调用都全量反量化整张权重成 F16** ⇒ 旧「dp4a 60.8 TFLOPS ≈ 峰值 97%」是**巧合误读**（n=512 根本不走 dp4a）；**R216 同形状实测**反量化 = 算子级 **−19%**（209 µs/call；有效带宽 ~860 GB/s 已在墙）但端到端**仅 3.8%**（`-ub 8192` 可拿回 ~2.9%）⇒ **不足以解释 1.4× 缺口**；⚠️「裸 cuBLAS 103 vs 60.8」是**跨形状对比，已作废**。子代理显式实现的 ABBA 四臂 = **−0.30% / −0.19%（零效果）** ⇒ **节点 DEQ 的 A4（自写融合内核/F16 副本缓存）作废**
    ⇒ 详见 `PLAN-GRAPH.md` 节点 **DEQ** / **DISPATCH** / **HMMA** 与 §3.2.3/**NF10**（参考实现 `v100-refs/ninfer-v100/src/ops/linear/w8/`，W8G32 ≡ Q8_0）。**判断「在跑哪条路」必须读分派链或做区分性实验，不得靠「数值接近某峰值」。**
    **「不要在 V100 上用张量核做注意力」不成立**（理由见上）。
- NCCL 调参（默认最好）。⚠️ **加卡一条已作废**：
  用户 2026-09-20 明确「**卡数 1–6 自由，跨岛不是瓶颈，直接排除可能性**」⇒ 早期"TP3 最优 / TP4+ 跨 NUMA 变慢"只是 butterfly+NCCL 早期、
  固定短上下文标尺下的历史实测，**需重扫**（Phase B4）。反证：1cat 生产服务本身是跨岛 TP4（0,1,3,4）且 221.6–263.2 tok/s。
  另：**加卡不买带宽**的结论仍成立于当时条件（瓶颈是 allreduce 实现，非拓扑）。
- Q8_0 在 ne11=8 强推 MMQ（**−31%**，必须走 MMVQ）、`--spec-draft-device`（draft 必须与 target 同设备）、
  多形状 decode graph 缓存（主机侧合计仅 2.10 ms/轮 = **上限 2%**）、CPU→GPU 采样迁移（上限 ~7%）、
  `VLLM_SM70_USE_BREAKABLE_CUDAGRAPH`（1cat 实测 −13% ~ −29%，**明确禁止**）。
- ncu / nsys 在本机**无法 profile** 18-29 GB 模型（device memory save/restore 崩）⇒ 改用 `test-backend-ops`。

**★ 每轮账本：一律以 `BASELINE-LEDGER.md` 的采用链为准（现在 = **B5 = B4 + FGC 完全关闭**（不设 `GGML_META_FULLGRAPH`，R276 边际反转），均值 **45.94 ms/轮**，tg 113.28/94.32/145.65，账本 R273/R274/R276；更早口径都已过期）。**
⇒ ⚠️⚠️ **R174 再次改写（四探针实测，取代上面这张表的旧口径）**：
   - **`[META]`**：`calls=576 | total=21.283 loop=19.743 dev=17.435 ar=2.304 ms/call | prologue=1.540 (7.2%)`；
     **576 x 21.28 ms = 12.25 s ≈ 该臂全部生成时间（12.4 s）** ⇒ **meta 主机循环几乎就是整轮本身**（每轮约 **3.6 次**调用），不是「53%」。
     每 call 里 **`dev` 17.4 ms（主机侧逐节点启动）** 才是大头，`ar` 2.3 + `prologue` 1.5 合计只占 18%。
   - **`[GRAPH]`**：`calls=82688 capture=2742 **replay=64875 (78.5%)** direct=15071 (18.2%) per_call=5.9us avg_nodes=40`
     ⇒ **不是「13.97% direct 吃掉 59%」**；但 direct ≈ 15071 x 40 节点 x ~9 us ≈ **5.4 s（43% 墙钟）**，且 `capture=2742 ≈ 1.4 次/轮`（图 key 约每轮换一次）。
   - **`[DIRECT_PROBE]`**：top5 = `attn_norm-0` / `cache_r_l0 (reshaped)(view)` ⇒ 抖动确实在 **GDN 递归状态视图**；但 85% 的字段计数是「首个差异之后的连带」，**不能当独立证据**。
   - **`[OP]`（每算子 GPU 时间）探针不可信**：出现负总量（事件对按 (device,节点下标) 复用且跨调用混对）=> **作废，不得引用**；要重做必须改成「下一次调用再查询、按设备配对」。
   - MTP 对照**尚未同口径**：`mtp4 = 83.55 t/s` 那一臂没有探针，而 DFlash2 那一臂带了四个探针 ⇒ 不可比（chain 3 的 C0/MTP 才是对照）。
   - ★★ **R180 / E4 定论（纯读码，见 `1cat-vllm-v100-study/E4-GRAPH-REUSE-VERDICT.md`）：draft 图 0% 复用是【结构性】的，不是 bug。**
     三道门**任何一道单独**就判 0%：① 槽位 —— `get_gf_res_prev()` = `gf_res_prev[n_outputs > 0]`（`llama-context.cpp:2439`）而 `gf_res_prev_active` 只有**一个**裸指针（`llama-context.h:374`）；
     ② token/embd 互斥 —— `allow_reuse`（`llama-graph.h:827-831`）明确要求「都带 token」或「都不带 embd」，而 draft 注入前向**只带 embd**（`speculative.cpp:1398-1431`）、块前向**只带 token**（`:1482-1486`）；
     ③ n_tokens 不等 —— 注入 `tok=8.86-8.93`（每轮变）vs 块恒 `8.00`（E3 实测）。注入批 `logits` 全 `false`（`:1428`）⇒ 槽 0；块批 `logits=true`（`:1485`）⇒ 槽 1。
     **同一机制也解释了 target 的 92%**（每轮只一次、形状恒定）⇒ 0% 与 92% 同源，结论自洽。
     ⇒ **否证**：想「按槽位分别记住 active」也救不了 —— `ggml_backend_sched` 只有**单份** splits/缓冲，miss 路径的 `ggml_backend_sched_reset`（`:1405`）已把另一张图的分配释放成悬垂 ⇒ 上游「只认紧邻上一次」是**正确设计**。
     ⇒ **撤销**「让 draft 图可复用 = 潜在 ~15 ms/轮」这个头号项：**该奖赏不存在**。那 7.4 ms（`alloc 1.9 + enqueue 5.5`）是「每轮必须重建两张不同图」的代价。
     ⇒ 唯一可动形态 = **每轮 2 次 decode 合成 1 次**（fused，见 `speculative.cpp:1396` 注释）⇒ 省 ~**3.7 ms/轮**（53.6 -> ~50，tg 97 -> ~104），**不是 7.4**。
     ⇒ 主战场仍是 `[META] dev=17.4 ms/call` 的主机逐节点派发。**唯一还没上机的直接观测**：`LLAMA_GRAPH_SLOT_DEBUG=1` 看 draft ctx 是否真在 `slot=0/1` 交替且 `hit=0`（探针已在 `llama-context.cpp:1372-1385`）。
     ★★ **已上机闭合（R181 臂 `e4slot`，40 s，`MEDIAN_TG=97.02`，sha256 门一致）**：draft ctx 24 次调用 **`hit=0` 全中**，
     `res` 在 `0x3f83e220`(槽1,`n_outputs=8`) 与 `0x48852fe0`(槽0,`n_outputs=0`) 之间**完美乒乓**，`prev_active` 永远指向另一只；
     target ctx 从 call=6 起 **一路 `hit=1`**（`res`/`n_outputs` 恒定）。⇒ 判决坐实，第①道门单独就够了。
     同一臂 `[RT]`：draft 556 轮 **`reuse=0`**，`build 127 + alloc 1892 + enqueue 5622 us/轮`；
     target 294 轮 `reuse=270`，**`enqueue 20665 us/轮`** ⇒ **两块合计 28.3 ms/轮（53.6 ms 轮时的 53%）**，这就是 `[META] dev` 的落点。
     详见 `E4-GRAPH-REUSE-VERDICT.md`（实证附录）与 `E5-META-DEV-BREAKDOWN.md`（dev 算术分解 + 下一个实验）。
     ★★ **E6（R185-R188，见 E6-MULTISLOT-VERDICT.md）**：[FPSTAT] 实测 832 calls **same_as_prev=0 distinct=24 max_count=266** ⇒ N6a 单槽缓存结构上永不命中；
     对照地板 6 次同配置 94.63-96.46 t/s（离散 ~1.9%），sha256 门 f3edac19... 从未破。
     稳态 [META] **total 10.66 / dev 7.19 ms/call**（旧账 21.28/17.44 作废）；~~[GRAPH] 稳态 direct 12.6% ⇒ 19 ms/轮是唯一够本的款子~~ **← 已被 E7 实测推翻，见下**。
     **五次改容器索引的尝试全部 abort**（:2144 节点断言 x3 -> ggml.c:1805 obj_new arena 耗尽 -> :2105 节点断言）⇒
     **容器与 buffer_simple_tensor/init_tensor_impl/tensor_images 共享 per-container 状态与 arena 生命周期，不改本体无解**（结构性改动，须先问用户）。
     已改走**正交战线**：形状敏感的 CUDA 图 key（= 上游 #28666 对 #28652 的修法，已关未并），env GGML_CUDA_GRAPH_KEY_SHAPE，只动 ggml-cuda.cu 一处。
     上游已编号入账：#28652(OPEN)/#28666(关未并)/#25406(关未并)/#24549(OPEN)/#27310(已并，仅部分融合)/#28549(已并，batch categories 仍是 TODO)。
     ★★★ **E7（R189，见 E7-CUDAKEY-VERDICT.md）：撤销 E5 的 19 ms/轮**。同源四臂 A/B（形状敏感 CUDA 图 key = 上游 #28666 的做法，
     env GGML_CUDA_GRAPH_KEY_SHAPE，只动 ggml-cuda.cu；四臂 sha256 f3edac19... 全一致）：
     **机制成立** —— direct 17345 -> 13806/13811（-20.4%）、replay 103977 -> 107715（+3.6%），与诊断逐项吻合；
     **性能为零** —— C0 均值 97.40 vs 特性均值 97.31（-0.09%，在噪声内；C0 两臂自身差 0.47）；ms/轮 56.8 vs 57.0。
     ⇒ **删掉约 1 s 主机工作（占臂时 6.8%）一点没省** ⇒ direct 不在关键路径上，**计数器增减不能当墙钟代理**；
     E5 的 dev 分解（replay 30us / direct 281us / capture 300us）是反推拟合、**过拟合，作废**。
     ⇒ **新纪律：不得再从"省主机 X ms/轮"推收益**；任何主机侧改动必须**同臂量边际**后才投入。该改动**不采用**（有 +0.7us/compute 与图对象代价）。
     ⇒ **同时停掉**以该前提为依据的结构性改动（meta 容器 per-fingerprint 缓存）；WIP 存档 wip-cuda-key-shape.patch。
     ⇒ 待判问题（已派实验）：**轮时的关键路径是主机还是 GPU** —— 用 env GGML_META_HOST_SPIN_US 在每次 meta 调用末尾注入已知忙等
     （3 ms/call x ~3.3 call/轮 ⇒ 若主机在关键路径应涨约 9.9 ms/轮），spin=0 vs spin=3000 各两臂。**这是之后所有优化项的排序依据。**
     ★★★ **E8（R191，见 E8-HOST-CRITICAL-PATH.md）：主机时间确实在关键路径上 —— 拿到可标定的杠杆。**
     四臂同源（env GGML_META_HOST_SPIN_US 注入已知忙等 3 ms/call；四臂 sha256 f3edac19... 全一致；**对照两臂差 0.001%**）：
     p1 5262.502/5262.553 -> 5762.777/5747.978 ms、p2 6486/6488 -> 7143/7147、p3 2718.765/2716.602 -> 3005.097/2980.420；
     ⇒ ms/轮 **+5.4 / +5.5 / +5.7**，tg 97.10 -> 88.7/88.9、78.78 -> 71.5、112.18 -> 101.5/102.3。
     注入 9.9 ms/轮只体现 5.4-5.7 ⇒ **穿透率约 55%**。
     ⇒ **标定杠杆：每 1 ms/call 的主机时间 ≈ 1.8 ms/轮**；当前 total 10.66 ms/call x 3.3 = 35.2 ms/轮，其中约 **19.2 ms/轮 在关键路径上**。
     ⇒ **要到 37.0 ms/轮 需砍约 11 ms/call，而总共有 10.66 ⇒ 必须几乎砍光主机时间。**
     ⇒ 与 E7 **不矛盾**：direct/capture 只是主机时间的一小块，所以"删 direct 无收益"与"主机时间值 1.8x"可以同时成立；**关键是那 10.66 ms/call 具体是什么**。
     ⇒ 三块已知：**dev 7.19**（149 compute/call x 48us，成分不明 ⇒ 13 ms/轮）/ **ar 2.22**（46.5 次 AR x 47.7us = NCCL 启动 ⇒ 4 ms/轮，对应 E1 图内 AR）/ **prologue 1.25**（重建 ⇒ 2.25 ms/轮；注意 same_as_prev=0 ⇒ 按内容哈希永远打不中）。
     ⇒ 下一步：**拆 dev**（按设备 + 每次 compute 的分布/分桶）。补丁存档 wip-host-spin.patch（默认 0 = 逐位相同，已上机验证）。
     ★★★ **E9（R192-R194，见 E9-DEV-BREAKDOWN.md）：dev 已拆开 —— 三卡对称 + 双峰。**
     两臂同源（dev1/dev2，8281/8282），MEDIAN_TG 97.81/97.09，sha256 门未破。末次 [META]：
     `dev=7.034 dev0=2.441 dev1=2.290 dev2=2.292 ar=2.211 ms/call`；`dev_hist_us n=118884 min=6 med=24 max=16367`；
     `<20=50823 20-60=51037 60-150=3061 >150=13963`；`nodes/sub=40.0`。
     ⇒ **dev 三卡完全对称**（不是掉队卡）；**86% 的 compute <60us（便宜）**；**>150us 占 11.7%（13963 次，均值约 224us）吃掉约 3.1 s（53%）**；
     ⇒ **11.7% 与 [GRAPH] 的 12.6% direct 精确对应** ⇒ 原估 direct 单价 281us 基本正确；**但 E7 证明长尾被 GPU 隐藏** ⇒ **靠 replay 治长尾不可靠**。
     ⇒ **折算（E8 杠杆）**：主体 <60us 2.4 ms/call => 约 4.4 ms/轮；ar 2.21 => 约 4.0；prologue 1.25 => 约 2.25；合计约 18.9（与 E8 的 19.2 吻合）。
     ★★ **机制已读通（R193/R194）**：切图规则 `2270: new_subgraph = (i+1==n_nodes) || axis==SPLIT_AXIS_PARTIAL` ⇒ **子图数 = PARTIAL 节点数 + 1 = 46.5+1**，
     且**每个边界恰好一次 AR**（`2553: i < n_subgraphs-1`）；**上游已有"延迟 AllReduce"机制 `get_i_delayed`（仅 MoE 启用）**；
     comm 接口 `ggml-backend.h:210 (comm_ctx, ggml_tensor ** tensors)` 的数组语义是**一个逻辑张量 x n_backends 个设备切片**（ggml-cuda.cu:1000-1060，
     `ncclGroupStart/End` 已在用）⇒ **单次 AR 的 47.6us 压不动，只能减次数**。
      ~~⇒ **优先级 1 与 2 是同一件事**：推广 `get_i_delayed` 的延迟 ⇒ **子图数与 AR 次数同时下降**（合计向约 8 ms/轮）。~~ **← 已被 E10/E11 否证。**
      ★★★ **E10（R195，子代理实测 + 源码已还原）：延迟 AllReduce 在本负载上结构上不可能 ⇒ 优先级 1 撤销。**
      `[DLY]` 探针实测：**target 调用 n=4950 节点 / 129 个 PARTIAL 边界，269/269 `del=0`**；
      `linear_attn_out-N` 首消费者是 RESHAPE（gap=8）、`attn_output-N` 是 ADD（gap=7）、`ffn_out-N` 是 ADD（gap=78/44）；draft 调用 n=648 / 11 边界 / 同样 del=0。
      ⇒ **被归约的值永远被紧邻的下一个节点消费** ⇒ 连上游 MoE 折叠 `get_i_delayed`（`:2104-2258`）也从不触发（`sub/call = ar/call + 1` 恒成立）。
      依赖条件：唯一 use + 该 use 是求和 ADD + 中间节点 MIRRORED 安全；且 `AR(a)+AR(b)==AR(a+b)` 只对那个 sum 成立 ⇒ 折不到别的 PARTIAL 上。服务器树 md5 已复原为 `9dbb5135e8eaf057f9e58c2fcd44a9b1`。
      ★★★ **E11（R195，复核 `/tmp/g4-ab.log` `/tmp/aron.log`）：设备端/内部 AllReduce 也死。**
      g4-ab ARM-A（NCCL）tg=98.12 `ar_us_avg=53.0` sha256 `f3edac19...`（=门值）；ARM-B（device AR）tg=94.45 `ar_us_avg=61.5` **sha256 `69207026...` 门破**；
      `aron`（device-side push AR）tg=89.80 **同样门破**，其对照臂 `aroff2` 还 `LAUNCH_FAILED`。⇒ **既更慢又改变数值，不采用。**
      ★★★ **E12（R195，读码 + `[DLY]`/`[RT]`）：账本落到具体结构上。**
      **target 一次 meta 调用 = 130 子图 x 3 卡 = 390 次图启动 + 129 次 NCCL AR**；draft 一次 = 12 子图 x 3 = 36 次 + 11 次 AR；
      每轮 = 1 次 target + 约 1.9 次 draft 调用 ⇒ **target `enqueue_us` 18.8-20.7 ms/轮**（`6074133/294`）、draft 合计约 7.5 ms/轮（含 `alloc` 1.8 + `build` 0.12）。
      ★★★ **E13（R196，`/tmp/ut.log` + `util-*.csv`，nvidia-smi 100ms 采样 + 活动窗口对齐）：主机是限流项，天花板已量出。**
      **无投机臂**（`SPEC=--spec-type none`，tg 44.92 = 22.26 ms/token）：利用率时序**几乎全是 >=55% ⇒ GPU 约 95% 忙碌**
      ⇒ **22.07 ms/token 与 438 GB/s 是真实 GPU 工作**，不是主机假象（两阶段诊断仍成立）。
      **投机臂**（q8c/ut1）：时序里大量 30-55% 段与空隙 ⇒ **轮内 GPU 约 35-45% 空闲** ⇒ 55.3 ms/轮 里约 **20-25 ms 是 GPU 空转等主机**（与 E8 的 19.2 定量吻合）。
      ⇒ **天花板 = GPU 忙碌约 30-33 ms/轮 ⇒ tg 上界约 168-185，前提是主机不再挡路。**
      ★★★ **E14（R196，同库同口径 A/B）：Q4_K_M 不是提速杠杆 —— 撤销「LOWBIT 省 10.4-13.8 ms/轮」。**
      q4a（17.11 GB）ms/轮 **60.1/56.7/59.0（均 58.6）** vs q8c（29.05 GB）**57.0/53.6/56.7（均 55.8）**；AL **4.55/3.81/5.62 vs 5.55/4.22/6.38**
      ⇒ tg **75.72/67.14/95.29 vs 97.32/78.78/112.44**。GPU 忙碌率**反而更高**（util-q4a 几乎全是 `#`，q8c 有大量 `=` 与空隙）
      ⇒ **k-quant 在 Volta 上每字节的 GPU 时间更多** ⇒ **权重字节数不是投机轮的约束**。两侧 greedy sha256 都是 `f3edac19...`（同一门值）。
      ★★★ **E15（R196-R197，FGC = meta 整调用单图捕获 = 1cat fullgraph 路线）：已上机，采用。**
      实现：CUDA 后端加 4 个 proc-address 入口（`ggml_backend_capture_{begin,end,launch,discard}`）+「父捕获进行中」计数（抑制后端自己的 per-subgraph 捕获）；
      meta 后端按签名（节点 data 指针 + 形状 + leaf 指针）缓存每设备一张 `cudaGraphExec_t`，第二次见到同一签名才录制，命中即 3 次 launch；env `GGML_META_FULLGRAPH` 门控。
      ⇒ **同源四臂 ABBA（唯一变量 env，四臂 sha256 `f3edac19...` 全一致）**：`[META] sub/call 47.6 -> 3.2`、`ar/call 46.6 -> 3.2`、`total 9.64 -> 3.87`、`dev 6.30 -> 1.40` ms/call；
      `[RT]` target `enqueue` **18.34 -> 8.68 ms/轮**、draft `enqueue+alloc` **5.42 -> 1.70**；**均值 ms/轮 54.35/54.14 -> 50.73/50.72（-6.5%）**；
      tg p1/p2/p3 99.83/80.59/115.78 -> **100.95/88.30/129.12**（p2 +9.6%、p3 +11.5%；**p1 几乎没动，待查**）。
      ⚠️ **穿透率只有约 25%**（主机少 14 ms/轮、轮时只少 3.6）⇒ 轮时已被「target 验证 GPU + 串行链（draft 6 + selector 5 + alloc/序言 5）」占住 ⇒ **下一阶段主战场**。
      ⚠️⚠️ **R276 反转 E15 的采用判定**：T8 之后 **FGC 关闭反而快 2.1%**（45.94 vs 46.94 ms/轮；关掉后 enqueue 8.3 -> 18.6 ms/轮 但墙钟更快 = 整图回放成了串行化税）⇒ **B5 起不设 `GGML_META_FULLGRAPH`**；B2 的 -6.5% 引用时必须注明已被 R276 反转。
      ★ **阶段成果账 = `1cat-vllm-v100-study/BASELINE-LEDGER.md`**（用户 2026-09-21 定的规矩：每个阶段成果记进去并**作为下一轮扩展的对比初值**；比基准差的一律不采用）。
⇒ ⚠️ `enqueue_us` **不是同步的 GPU 时间**：`llama-context.cpp:1436` 第二个参数是 `batched`，同步在 `:1376` 且条件 `pipeline_parallel`（harness 永远传 `--tensor-split` ⇒ 恒假）。

**真正待查（按头寸排序）**
1. **长上下文 attention / prefill**：本模型 `head_dim=256 / 24 Q 头 / 4 KV 头(GQA=6) / 16 层全注意力`；我们 256K prefill **370 t/s、TTFT 672 s**，1cat 公开 32K/64K prefill **3,567–4,069 t/s** ⇒ 用户真实场景（256K agent）里这是最大一块。
2. ~~draft 前向 13.8-14.4 ms（头寸 8-12×）~~ **已结清（R210）：那是 FGC 之前的口径**；E16 的 post-FGC 拆解里 **draft 阶段只剩 6.13 ms/轮** ⇒ 头寸从 ~11 ms 缩到 ~3.5 ms（节点 **PB**）。
3. **★ R248-R254 已定位（取代旧的"21.8 ms 提交窗口"）**：**sched 每次调用都走慢路径**（`ggml-backend.cpp:1614-1640` 无条件 `synchronize x3` + 重算 buffer 方案 ≈5 ms/call）；实测成因 `bic=556/allocfail=4` ⇒ **99.3% 是 `backend_ids_changed`**（每轮 4-6 种图交替）。解释了 E13「GPU 35-45% 空闲」与 E8「主机穿透 55%」。三种修法见账本 R253。
4. 量化 matmul 的 **M≈5-12** 段（头寸已封顶 1.36×，放最后）。

## 1.9 ★★ 环境边界与加载路径（用户 2026-09-21 19:1x/19:2x 两次明示；最高优先级）

### 边界表（照做，不解释）
| 类别 | 对象 | 允许的行为 |
|---|---|---|
| **正式环境（绝不碰）** | `/root/llm/systemd/llama-server.service`、`/root/llm/systemd/vllm-1cat.service`（及其软链 `/etc/systemd/system/...`） | **只读**；不 start/stop/enable/disable/改文件 |
| **正式目录（绝不写）** | `/root/llm/llama.cpp`、`/root/llm/ac922env` | **只读参考**；不编译、不落文件、不删 |
| **服务状态（不要动）** | `vllm-1cat` / `llmscope` / `new-api` | **保持用户交给我的状态**（2026-09-21 19:15 实测：`inactive / active / active`）；只允许 `systemctl is-active` 这类只读查询 |
| **可操作区** | `/root/llm/test/**`、`/mnt/3.84t/**` | 读写、编译、跑测、建脚本 |
| **模型加载** | **只从 `/mnt/3.84t/**`** | **永远不用 `/root/llm/models/`** |

### 已核实的路径（2026-09-21 19:15）
- 开发树：`/root/llm/test/v100-opt/llama.cpp`（在受权区内）
- 本地盘模型：`/mnt/3.84t/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf`（29,047,086,048 B）、同目录 `Q4_K_M`（17.1 GB，LOWBIT 现成样品）；
  draft `/mnt/3.84t/llm-models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`（1.14 GB，md5 `958589e393740557f3c1bb5d51720f83`）
- harness 默认已指向本地盘（`/root/p60-ab-harness.sh:20-21`，备份 `.bak-local`）

### ⚠️ 一处必须记住的更正（我自己先说错的）
**harness 不会按进程名杀 server**：`grep -n -e kill /root/p60-ab-harness.sh` 只有 `:105` 与 `:151` 的 `kill "$pid"`（杀它自己启动的那个）。
⇒ 残余风险只有三条：**端口冲突**（harness 默认 8120 vs 生产 8080）、**GPU 争用**、**`drop_caches` 会冷掉正在运行服务的页缓存**。
⇒ 因此安全措施是「**预检 + 显式端口 + 模型路径断言**」（见 `scripts/run-arm.sh`），**不是**去 kill 任何东西。

### 其余（用户明示）
- **开发/测试/调试都在本地盘**：构建目录、`CCACHE_DIR`、libdir、脚本一律迁到 `/mnt/3.84t/v100-opt/`（搬迁时必须无 llama-server/cmake 在跑，见 §4.37）。
- **不要再为磁盘/环境问题与用户纠缠**：需要什么直接做，做完报数。

## 2. 红线（不可覆盖）

- **绝不代用户** `git push` / `git commit` / `gh pr create` / `gh pr comment` / `gh issue create`，也**绝不代写** PR 描述 / commit message / reviewer 回复。
  本项目是**私有 fork，不做上游贡献**。（若用户明确要求代提交：`llama : <描述>` + `Assisted-by: <助手名>`，**不要** `Co-authored-by:`。）
- **不改** `vllm/`、`1cat-vllm/`、`vllm-forkpoint/` 三个目录的内容（只读对照）。
- 服务器上 **`/root/llm/systemd/` 整个目录只读**（用户 2026-09-20 明示：「当前目录是我的原始服务单元，请你不要修改，可以查看，这是在我们这个项目开始之前就在正常运行的内容」）。
  该目录 7 个文件均为 Sep 7–17（早于本项目 9/20 开工）；`/etc/systemd/system/{vllm-1cat,llmscope}.service` 是**软链到它** ⇒ 改那里等于改原始单元。
  **项目自建单元一律放 `/etc/systemd/system/`**（现有 4 个，全部 disabled）；服务只 `systemctl stop/start`，**不改 unit 文件**。
- `llama.cpp/` 源码：**只用 ASCII**（禁 `—` `→` `×` `…`，用 `-` `->` `x` `...`）；**不新增 `tests/*` 文件**；
  **大改动 / 新模式 / 新子系统：先停下问用户**；简单优先（完成 90% 的简单改动 > 完成 100% 的复杂改动）；每行代码都要能独立维护。
- `olddoc/qwen-private/QWEN.md`（原 `QWEN.md`）里含服务器密码 —— **不要把它提交 / 分享 / 复制进仓库**。旧文档回档见 `olddoc/`，本机参考模型见 `models/`，当前有效文档见 `docs/v100-dev/`（二次开发 3+1；**主攻 Q8_0，见 `05-量化范围.md`**）。

## 3. 环境

**开发测试服务器（所有编译 / 测试 / bench 都在这台，不是本地 Windows）**
- 连接：`ssh -o BatchMode=yes root@192.168.50.235`（key 已授权、非交互；**Windows 本机 ssh 直接可用**，无需 WSL）。
  密码见 `QWEN.md` §0.6（本地局域网个人机凭据）。主机名 `ac922-netboot`。

**连接自检配方（2026-09-20 在 DSH 实测通过）**
- host key 已在 `~/.ssh/known_hosts`（ed25519 `SHA256:zMkXtFuNe1XBAPsjbMZ/lbvkOQUxdEKfMVCYxt6eyEE`）；**`~/.ssh/config` 里没有这台机器的别名**，一律写 IP。
- ⚠️ **PowerShell -> ssh.exe 的引号陷阱（必守）**：远端命令里的**双引号会被 Windows argv 吃掉**，而 PowerShell **双引号**串里的 `$var`/`$(...)` 会在**本地**展开 ⇒ 轻则远端语法错（实测 `sleep: missing operand`），重则本地跑了别的程序。
  **规矩**：远端命令用 **PowerShell 单引号**串，**内部不要出现双引号与 `$(...)`**；逻辑一复杂就**本地写 `.sh` -> `scp` 上去 -> `ssh host 'bash /root/x.sh'`**（本项目既有约定，本文件命令都按此写）。
- 单次 ssh 往返 **~1.5-3.0 s**（连接建立慢；与 GSSAPI/压缩无关，Windows OpenSSH 无 ControlMaster 复用）⇒ **尽量把多条命令合并成一次调用**，不要在循环里反复 ssh。
- **长任务**：本地写脚本 -> scp -> 远端 `nohup ... > /tmp/x.log 2>&1 < /dev/null & echo LAUNCHED pid=$!`（已实测：立即返回，之后轮询日志/产物）。
- **应急通道**：本机有 `plink`（`C:\Program Files\PuTTY\plink.exe`，支持 `-pwfile`/`-hostkey`/`-batch`），但服务端走 keyboard-interactive 且**用 QWEN.md 记录的密码实测未通过**（可能已改）⇒ **目前唯一可用通道是 key**；key 若失效需人工交互登录。
- 连不上时排查顺序：`ping 192.168.50.235` -> `ssh -vvv` -> 确认 `~/.ssh/id_rsa` 与 `known_hosts` 仍在 -> 换 plink 试。
- **自检脚本**：`.dsh/tmp/ac922-check.sh`（scp 上去 `bash` 跑；一次打印 host / CPU / 内存 / 磁盘 / 6×GPU / 服务 / 项目路径 / 工具链）。
- **ppc64le（IBM POWER9，176 核）**，AlmaLinux 8.10，**6× Tesla V100-SXM2-16GB**，driver 550.54.15，**CUDA 12.4**；无 Rust、无 docker。
- **无盘 NFS**（`/` = 192.168.50.84）⇒ **下载 / 加载模型 / 编译共用一张网卡，不要并行**（用户明说"可以多等等"）。
- V100 HBM2 被映射进系统内存（PPC64LE 特性）⇒ 页缓存会吃显存，正式测量前 `sync; echo 3 > /proc/sys/vm/drop_caches`。
- 性能工具**不在 PATH**：`/usr/local/cuda-12.4/bin/{ncu,nsys}`。
- **6 张卡全归本项目**，可自主 `systemctl stop/start`、kill、抢占、重编，**无需逐次同意**；测试前确认 GPU 空闲（`nvidia-smi`）。
  腾卡/恢复：`systemctl stop vllm-1cat llmscope` ⇄ `systemctl start vllm-1cat llmscope`（`new-api` 不占显存，一直在跑）。
- **拓扑**：GPU 0/1/2 = NUMA0、3/4/5 = NUMA8，组内两两 **NV2**、跨组走 `SYS`。
  ⚠️ **卡数 1–6 自由、跨岛不是瓶颈**（用户 2026-09-20 明确：「这个你可以直接排除可能性」）⇒ **不要据此把方案限死在 3 卡**；
  选卡按实验目的与实测决定（早期「TP3 最优」是 butterfly 时代 + 固定短上下文标尺下的历史实测，需重扫）。
  参考：4 卡时我们 Q8_0 为 **7.25 GB/卡**，已优于 1cat 的 8.68 GB/卡。

**源码 / 构建 / 库**
- 本项目源码+构建：`/root/llm/test/v100-opt/llama.cpp`（`build/`、`build-nccl/`；`test-backend-ops` **只在 `build-nccl/bin/`**）。
- ⚠️ `/root/llm/llama.cpp/bin/*` 与 `/root/llm/test/llama.cpp/build/bin/*` 是**上游 `0.4.0-dev 434ddbb`，不是本项目 b11053** —— 别拿它们当本项目 binary。
- **库目录（A/B 用 `LD_LIBRARY_PATH` 切换）**：`/root/libdir-nccl`（**当前最好**）、`/root/libdir-rt`（butterfly 基线）、`/root/libdir-27858`、`/root/libdir-{c4,volta2,pristine,final,...}`。
- **度量脚本**：`/root/p60-ab-harness.sh`（主力；参数 `CARDS/SPLIT/TAG/NPRED/PORT/L/M/D/NODROP/P2P/ARENV/DEVD/NGLD`）、`/root/p60-ab-harness2.sh`（超集）。
- 模型：`/root/llm/models/`（`Qwen3.8-27B-GGUF/` Q2_K_XL 9.14 GiB 迭代用；`...TurboFCFusion-gguf/` Q4_K_M 17.23 GiB 生产用；`Qwen3.8-27B-DFlash2-GGUF/` draft 1.14 GB）。
- 编译（改 `.cu` 后；系统 gcc 8.5 会失败，**必须 gcc-toolset-12**）：
  ```sh
  CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ \
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CUDA=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc -DCMAKE_INSTALL_RPATH=/root/llm/llama.cpp/lib64
  cmake --build build --config Release -j128
  # 编 NCCL 版必须额外加（包里只有 libnccl.so.2，find_library 找不到）：
  #   -DNCCL_INCLUDE_DIR=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/include
  #   -DNCCL_LIBRARY=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/lib/libnccl.so.2
  ```
- **本地**：Windows 写码；WSL `Ubuntu-20.04` 存在但需手动 `wsl` 启动，且**没装 cmake**；**`.cu/.cuh` 在 WSL 编不了（无 CUDA）⇒ 唯一编译关口是 AC922**。

## 4. 度量纪律（必守，踩过的坑）

1. **同源 A/B**：两侧必须**同一 build dir、同一套 CMake 参数**产出，唯一变量是待测代码；记录两侧 `--version`。
2. **整目录切换 + `LD_LIBRARY_PATH`**：`llama-bench`/`llama-server` 只是 ~210 KB 启动壳，代码在 `libggml-cuda.so`（~125 MB）里，壳的 `DT_RUNPATH` 是**指向 build 目录的绝对路径** ⇒ **只复制可执行文件做 A/B 完全无效**。
   **有效性自检**：`md5sum` 两侧的 `libggml-cuda.so` **和 `libllama-common.so`**（spec 代码在后一个里），相同即 A/B 无效。
3. **每配置 ≥2 次并报离散度**（本机 256K prefill 同配置能漂 **8%**）。
4. **投机解码的 tg 是"含接受的吞吐"**，随接受率一起动 ⇒ 归因 kernel 必须关 MTP（`--spec-type none`）或固定 seed；
   **报 t/s 必须同时报 AL 与每轮 ms**（`ms/轮 = AL ÷ tg`）；**投机解码对比必须 ≥3 prompt + 固定 seed**（单 prompt 结论会翻转）。
   **结果报告凡该任务测过预填充，必须附 pp 对比值（带提升/降低标注）；该任务没测预填充就不列**（用户 2026-09-22 定）。变化值一律标注「提升/降低」，tg 越高越好、ms/轮 越低越好。
5. 跑测前 `nvidia-smi` 确认空闲，并记录跑前跑后；（正式对外数字**必须带 `drop_caches`**，诊断可 `NODROP=1`）。
6. `>10 min` 的任务**必须 `nohup` 到后台**；等服务就绪轮询 HTTP `/health`，不要 grep "model loaded"。
7. `llama-bench` **不打印库内 INFO 日志**（要用 `llama-server`）；`LLAMA_LOG_INFO` 在 libllama 里可能看不到 ⇒ 先用 `fprintf(stderr,...)` 证明函数被执行。
8. ⚠️ **已更正（E3）**：`graph_compute(res->get_gf(), ubatch.n_tokens > 1)` 的第二个参数是 **`batched`（选线程池），不是同步开关**；
   真正的同步在 `llama-context.cpp:1376`，条件是 `cparams.pipeline_parallel`，而它要求 `split_mode==LAYER && !has_tensor_overrides()` ——
   本 harness **永远传 `--tensor-split`** ⇒ **两种模式都不同步**。
   ⇒ 因此 **`enqueue_us` 是「异步提交窗口」，不是 GPU 时间**；旧文「M=8 可信 / layer 数字是假的」**不成立**（已在 HANDOFF §6.2 与 goal-phase1 §5 更正）。
9. **复现「当前最好」必须带 `P2P=1` 且 `L=/root/libdir-nccl`**：harness 默认 `L=/root/libdir-rt`（butterfly）且**不开 P2P**；
   记录里的 94.18/95.20 都是 `CARDS=0,1,2 SPLIT=tensor L=/root/libdir-nccl P2P=1` 跑出来的。
10. 诊断工具：`test-backend-ops perf -o <OP> -b CUDA0`（在 `build-nccl/bin/`）；**ncu/nsys 路已封死**（见 §1）。
11. 本地 Windows 树与服务器树 md5 不同是**行尾差异**（`core.autocrlf=true`）⇒ 用 `git ls-files --eol` 判定，别慌。
12. **`llama-bench` 与 `llama-server` 的 `-ts` 语法不同**：`llama-bench` 要**斜杠**（`-ts 1/1/1`），`llama-server` 要逗号（`--tensor-split 1,1,1`）。
    2026-09-20 实测：给 `llama-bench` 传逗号 ⇒ split 解析成垃圾 ⇒ **在 `llama_model_load_from_file` 直接 abort**（只打回溯、错误正文被 tail 吃掉，极易误判成「模型加载失败」）。
13. **远端命令连 `#` 都不能出现**：`grep -a -e '###'` 这类写法在 Windows→ssh→bash 路径上引号会被吃掉，`#` 变注释 ⇒ `-e` 丢参数。规矩：远端命令**只留裸参数**，引号与 `#` 一概不要。
14. **无盘 NFS + 有限网卡**（用户 2026-09-20 提醒）：模型加载慢是正常的；**不要并行跑多个加载/编译/下载**；`llama-bench` 每臂都要重新加载一次模型（29 GB），排实验要算这个成本。
15. ⚠️ **`sed -i s/\r$//` 不加引号会被 bash 吃掉反斜杠** ⇒ 实际执行 `s/r$//` ⇒ **把 LF 文件里每行结尾的字母 `r` 删掉**（实测：`libdir-instr` 变 `libdir-inst`，脚本静默损坏）。
    - 本机教训（2026-09-20）：我上传的所有 **LF 脚本**都中招；**C++ 源码是 CRLF**（行尾是 `\r` 而不是字母 r）所以**没被破坏**，编译成功即印证。
    - **正确做法**：**本地就转成 LF 再 scp**（不要指望远端 sed）；或用 `cat -A` 抽查上传后的文件；或 `grep -n` 抽查关键字符串（我后来靠这条才发现）。
16. **编译**：本机原先**没有 ccache**（`GGML_CCACHE=ON` 但 `GGML_CCACHE_FOUND-NOTFOUND`）；2026-09-20 已用 `dnf install -y ccache`（3.7.7）装好并重新 configure 了 `build-instr`（`GGML_CCACHE_FOUND=/usr/bin/ccache`）。
    **改动后走增量编译**（同一个 build dir 直接 `cmake --build build-instr -j128`，只重编改动的 TU；
    **并发度用 128**——2026-09-21 用户明确授权「可以 128 线程并发编译」；POWER9 176 核，实测 `-j128` 稳定）；**不要把编译输出 `| tail -30`**（会吃掉真实报错；`Error 2` 但 `error:` 计数为 0 就是这么来的 —— 先重跑一次抓全量日志）。
17. **本模型几何（GGUF 实测，勿再写错）**：`head_count=24`、`head_count_kv=4`（GQA=6）、`key_length=value_length=**256**`、`embedding_length=5120`、`block_count=65`、`full_attention_interval=4`（16 层全注意力 + 48 层 GDN）。
    ⇒ 任何 FA/attention 改动前**按 D=256 / 24Q / 4KV 重新核对模板实例化**（旧文档的 128/40/8 与 40/4 都是错的）。
18. **A/B 的 md5 自检必须覆盖四个库**：`libggml-cuda.so`、**`libggml-base.so`（调度器在这里）**、**`libllama.so`（llama_context 在这里）**、`libllama-common.so`；
    再加一道**二进制内标记校验**：`strings <lib> | grep -a -c <新格式串>`，确认新代码真的进了**被加载的那个库**（只看源码 grep 会骗人：本轮就因此白跑了一次 A/B）。
19. **上传与编译不能并发**：改完源码**先上传+校验**，再启动编译/跑测；**编译期间绝不改服务器源码**——否则会出现半新半旧的库，结论作废。
20. **结构性事实（2026-09-21 定论，别再试局部补丁）**：llama.cpp 的 `ggml_backend_sched` 只有**单份** splits/graph，
    且上游 #28549 的守卫要求"**连续同一个图 arena** 才复用"；而 DFlash2 每轮是 注入(AL) ↔ 块前向(8) **交替**，
    ⇒ 每轮必然 reset + 重切 + 重分配：主图 `alloc_splits` ≈ **5 ms/次**（`GGML_SCHED_SPLIT_CACHE` 与"按批大小分槽的 arena"实测均**无效**），
    draft 侧图管理 ≈ **12 ms/轮**。要根治只能走静态形状 / 整轮单图（1cat fullgraph 路线），属大改动。
21. **长等待的跑测交给子代理**：主线只做改码/分析，跑测（编译 + 双臂 ~10-30 分钟）派后台子代理，任务书写清 ssh 形式、脚本路径、要回报的 grep 行与禁令。
22. ⚠️ **服务器源码树不是 git 仓库**（2026-09-21 实测：`/root/llm/test/v100-opt/llama.cpp` 没有 `.git`）。
    带 `2>/dev/null` 的 `git status` 会**静默失败**，看起来像"干净"——本轮据此误判过一次，且当时 `libdir-instr` 里其实是**带 push 实验码**的构建
    （`grep -c push_flag_stride` = 7 + `strings | grep -c "push allreduce enabled"` = 1）。
    **规矩**：判断服务器状态**只用 md5 + 二进制标记串**；权威同步 = 本地 `git archive HEAD`（tar）-> scp -> 服务器解包覆盖。
24. ⚠️ **同一台机器上永远不要并发跑两个测量任务**（2026-09-21 实测教训）：即使两个任务的卡**不重叠**（一个在 0/1/2 = NUMA0，一个在 3/4/5 = NUMA8），
    仍会互相污染：主机侧（采样/selector 线程、元后端 417 次 graph_compute 的提交）与**页缓存/NFS**（两个 29 GB 模型）都是共享资源，
    而本项目每轮的瓶颈恰恰**在主机侧与页缓存**上 => 实测 `llama-bench -d 32768` 得到 `13.45 +- 6.62 t/s`（离散 49%，反推斜率与 256K 实测自相矛盾）=> **该臂作废**。
    **规矩**：测量必须独占机器；"并行加速"只适用于**编译**与**纯读码/调研**。

25. ⚠️ **`BUILD_RC=0` 不等于"真的编译了"**（2026-09-21 实测踩到）：一次 `cmake --build` 只输出了 `BUILD_RC=0` 一行、**没有任何 `Built target` 或编译行**，
    实际是命令根本没执行（根因仍是 §3.13 的引号陷阱：`bash -c "..."` 内层串被 Windows argv 吃掉 ⇒ 空命令返回 0）。
    **规矩**：构建**必须**放在 `.sh` 文件里执行（不要用 `bash -c "长串"`）；
    构建后**三查**：① `grep -c -e error` 为 0；② 日志里有 `Built target`/`Building` 行；③ **二进制标记串** `strings <lib> | grep -c <新串>` 非 0（§4.18 已要求，本次是它救了场）。
    需要强制重编时**直接删目标 `.o`**（`rm -f build-instr/.../<file>.cu.o`）最稳。

26. ⚠️ **harness 端口不设防会把整批实验静默作废**（2026-09-21 实测）：`/root/p60-ab-harness.sh` 固定用 `PORT=8120`；上一臂的 `llama-server` 若没被 `pkill -9 -x llama-server` 收干净（它仍占着 309 MiB x3 的 CUDA 上下文），下一臂的 server 会直接
    `E srv start: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 8120` => `health ok=0` => **`STATUS=LAUNCH_FAILED`**，一次 8 臂全废。
    更坏的是 `pgrep -xc llama-server` **有时为 0**（进程在退出中/名字被截断），所以"看起来没有残留"并不作数。
    **规矩**：① 每臂之前必须 `pkill -9 -x llama-server` 并**等到端口真的可连**（用 `(exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null` 探测，最多等 90 s）；
    ② 每臂**用不同 PORT**（如 8131..8138）规避 TIME_WAIT；③ 看结果时**先确认没有 `LAUNCH_FAILED`**，再读数字。
27. ⚠️ **判断投机指标不要 grep server 日志**（2026-09-21）：harness 用 `grep ... | tail -1` 抓 "tokens per second" 与 "mean len"，既**有 `tail -1` 竞态**（响应返回与日志落盘的先后）**口径也不明**。
    **权威来源是响应 JSON 的 `timings`**：`/tmp/p60-<TAG>-p{1,2,3}.json` 里的 `predicted_n / predicted_ms / predicted_per_second / draft_n / draft_n_accepted`。
    工具：`python3 /root/timings.py <tag1> <tag2> ...`。由此可精确算 `rounds = draft_n / n_max`、`AL = predicted_n / rounds`、`ms/轮 = predicted_ms / rounds`。

29. ★ **动手否定/设计一个结构性修法之前，先查上游 PR 与 issue**（用户 2026-09-21 明确要求：「要习惯使用互联网搜索 PR 或者论坛来确定一些事件」）。
    本项目 base 是 b11053，很多「我们以为是自己的发现」其实上游已有 PR；反之我们的疑点也常有上游讨论可直接借用。
    查证路径：`web_search` -> GitHub **API**（`https://api.github.com/repos/ggml-org/llama.cpp/pulls/<n>` 与 `/issues/<n>/comments`，
    比抓 HTML 干净得多：HTML 主要是模板噪声，API 直接给 title/state/merged/body）。
    实例（2026-09-21 Round 100）：本想自己推翻 draft 图复用的修法，一查发现 #28549（已合并，就是我们树里那两个槽的来源）、
    #25406（已关闭未合并，独立指出 `split_graph` 每次换 uid 导致复用快路径永远打不中，并给出 `GGML_SCHED_SPLIT_UID_REUSE` 的实现思路）。
    **规矩**：① 任何「上游已知问题」的判断都要有 PR/issue 编号做依据；② 引用时记下 **state/merged/日期**（未合并的 PR 只能当参考不能当依据）；
    ③ 查证结果写进 `SESSION` 文档并注明 URL。

28. ⚠️ **解包会刷新 mtime ⇒ 触发全量重建**（实测 2026-09-21：`git archive` 解包后 mtime 全变新，重新编译了**全部 CUDA 模板实例**，`ps` 显示 nvcc 是**直接调用**的
    => **ccache 不覆盖 CUDA 目标（只覆盖 C/C++）**，所以 `.cu` 无论内容是否变过都会重编，时长以十分钟到一小时计）。
    另外：构建脚本**必须先查 BUILD_RC 再继续**——`cp -a build-instr/bin/. /root/libdir-instr/` 在构建失败时会把**旧库**覆盖过去，
    后续所有"测量"都变成在旧库上跑（静默作废）。

30. ⚠️ **绝对不要用 `source script.sh` 去"语法检查"一个 harness/测量脚本 —— 它会真的执行整个脚本**（2026-09-21 R145 实测：我用 source 检查 sed 补丁，结果真的启动了一个 29 GB 的
    llama-server（libdir-rt / 端口 8120）并执行了 `drop_caches`，撞上另一个代理的正确性门，违反 §4.24）。
    **规矩**：语法检查只用 `bash -n`；查变量默认值用 `grep`/`sed -n`；**永远不要 source**。清理误启动的进程用**精确 PID `kill -9`**，并事后核对 `pgrep -a -x llama-server`。

31. ⚠️ **实验性源码改动会留在服务器树里，下一次构建就静默带上它**（2026-09-21 R147）：R3 的 FA D256 常量 A/B 把 `fattn-mma-f16.cuh` 改成 jusko 版（实测 **-1.2% 回归**），
    实验结束**没有还原**；此后任何人 `cmake --build build-instr` 都会继承这个回归。已用 `/root/faab/fattn-mma-f16.cuh.orig` 还原并核对 md5 `bb366ccaeaf1f053d3a96baf7e1a972a`。
    **规矩**：① 只为实验的源码改动必须先留 `.orig`；② 实验结束**立刻还原 + md5 核对**；③ 要保留就必须同步更新"服务器树当前状态"记录（否则下一次构建的含义不明）。

32. ⚠️ **`pgrep -x llama-bench` 会自匹配**（2026-09-21 R147）：命令行里含该字面串的 `bash -c` 包装进程会被命中，导致 busy guard 误判中止（子代理实测两次）。
    busy 检查一律用**括号转义 + 上下文锚定**：`pgrep -f 'llama-benc[h] -m'` / `pgrep -f 'llama-serve[r] --model'`；`pgrep -a` 输出里的自身匹配同样要按这个思路排除。

33. ⚠️⚠️ **`--ctx-size` 只分配 KV，不填充它**（2026-09-21 R160，**我自己踩的**）：`p60-ab-harness.sh` / `z1-harness.sh` 的固定 prompt 是**一句话**（P1/P2/P3 各约 100-200 字符），
    所以 `CTX=8192 / 32768 / 131072` 三个臂的 decode **全都在 n_kv 只有几百的同一条件下发生** —— 我据此得出的「零斜率」是**测量设计的假象**，并一度写进了本文件 §1（最危险的地方，因为每次会话自动加载），已撤回。
    **硬证据**（来自响应 JSON，`/tmp/p60-<tag>-p{1,2,3}.json` 的 `prompt_n`）：三臂**逐字相同** —— 8K / 32K / 128K 都是 **76 / 106 / 84** token；
    三臂 tg 44.90-45.31 t/s（22.02-22.18 ms/token，总离散 0.7%）—— 条件相同，所以相等，**与深度无关**。
    **规矩**：① 凡是要变上下文的实验，必须先**证明上下文真的变了**（看 `[RT] perf` 的 `n_ctx`/`rounds` 不够，要看响应 JSON 里的 KV 使用或 prompt 长度）；
    ② 正确的深度测试用 **`llama-bench -d <depth>`**（有真实深度旋钮），或先用长 prompt（`/tmp/prompt256k.txt`）把 KV 填满再测 decode；
    ③ **「改了哪个参数」不等于「那个参数真的起作用了」** —— 上机前确认实验条件真被改变，否则会得到一条漂亮但虚假的曲线。
34. TIME - **A/B 的墙钟有 85-95% 花在模型加载上，而它是网速硬顶；诊断必须走快路径**（2026-09-21 R161，用户提问触发）：
    - 实测：每个 harness 臂 `health ok=1 after 255s`；29.0 GB / 255 s = **114 MB/s**，正好是无盘 NFS 的千兆线速。
      FA prefill 臂 284 s 里只有 **15 s** 是真正的测量（pp32768 @2163 t/s）=> **加载占 95%**。
    - 同一文件热读（`NODROP=1`）harness 自述 **约 15 s**（1.9 GB/s）=> **冷/热差 17 倍** => 瓶颈在**网络路径**，不在 GPU、不在 PCIe（29 GB 过 PCIe 只要几秒）。
    - **两档口径（必须事先声明，且同一次比较里不许混用）**：
      (1) **官方数字**（要写进结论的）：`drop_caches` + 全尺寸 Q8_0 —— 这 255 s 必须付；
      (2) **诊断 / 斜率 / 探针**：`NODROP=1`（约 15 s）+ 可选小模型 Q2_K_XL（9.14 GiB，冷加载约 80 s）。
    - 另外两个可叠加手段：`llama-bench -r 3`（一次加载出 3 个点，**顺带得到离散度**）；`llama-bench -d` 若支持逗号列表则**一次加载扫多个深度**（待验证）。
    - 教训：R161 之前 Z1 的三个**诊断**臂按**官方**口径做了冷加载，白付约 13 分钟。
    - **每小时能做的实验数**是本项目的真实稀缺资源 —— 排实验时先问：这一臂需要官方口径吗？
35. WAIT - **长等待一律用「远端内部轮询 + 结束符」的单次阻塞调用，不要反复发起短调用**（2026-09-21 R166，**用户提出的标准流程**）：
    - 症状（本会话真实发生）：一个诊断臂冷加载 260 s，而我每轮 ~13 s 就去问一次 ⇒ **空转十几轮、烧掉大量上下文**，而结果不会因此提前到达。
    - **标准流程（两级看门狗，脚本 `/root/wait-for.sh`，已实现且三路功能测试通过）**：
      (1) **内层 60 s 轮询**目标文件里的结束符；**同一循环里做异常检测**：文件不在、或**文件大小连续 N 分钟不变**（默认 5 tick）⇒ 立刻中断并返回 `ABORT_STALLED`；
      (2) **外层每 10 分钟打一行 `HEARTBEAT`（带文件大小）**，防止内层自己出故障时静默永久等待；再加 `max_minutes` 硬上限 ⇒ `TIMEOUT`；
      (3) **永远只打印一行 `WAIT_RESULT=...`**（`OK` / `ABORT_NOFILE` / `ABORT_STALLED` / `TIMEOUT`），调用方按它分支，不靠猜。
    - WARNING **异常判据不要用 `pgrep`**：`pgrep -f <模式>` 会命中**脚本自己的 argv**（命令行里就含那个模式）与 `bash -c` 包装 ⇒ 实测让「生产者已死」这一支**永不触发**、退化成 `TIMEOUT`。**改用「输出文件停止增长」**这个与进程名无关的信号。
      （与 §4.32 同源：**任何依赖「命令行里出现某字符串」的判据，都要先问「我自己会不会匹配上」**。）
    - WARNING **一次阻塞调用的 timeout 要覆盖真实最坏耗时**（本会话用 900000 ms = 15 min 成功等到两臂）。若外层工具提前超时，只是回到轮询，不会损坏任何东西。
    - ⇒ **「每小时能做多少实验」是稀缺资源；「每小时我发起多少次无意义查询」同样是。**
36. ⚠️ **`llama-bench -r 1` 的均值被「第 1 次重复」的瞬态偏低 20-27% —— 要写进账的数字必须 `-r >= 8` 并报 `+-`**（2026-09-21 R168，**子代理发现并自查出来**）：
    - 硬证据（同深度、同口径、只改 `-r`）：`d8192 -r 1 = 33.62 +- 0.00 t/s`（29.744 ms/token）vs `d8192 -r 8 = 41.43 +- 2.86 t/s`（24.137 ms/token）=> **+23% 吞吐**。
      第二处独立确认：`d300 -r 2 = 38.81 +- 6.62` vs `-r 10 = 42.61 +- 3.01`；按 `mean(r) = (S + (r-1)F)/r` 反解出 **S = 34.06、F = 43.56 t/s**（rep1 慢约 22%）。
    - ⇒ **稳态下 d300(F 43.56) 与 d8192(F 42.55) 只差约 2%** —— 也就是说，用「r=10 的稳态」比「r=1 的瞬态」得到的所谓**浅段 0.7952 us/ctx-token 是假象**（H5 已撤回）。
    - ⇒ 这条**同时解释了** §4.24 里那个「`llama-bench -d 32768` = 13.45 +- 6.62 t/s、离散 49%」的臂：先别怪并发，先查 rep 数与离散。
    - ⇒ **不受影响**：harness 的数字（NPRED=512，一整段生成的稳态平均，瞬态被摊掉）—— 438 GB/s、权重流 40%、两阶段策略仍成立。
    - **规矩**：① 任何要写进文档/结论的 `llama-bench` 数字：`-r >= 8`；② 只有 `-r 1` 时**只能当探路**，不得进账；③ 报数必须带 `+-`；
      ④ 多深度用**一次加载的逗号列表**（`-d 256,384,...,131072`，`-r 8`），且**升序排列**（若瞬态是「每臂一次」，它会落在最便宜的深度上）。
37. ⚠️⚠️ **替换共享库会杀死正在跑的测量（SIGSEGV 139，且日志里什么都不打）—— 构建与测量必须互斥**（2026-09-21 R172，实测踩到）：
    - 现象：Z1 的全深度臂 17:41:59 启动，6 s 后 `BENCH_RC=139`，**连 `ggml_cuda_init` 都没打出来**；同一时刻 probe 链在执行 `cp -a build-instr/bin/. /root/libdir-instr/`。
    - 解释：动态加载器已 mmap 那两个 .so，而 `cp -a` 是**原地截断/重写同一个 inode** ⇒ 运行中进程的代码页被换掉 ⇒ SIGSEGV。**不是**参数列表/内存/llama-bench 的 bug
      （Z1 用不存在的模型 + 空 `CUDA_VISIBLE_DEVICES` 复现同一列表：正常报错、`PARSE_RC=0`，不崩）。
    - **规矩（双向，缺一不可）**：
      ① **构建方**：编译**和** `cp -a` 到 libdir 之前必须确认无 `llama-bench` / `llama-server`；落地期间用锁文件 `/tmp/LLAMA_BUILD_LOCK` 声明占用，结束才删除；
      ② **测量方**：preflight 除查 llama-bench/llama-server 外，还要查 `pgrep -f 'cmake --buil[d]'` **与 `/tmp/LLAMA_BUILD_LOCK` 是否存在**；
      ③ 任何『我以为对方会退让』的假设都不算数 —— 本次就是两边都以为对方会等。
    - 附带：**`drop_caches` 也会毁掉别人的 NODROP 臂**（页缓存被清 ⇒ 29 GB 冷加载 255 s）=> 它同样属于『修改机器状态』的动作，必须独占机器。
38. ⚠️ **设备集（`CUDA_VISIBLE_DEVICES`）是一个 8.8% 的隐藏变量；斜率可分辨性要用 SEM 判**（2026-09-21 R171，子代理实测 + 自审）：
    - 同深度、同 `-r 8`、同 drop_caches 口径：`d8192` 在**6 卡可见**下 **41.43 +- 2.86 t/s**，在 **`CUDA_VISIBLE_DEVICES=0,1,2`** 下 **45.08 +- 2.74 t/s**
      => **差 8.8%**，而两臂唯一差别就是「看得见几张卡」（`-ts 1/1/1` 两边相同）。harness 一直显式设 0,1,2，**手工 llama-bench 臂没设**。
    - ⇒ **作废**：R165 那三个 r=1 点与 R168 那两个 r=8 点（都是 6 卡可见口径）**不得再引用**；深度结论一律以 R170 的 0,1,2 密集阶梯为准。
    - ⇒ **规矩**：任何 `llama-bench` / `llama-server` 臂**必须显式写 `CUDA_VISIBLE_DEVICES`**，与对照臂一致，并记进结果行。
    - **可分辨性判据**：报斜率用 **SEM = `+-` / sqrt(r)**，不是原始 `+-`。R170 阶梯（r=8，SEM = 0.35-0.49 ms/token）里
      **每一对相邻深度的差（0.005-0.334 ms/token）都小于自己的 SEM** ⇒ 相邻斜率**一个都不可分辨**，只有聚合量可分辨
      （256->8192 = 0.105 µs/ctx-token；4096->8192 = 0.138）。原始 `+-` 由 rep1 瞬态主导且对各深度相同，差分时会抵消 —— 所以 SEM 才是正确的尺子。

## 5. 代码现状（改动只落在 `llama.cpp/`）

- 本地 `F:\vllm+llama.cpp\llama.cpp`：**HEAD `fb2fbcdea`（= FGC 三提交顶点 `6c4452462` + selector 向量化默认 `0db90abd6` + sm70-attn D256 `fb2fbcdea`）；⚠️ 工作区有 1 个未提交改动 = T8 多槽 gallocr 三文件（`ggml-alloc.{c,h}` + `ggml-backend.cpp`，账本 R273 = B4，diff 归档 `wip-galloc-slots.patch`）**；base 钉在 tag b11053，不跟 master。更早的 9 个自建 commit 见 `HANDOFF.md`。
  ⚠️ **本文件的有效装载上限低于 65,536 B**：工作区指令预算是**共享**的（`llama.cpp/AGENTS.md`、`llama.cpp/CLAUDE.md` 也占），实测 65,391 B 时已被**截掉 147 B**（截的是装载副本，磁盘文件完整）⇒ 维护时请把本文件控制在 **~64.8 KB 以内**，优先删过期内容而不是新增。
  ⚠️ **服务器树当前状态（R274 核实）**：R257 状态 + **T8 多槽 gallocr 三文件（默认 ON 版，与本地工作区一致）**；**B4 基准库 = `/root/libdir-t8b`**（`libggml-base 19675f8b…`、`libggml-cuda 30876545…`、`libllama 031009ae…`、`libllama-common d3affdd2…`）；`/root/libdir-t8`（=8 旧版）/`libdir-instr`（B3）/`libdir-sm70`（R270）为历史快照。**不再假设它等于 HEAD。**
- 相对 b11053 = **13 文件、+587/−33**，四类：
  | 类别 | 文件 | 说明 |
  |---|---|---|
  | **C4** Volta MMVQ 参数 | `ggml/src/ggml-cuda/mmvq.cu` `mmvq.cuh` | `MMVQ_PARAMETERS_VOLTA`：V100 ncols=1 → nwarps=2；tg128 36.23 → 37.52（单卡 +2.8%） |
  | **C5** K-quant 交叉点 | 同上 | V100 的 mmvq↔mmq 交叉点 = **4**（上游独缺 Volta，落回默认 8）；生产 Q4_K_M ne11=8 **+15.0%** |
  | **PR #27858** DFlash2 CPU selector | `common/speculative.cpp/.h`、`common/common.cpp`、`src/llama-{ext.h,model.cpp,model.h}`、`src/models/dflash.cpp` | 修 `tensor + draft-dflash` 硬崩；selector 已并行化（8 线程） |
  | **计时量具**（env-gated） | `src/llama-context.{h,cpp}`、`common/sampling.cpp`、`tools/server/server-context.cpp` | `LLAMA_ROUND_TIMING` / `LLAMA_SPEC_TIMING` |
- ⚠️ 树里留有 **7 处 `[RT]` `fprintf` 调试探针**（已核实：`llama-context.cpp` 4 / `server-context.cpp` 2 / `sampling.cpp` 1）——清理属待办 4，另开 commit。
- 研究档案仓库 `1cat-vllm-v100-study`（独立 git 仓库）。**接手必读的四份（2026-09-20 DSH 新建）**：
  | 文件 | 用途 |
  |---|---|
  | `AUDIT-2026-09-20-dsh.md` | **可信度索引**：E1–E16 更正 + F1–F7 修正事实 + 已落地改动清单 |
  | `1CAT-PORT-BACKLOG.md` | **移植清单**（1cat v1.5.0 DFlash2 特性 + SM70 代码区 + 第三方 llama.cpp 先例 → 批次 P1–P7） |
  | `EXTERNAL-REFS.md` | **外部先例 + 官方模型卡与推荐参数**（含与我们现有配置的对账） |
  | `CODEX-ARCHIVE-INDEX.md` | 工作区外的早期 Codex 资料索引（**只索引不复制**，含「勿沿用其结论」的理由） |
- 历史档案：`FINAL-REPORT.md`、`goal-phase1-findings.md`（§1-§18 最全）、`premise-check-1cat-vs-llamacpp.md`、`phase0-round-breakdown.md`、
  `1cat-sm70-gemm-tactics.md`、`c5-volta-crossover.md`、`same-source-ab.md`、`dflash-in-llamacpp.md`、`mtp-sampler-cpu.md`、`stress-test-256k.md`、`l3-acceptance.md`、`baseline.md`、`WORKPLAN.md`、`core-changes.md` 等（**已被审计更正处均已就地标注**）。

## 6. DSH 迁移说明（Qwen Code -> DSH，2026-09-20）

| 项 | Qwen Code（旧） | DSH（现在） |
|---|---|---|
| 自动加载的规则 | `QWEN.md`（工作区根，60 KB） | **`AGENTS.md`（本文件，精简）**；`QWEN.md` 保留为持久规则全文，按需读 |
| 项目技能 | `.qwen/skills/llama-stress-test/` | **`.dsh/skills/llama-stress-test/`**（已复制） |
| 全局设置 | `.qwen/settings.json`（模型/权限，DSH 不适用） | DSH 配置在 `~/.dsh/`（`settings.yaml`、会话、profiles） |
| 设备档案 / SSH 技能 | `~/.qwen/skills/ssh-server/`（`HOSTS.md` 等） | 旧目录仍可查阅；DSH 直接用 shell 里的 `ssh`（key 已授权） |
| 子代理 | Qwen 的 `Explore` 子代理当时**持续 500 不可用** | DSH 子代理可用；**读码/调研可派子代理，改码/上机/对用户的结论仍走主线**；子代理产出**当证据不当结论**，关键行号要抽查 |

**新会话开场建议动作**：`ssh -o BatchMode=yes root@192.168.50.235 'nvidia-smi; systemctl is-active vllm-1cat llmscope'` 看 GPU 与服务状态，
再按 HANDOFF §12「复制粘贴区」继续。
