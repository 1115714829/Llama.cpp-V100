# HANDOFF — llama.cpp V100 / SM70 专项优化项目交接
> ## ★ 最新定格（2026-09-21 Round 143 - 读这一段就够）
>
> **目标（用户重申）**：tg **>= 150 T/s**（= AL 5.55 下 **ms/轮 <= 37.0**，现 55.5）；180 / ~20 ms 是上界；对标 1cat 17.463 ms/轮。
> **入口是 `PLAN-GRAPH.md`**（§1 主图 / §3 七个外部项目解读矩阵 / §3.4 使能链）；**账在 `PLAN-to-180ts.md`**（本轮已按 150 重写）。
>
> **本轮新事实**
> 1. **N0 定论（R142）**：target decode 后强制同步只多花 **0.4-0.8 ms/轮**，两臂 `draft_n`/`draft_acc` 逐位相同
>    ⇒ target 步**本就同步**、无整轮重叠 ⇒ **可加性模型成立**（轮时 ≈ 主机 + GPU 串行）⇒ **省主机时间是有效的**。
>    R1 的『省 4.8 ms 主机但轮时不动』= 代价从主机搬到了 GPU。证据 `N0-VERDICT-2026-09-21.md`。
> 2. **N3（D256 FA 常量 A/B）**：`libdir-fa-base` pp32768 = **2163.26 t/s** vs `libdir-fa-v2` = **2135.90 / 2137.86**（两轮）
>    ⇒ jusko 的 D256 常量在我们这里**慢约 1.3%**（待 base 第二轮复核后定性）。1cat 公开 32K/64K prefill = 3567-4069 ⇒ **预填充差 1.65-1.88x**。
> 3. **7 个参考项目解读矩阵完成**（57 条提取物 + 证据等级 A/B/C + 『该抄哪个文件:行』反查表）见 `PLAN-GRAPH.md` §3。
> 4. **使能链 §3.4**（用户要求『A 现在看着亏但 B 的基础是 A』）：E1 push AR -> N11、E2 GPU selector -> N11、E3 metadata 缓存 -> N11、
>    E4 降 n_max/T=4 -> jusko 全部 T=4 栈、E5 N1 -> N8/LOWBIT/N3、E6 f16 KV -> K1 改写、E7 MMVQ x4 -> LOWBIT、E8 主机循环 -> K4。
> 5. **新增节点**：N12（MMVQ x4）、N13（KV dtype A/B）、**N14**（按 context 自适应 n_max，零代码）、**T4**（用 AL 换 T=4 内核生态）。
>
> **头号判断（`PLAN-to-180ts.md` §2）**：到 150 需要 -18.5 ms；**只要 N6（主机 metadata 缓存，-12~-16）与 N1（q8_0 KV 张量核，-3~-13）各自落地一半就到了**。
> **先手是 Z1-Z5 零代码量测**（ctx 斜率 / FA kernel 只读诊断 / n_max 扫描 / KV dtype / split 模式<已完成>），必须排在结构性改动之前。
>
> **下一步（按序，每条都带现成证据/脚本）**
> 1. **收 Z1 的 128K 与 128K-DFlash2 两臂** —— 判据**已预先写在** `Z1-PREREG-2026-09-21.md` §5.5：
>    128K 无投机 ~45 t/s ⇒ KV 仍被隐藏（N1/N8T 在 128K 以下无价值）；~24 t/s ⇒ KV 开始显形。
> 2. **Z2 探针臂**：编译代理在等机器，一次编译带两个探针（`GGML_CUDA_FA_KERNEL_DEBUG` + `GGML_META_KEY_DEBUG`）。
>    要看两件事：`[FAK]` 确认 8-token verify 走 **TILE** 且 `n_kv % 256 == 0`（决定 `use_gqa_opt` 成不成立）；`[MKEY]` 判 CUDA 图 key 是否在旋转容器间交替。
> 3. **阶段一第一项 = N6b（形状稳定化）** —— §2.7 证明它是 150 的**硬前提**（没有它阶段一只有 tg 128）。
>    第一步是**纯诊断**：`GGML_META_NODE_DIFF`（设计见子代理规格，已记在 §4 R149），逐字段报「哪些 ne/nb 在变、变成什么」。
> 4. **Z3（n_max 扫描）无需新脚本**，直接传 `SPEC="--model-draft ... --spec-draft-n-max N"`；**Z4（KV dtype）用 `/root/z4-harness.sh` 的 `CTKV` 旋钮**。
> 5. **Z7（模型大小标定）**：同 harness 换 Q2_K_XL ⇒ 判定 438 GB/s 里多少是带宽、多少是固定开销，直接决定 BW1（+16%）与 N6/P-B 的优先级。
> 6. **N7 暂缓**：ninfer 的 op 是「路径选择」，与我们的 CPU selector 语义不同（R159），要先做语义映射。
> > **R146-R158 更新（覆盖上面 R143 的部分内容）**
> 6. **有效带宽 438 GB/s**（Z1 实测）⇒ 带宽类估算 ÷1.82；「权重流 12.1 ms 已到顶」**撤销**，实为 **22.1 ms**（占轮时 40%）。
> 7. **LOWBIT 改判 +23~33%**（原 +10~13%）；新增主线 **BW1**（q8_0 效率 438->700 = +16%）。
> 8. **两阶段**：阶段一主机三件套（N6a+N6b/P-B/N7）⇒ tg ~161；阶段二 LOWBIT/BW1 ⇒ 180。**N6b 是硬前提**（否则只有 128）。
> 9. **Z1 零斜率**：8K 45.31 / 32K 45.25 t/s ⇒ KV 在 128K 以下被隐藏 ⇒ **N1/N8T 是 256K+ 项**。
> 10. **N3 已判**（jusko D256 常量 pp32768 −1.19%，只覆盖 ncols=64 行，greedy sha256 一致）；**sm70-attn 审计**：两道硬门把我们挡掉，对 verify 零影响，数字不可复核（C 级）。
> 11. 新增两个 env 门控探针（已提交、已上传服务器树）：`GGML_CUDA_FA_KERNEL_DEBUG`（a8fb6542f）、`GGML_META_KEY_DEBUG`（32df47bd5）。
> 12. 纪律新增：实验源码改动必须还原（§4.31）、`pgrep -x llama-bench` 自匹配（§4.32）、**绝不 source 脚本**（§4.30）。
> > **状态**：`llama.cpp` HEAD 见 `git log`（11 个自建 commit）；`study-docs` 见 `git log`（含 §3 九项目矩阵、§3.4 使能链、§3.5 上游侦察、§3.6 N8T 配方、N0/N3 判决、Z1 先验预测）。
>
> ---
>
> ## ⏱ 5 分钟接手块（2026-09-21 会话末定格 - 以下为 Round 98 时的定格，部分已被上面覆盖）
>
> **状态（2026-09-21 Round 98 更正）**：本地 `llama.cpp` **停在 `c2d716519` 但工作树是脏的**（A2 索引式写入补丁 + `GGML_RS_INDEX_WRITE` 门控，未提交，gate 默认关）；
> **服务器源码树不是 canonical**：`/root/llm/test/v100-opt/llama.cpp` 里也带着同一个 A2 补丁（`grep -c GGML_RS_INDEX_WRITE src/llama-graph.cpp` = 2），
> 且 `/root/libdir-instr/libllama.so.0.4.1` 是 **13:21 用这棵树重建的**。A4 探针（`GGML_CUDA_FA_SPLIT_FLOOR`）也在，默认关。
> ⚠️ 两者都是 env 门控、默认关 => 行为等同 canonical（Round 97 的官方口径实测已用 `AL 5.55/4.22/6.38` + `sha256 f3edac19...` 逐位证明），
> 但**任何要把服务器树当基线的新实验，必须先用 `md5 + 二进制标记串` 确认它到底带了哪些补丁**（AGENTS §4.22）。
> 实验补丁 `patches/0001..0006`；研究仓库最新提交见 `git log`。
> **规格文件（2026-09-21 新增，用户已授权做结构性改动）**：
> - `PLAN-to-180ts.md` —— 总计划（四项 + 已证伪清单 + 零代码实验清单 Z1–Z5）
> - `SPEC-P-A-ar-in-graph.md` —— P-A（AR 纳入图捕获）：现状源码事实、两条候选路线、可复用资产（patches/0006 的设备侧协议）、四步实施、硬门与止损
> - `SPEC-P-D-longcontext-attention.md` —— P-D（长上下文 decode attention）：A4 证伪并行度路线的证据、三步计划（先向量化+ILP，再考虑专用内核）、止损
> - P-C（mask/KV 宽度分桶）与 P-B（draft 减少 kernel 数）：P-C 首次子代理尝试零写入被中断，记录在 `SESSION` §23（含三条判读路径）；P-B 的量化诊断进行中（`/tmp/pbdiag.log`，用已提交的四个探针，无需改代码）
> - **Round 95-96 新增（证据在 `SESSION` §25）**：
>   ① **AR == 切图边界**（`ggml/src/ggml-backend-meta.cpp:2434-2463`：每子图 = 每设备一次 graph_compute + 一次主机侧 AR）
>      => 每轮 139 子图 / 138 次 AR / **417 次 graph_compute**，每次 AR 都是隐式跨设备栅栏。
>      => **P-A 的价值在「合并边界」，不在「省主机时间」**（R1 已证伪后者）；只把 AR 挪进图而不合并边界 = 复现 R1 的零收益。
>   ② **8K TP 重扫（诊断口径）**：TP2/3/4/6 = 60.3 / 62.4 / 69.7 / 94.2 ms/轮 => **卡越多越慢**，权重流 1/N 的收益被 AR/提交全部吃掉；
>      非提交部分四臂恒定（40/35/31/32 ms）。⚠️ 绝对数字低于权威口径（NODROP + 背靠背），只取趋势；官方口径交错重测中（`/root/tp-ab.sh`）。
>   ③ **`--spec-draft-device` 的「必崩」结论可能已过期**：`src/models/dflash.cpp:160-161` 已按单设备 draft 设计，
>      `common/speculative.cpp:2817-2823` 对 `n_devs == 1` 直接给 `LLAMA_SPLIT_MODE_LAYER` => **draft 侧 0 次 AR、约 1/3 kernel 数**（draft 仅 1.14 GB）。
>      代价：会切到图内 selector（`speculative.cpp:994`）。正在干净口径重测（`/root/devd-ab.sh`）。
>      **判决（14:05）**：仍然崩，根因就是 2026-09-20 记的那条 —— `ggml-backend.cpp:942: pre-allocated tensor (output.weight) in a buffer (Meta()) that cannot run the operation (NONE)`。
>      本 draft 是全词表 draft（GGUF 无自己的 head），lm_head 就是 target 那个住在 `Meta()` 里的 tensor，单卡 draft 无法执行 => **该路线在当前架构下封死**（三条解法都属结构性改动，本轮不做）。
>      附注：`--spec-draft-device none` 语义是「不要 offload draft」=> 解析成空设备表 => 加载直接失败；默认值必须传**空字符串**。
>   ④ **⚠️ 度量口径事故已定论（Round 96）**：本轮一批数字误用 **NPRED=192**，而权威口径是 **harness 默认 NPRED=512**。
>      差异**不是库**：libdir-nccl 与 libdir-instr 在 192 下 `pred_ms` 2433.5 / 2544.0、`draft_n`/`draft_acc` 逐位相同（同一条采样轨迹），instr 还快 4.5%。
>      真因是 192 的 `pred_ms` 含约 **500 ms 一次性热身**（38.7 轮里占 13%，91.7 轮里占 3.6%）+ 短生成接受率偏低（p1 0.561 vs 0.653）。
>      ⇒ **报数字必须写明 NPRED**；判定投机指标一律用 `/root/timings.py <tag>` 读响应 JSON 的 `timings`，不要 grep server 日志（有 `tail -1` 竞态）。
>   ⑤ **官方口径（NPRED=512）重跑完成（`/tmp/round97c.txt`）**：
>      - **权威基线精确复现**：tp3a `MEDIAN_TG=98.64`、`AL 5.55/4.22/6.38`、`greedy sha256 f3edac19...` 逐位一致，**55.7 ms/轮**。
>        => 权威口径 = `CARDS=0,1,2 SPLIT=tensor L=libdir-instr P2P=1 NPRED=512`。**今后一律照此**。
>      - **Z2 关闭：TP3 最优**。TP2（两臂 83.72 / 83.39，离散 0.4%）中位 tg 83.5、57.9 ms/轮 => TP3 快 18%。
>        192 口径下排序一致（TP3 78.49 > TP2 60.52）=> §25.2 的相对趋势成立；加卡到 4/6 更差的原因在 §25.2/§25.6。
>      - 新事实：TP3 的 `enqueue/轮` 比 TP2 **多** 4.4 ms（20.3 vs 15.9），但总轮时仍少 2.2 ms
>        => **AR/提交成本不是每轮主导项**，权重流（每卡 1/3 vs 1/2）才是。**这条削弱了 P-A 的优先级**（其价值只在"合并 138 个切图边界"，不在省 AR 时间）。
>   ⑥ **★ 本轮最有价值的发现（`SESSION` §25.7）：draft 上下文的图一次都没复用。**
      两行 `[RT] perf` 都在日志里，之前只看了最后一行：target `reuse=270 rebuild=24`，**draft `reuse=0 rebuild=556`**。
      draft 的 `alloc_us` 总量 **1.043 s > target 的 0.648 s**（draft 只有 5 层 / 1.14 GB）=> 摊到每轮 alloc 约 1.9 ms + build 0.13 ms。
      **P-B 的正确做法因此改写**：不是「融合 draft 算子」，而是**让 draft 的图能复用**（target 在同样的「注入/块」交替下复用 92%，说明机制上可行）。
      纪律：判定图复用必须 `grep -a -h "RT. perf" <log>` **全取两行**，不要 `tail -1`。
>   ⑦ **根因已定位到源码（Round 97）**：`src/llama-context.h:371-374` —— 图结果缓存有**两个槽**
      （`gf_res_prev[n_outputs > 0]`，注释说是为了给「有输出/无输出」两批不同的 CUDA 图缓存键），
      但 `gf_res_prev_active` **只有一个指针**（`llama-context.cpp:1372/1387/1419`）。
      DFlash2 的 draft 每轮严格交替「注入(n_outputs==0) / 块前向(n_outputs>0)」=> 单指针永远对不上 => **reuse=0**；
      target 每轮只 1 次调用 => 同槽连续命中 => 92%。
      修法与风险分析见新规格 `1cat-vllm-v100-study/SPEC-P-B-draft-graph-reuse.md`（3 条候选路线 + 必须先证明的 arena 安全性）。
      已派子代理做**判定性探针实验**（env 门控 `LLAMA_GRAPH_SLOT_DEBUG`，只验证假设，不改默认行为）。
>   ⑧ **Round 103-106 实况（最新，优先于以上）**：
      两个判定性实验已出结果，**计划的价值排序因此改变**（数据在 `SESSION` §25.10）：
      - 槽位探针：draft 槽号**严格交替** => `hit=0/24`、`reuse=0`（假设成立）；且发现 `can_reuse` 本身也会 false（第二因）。
      - `GGML_META_HOST_TIMING`（主线写的探针，`ggml/src/ggml-backend-meta.cpp`）：**`dev` 72% / `ar` 15% / `prologue` 13%**。
      => **P-A 正式出局**（AR 主机时间只有 2.79 ms/call，即使归零也只值约 5% 轮时）。
      - `dev` = 13.3 ms/call = **约 90 µs/次设备调用**；同一次运行里 GRAPH 探针给出
        `calls=30208 capture=1596 replay=18495 direct=10117`（5.3% / 61.2% / **33.5% direct**），
        而 decision 只占 4.6 µs => **95% 的时间在 decision 之后，三分之一调用根本没走 CUDA 图重放**。
      - **新主攻 = `SPEC-P-E-direct-path.md`**：估收益 **每轮约 10 ms（~18%）**；第一步必须加 `GGML_CUDA_DIRECT_DEBUG`
        打出「到底哪一项属性不符」（**现成的 `prop diff` 探针会误报，不许当依据**）。
        若不符项集中在「mask/KV 视图宽度」「递归状态视图指针」=> **与 P-C、A2 同根**，那两项收益要按 decode 稳态重估。
      - ⚠️ direct 占比在漂（pbdiag 18% vs metadiag2 33.5%），官方口径 2 臂复测进行中（`/root/direct-ab.sh` -> `/tmp/direct-ab.txt`）。
>   **⚠️ 状态更正（Round 106）**：
      本地 `llama.cpp` **不再干净**，有 3 处未提交改动：A2 索引式写入补丁、`llama-context.cpp` 的 SLOT 探针、`ggml-backend-meta.cpp` 的 META 探针。
      服务器 `/root/llm/test/v100-opt/llama.cpp` 同样带着这些；`/root/libdir-instr/libllama.so.0.4.1` 是带 SLOT+META 探针的构建（均 env 门控、默认零影响）。
>
> **权威数字（正式口径，四次实测离散 <0.5%）**：tg **98.12 / 98.70 / 98.88 / 98.88 t/s**，AL 5.55/4.22/6.38，**57.6 ms/轮**，greedy sha256 **f3edac19...**（同配置逐位可复现）。
> 长上下文（§21）：8K 24.9 -> 256K **53.0 ms/token**（+113%，斜率 0.113 us/KV-token）；256K+DFlash2 投机 = 34.13 t/s。
>
> **每轮预算（§18.9 闭合到 56.6 ms）**：权重 12.1 + **AR 7.3**（事件计时 53 us x 138）+ M8 增量 6.5 + M1 其他 5.5 + alloc 2.2 + **draft 13.6** + selector 2.7 + **未归因 6.7**。
>
> **下一步只有两条路**：① **等你批准**的四项结构性工作（P-A AR 入图 / P-B draft 算子融合 / P-C mask-KV 分桶 / P-D 长上下文 attention 内核，判据见 `PLAN-to-180ts.md`）；
> ② 不需批准的收尾项：6.7 ms 余量的三处门控计时（采样器 / 投机接受记账 / 服务器簿记）、树里 7 处 `[RT]` 探针规整。
>
> **第一批命令**：
> ```
> ssh -o BatchMode=yes root@192.168.50.235 'nvidia-smi; systemctl is-active vllm-1cat llmscope'
> cd /root && env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=chk NPRED=512 bash /root/p60-ab-harness.sh   # 复现权威数字
> tail -3 /tmp/lc-par.log /tmp/a4.log /tmp/d256.log    # 本轮长上下文实测留档
> ```
> **勿重做（已证伪）**：见文末"不做清单"与 `PLAN-to-180ts.md` §4。
> ## ★★ 2026-09-21 DSH 会话实况（优先于本文件其余内容）
>
> **权威顺序**：本节 > `AUDIT-2026-09-20-dsh.md` > `SESSION-2026-09-20-measurements.md`（逐条实测与纠正，§0-§11 与 **§16**）> 其余。
>
> **◎ Round 45-54 新增：四个实验的判决（全部有受控 A/B 或源码证据）+ 决策表**
> - **R1（设备侧 push AR）证伪** ✗：主机侧确实省 4.8 ms/轮（`enqueue` 5.94->4.56 s），但**轮时无改善** =>
>   在图捕获体系下 **enqueue 窗口不在关键路径**（主机本就跑在 GPU 前）；且 push 内核在役 `ar_us_avg` **61.5 us vs NCCL 53.0 us（更差）**。
>   归档 `patches/0006`。**AR 的唯一出路是 R2（入图，1cat 18 us/次，可省约 4.8 ms/轮）**。
> - **A2（递归状态索引式写入）逐位中性但前提被推翻** ✗：8K 臂 **greedy sha256 逐位不变（f3edac19）** ✓ 证明索引与视图偏移语义等价，
>   但 `llama-graph.cpp:48-65` `can_reuse_kq_mask` 要求 `ne[0]==n_kv`，而 n_kv 每块增长 => prefill 每块重建与递归 head 无关。归档 `patches/0005`。
> - **A4（更深的 FA KV 切分）证伪** ✗：`-d 131072` 下 floor 0/64/256/1024 = **26.85 / 24.59 / 22.71 / 22.43 t/s（单调变差）**
>   => 上游"一波"启发式在本形状就是最优；**128K+ decode 是内核吞吐问题**（VEC 有效带宽约 105 GB/s vs roofline 800）=> 需重写长上下文 attention 内核。归档 `patches/0004`。
> - **AR 成本更正**：事件计时的在役值是 **53.0 us**（不是主机侧 68-95）=> 每轮 7.3 ms（§18.6）。
> - **修正后的每轮预算（§18.9，闭合）**：权重 12.1 + AR 7.3 + M8 增量 6.5 + M1 其他 5.5 + alloc 2.2 + **draft 13.6** + selector 2.7 + 未归因 6.7 = **56.6 ms**。
> - **需要用户批准的决策表**（收掉 2.3x 差距的主路径）：
>   | 项 | 预期 | 性质 |
>   |---|---|---|
>   | R2：AR 纳入图捕获 | -4.8 ms/轮 | 改 meta 后端切图边界 |
>   | mask/KV 宽度分桶 | prefill/TTFT -20%（64K alloc 8.0 s / 39.25 s） | 改 attention 读数范围，需重验 sha256/AL |
>   | draft 侧 13.6 ms/轮 | 最大单项 | 单调度器 + 形状交替 => 整轮单图/静态形状 |
>   | 长上下文 attention 内核重写 | 256K decode 34 -> ? | 新内核（1cat partition 路线） |
> - **状态**：本地 `llama.cpp` **干净 at `c2d716519`**；服务器 = canonical + 仅 A4 探针（默认关，已归档）；
>   全部实验补丁在 `1cat-vllm-v100-study/patches/0001..0006`；研究仓库最新提交 `4587a59`。
>
> **◎ Round 40-44 新增（细节见 SESSION §16.1-16.13）**
> - **模型是稠密的**（GGUF 实测 `feed_forward_length=17408`、**无 `expert_count`**）=> MoE 方向全部作废。
> - **roofline**：256K prefill 已到顶（28 PFLOP / 292 s）；**长上下文 decode 是洼地**（M=1 时 54 ms/token vs 5.7 ms 下限）；
>   target 步的权重流 9.02 GiB/卡 = 12.1 ms 已接近 roofline。
> - **账本闭合（决定优先级）**：普通解码实测 **24.9 ms/token** = 权重流 **12.1 ms** + 138 次 AR **13.1 ms**（误差 1%）
>   => **target 步里 40-50% 是 allreduce 延迟**，这是与 1cat（17.463 ms/轮）差距的主因。
> - **AR 的本质（源码核实）**：`ggml-backend-meta.cpp:2434-2462` 把图切成 **139 段**，每段对 3 个后端各调一次 graph_compute，
>   段间由**元后端主机侧调用** allreduce（`:2453`）=> **138 次 AR 全都不在 CUDA graph 内**，GPU 在主机往返期间空转。
>   NCCL 单次 95 us 里几乎全是主机入队 + 3 次设备切换（`ggml-cuda.cu:1000-1034`，我们命中"小张量 FP32"分支）。
>   => **微优化（去 memset/set_device、换 ring/P2P）只有 us 级收益**；必须走 **R1 设备侧 AR（无主机往返）** 或 **R2 把 AR 纳入图捕获（= goal P2）**。
> - **FA 派发（Volta 本形状）**：M=1 -> VEC（q8_0 直读）；**M=2..8 -> TILE，强制把整段 KV 反量化成 f16**（256K 约 26 GB/步，正是投机解码路径）；M>=9 -> MMA（只有 prefill 用到）。
>   VEC 只实例化 cols<=2、TILE 签名即 `V.h2` => "让 TILE 直读 q8_0"**不是有界改动**。
> - **上游 split-KV 存在但切分度不随上下文增长**（`parallel_blocks` 由 occupancy/wave 决定；上游 #28734 仍 open）；
>   **1cat 的 V100 答案** = `FLASH_ATTN_V100` partition split-KV（256/512/**1024**）+ fp8 KV 存储 + fp16 HMMA（NVFP4+marlin）+ D256 专用内核与 smem K ping-pong。
> - **prefill 差距是天花板差距**：我们 43.7 TFLOPS/卡 ≈ V100 dp4a 峰值（62.8 TOPS）的 70%；1cat 走 fp16 HMMA（峰值约 2x）=> 要拿必须换权重格式/自写 GEMM。
> - **新纪律**：服务器树**不是 git 仓库**（状态只能靠 md5 + 二进制标记串）；`git archive` 解包会刷新 mtime 触发全量 CUDA 重编（ccache **不覆盖** CUDA）；**编译用 `-j128`**（用户授权）。见 AGENTS §4.22/23。
> - **本轮在跑的任务与日志**（会话结束后仍在跑，接手先看这些文件）：
>   | 脚本 | 内容 | 日志 |
>   |---|---|---|
>   | `/root/lc-par.sh` | 长上下文**投机**标尺 32K/64K x {q8_0,f16}，卡 3/4/5 | `/tmp/lc-par.log`、`/tmp/lc-spec-*.txt` |
>   | `/root/lc-chain.sh` | TP4(q8_0/f16)、TP6(q8_0) @64K（等 PAR_DONE） | `/tmp/lc-chain.log` |
>   | `/root/fa-split-probe.sh` | 编译 `GGML_CUDA_FA_SPLIT_FLOOR` 探针并扫 floor={0,64,256,1024} | `/tmp/fa-split.log`、`/tmp/fs-build.log` |
>   | `/root/lc-sweep.sh` | 已停（保留 d8192=40.19 t/s 一点）；尾部臂让带宽给上面的投机标尺 | `/tmp/lc-sweep.log` |
> - **本地树干净**（`c2d716519`）；探针改动已存为 `patches/0004-fa-split-floor-probe.patch`（**服务器源码里仍有该改动**，可直接编）。
> - **长上下文深度曲线（§17.1.0，投机路径，q8_0，卡 3/4/5）**：每轮 **57.6（8K）-> 60.6（32K）-> 65.8 ms（64K）= +14%**；
>   prefill 2170（正式口径）-> 1502 -> 1352 t/s；**AL 32K->64K 持平（2.81 -> 2.79）=> AL 对深度不敏感**（此前"长上下文 AL 下降"是合成 prompt 的内容效应，已纠正）；
>   **真正快涨的是 host 侧图管理：alloc 2.44 -> 12.2 -> 15.1 ms/轮，rebuild 8% -> 48% -> 62%**（长上下文第二个抓手）；
>   128K 之后 attention/KV 通量才主导（256K 实测每轮约 146 ms）。
> - **32K KV dtype 对照（§17.1）**：每轮 **60.6 ms**（q8_0）/ **61.2 ms**（f16）vs 8K 的 57.6 ms
>   => **32K 只贵 5%（约 3 ms/轮）**；KV dtype **无实质差别**（prefill 1502 vs 1519 t/s）=> **继续用 q8_0**；
>   ⚠️ tg 从 96 掉到 46-50 的主因是 **AL（5.55 -> 2.81-3.06）**，但该 AL 受"合成重复源码 prompt"混淆，**未定论**（每轮 +5% 已定论）。
>   另发现 **32K 时约 50% 的轮会重建图**（8K 仅 8%）=> 长上下文下 host 侧图管理是第二个抓手。
> - **下一步（按证据排序）**：① 长上下文 KV dtype / TP 度数结论（数据在路上）=> 零代码配置建议；
>   ② 若 256K 仍差 => 按 `PORT-PLAN-sm70-longcontext.md` 走工作流 A（深度驱动切分，探针已在服务器上）；
>   ③ 短上下文主指标只有 AR 结构改动（R1/R2）能拿 13 ms => 需用户批准后立项。
>
> **◎ Round 40 新增（细节见 SESSION §16）**
> - **树/库一致性事故已修**：服务器源码树**不是 git 仓库**（`git status` 静默失败 => 被误判为"干净"），
>   且 `/root/libdir-instr` 当时是**带 push 实验码**的构建（`push_flag_stride` x7 + 二进制标记）。
>   已用本地 `git archive HEAD`（176 MB）权威同步服务器源码并重建；复核臂 **MEDIAN_TG = 99.76 t/s、AL 5.55/4.22/6.38、
>   greedy `f3edac19...`** => 记录数字有效、可复现（push 码默认不激活，不影响性能与数值）。
> - **模型是稠密的**（GGUF 实测 `feed_forward_length=17408`、**无 `expert_count`**）=> 一切 MoE / 专家小 GEMM 方向作废。
> - **roofline 定位**：256K prefill **已到 roofline**（attn 13.5 + GEMM 14 = 28 PFLOP，实测 292 s / 896 t/s）；
>   **长上下文 decode 才是洼地**（M=1 时 54 ms/token，roofline 5.7 ms => **9.5x**）；
>   target 步的 2.7x 余量主要被 138 次集合通信的串行延迟（~9.4-13 ms）吃掉。
> - **FA 在本形状（Volta, D=256, GQA=6）的派发**：**M=1 -> VEC**（q8_0 直读，无反量化）；
>   **M=2..8 -> TILE，强制 `need_f16_K/V` => 每次调用把整段 KV 反量化成 f16**（256K 约 **26 GB/步**，正是投机解码所在路径）；
>   M>=9 -> MMA（我们只有 prefill 会走到）。
> - **上游 split-KV 存在但不随上下文增长**（`parallel_blocks` 由 occupancy/wave 决定，V100+256K 仅 6-26 路；上游 issue #28734 仍 open）。
>   **1cat 的 V100 专项答案** = `FLASH_ATTN_V100` 的 partition split-KV（256/512/**1024**）+ **fp8 KV 存储（算前展开 fp16）** + fp16 HMMA（NVFP4 权重）。
> - 下一步判据：长上下文 **KV dtype x 深度曲线**（`/root/lc-sweep.sh`，正在跑）；"按 tile 反量化"内核改造属大改动，**先问用户**。
>
> **验收对账（同日实测，带 drop_caches）**
> | 项 | 项目最初 | 现在 |
> |---|---|---|
> | 主口径 tg（Q8_0 + DFlash2 n=7, ctx 8192, 3 prompt） | 55.95（起点基线）/ 95.20（上次正式） | **96.43 / 99.37**（两臂，AL 5.55，离散 3.0%，greedy 逐位一致） |
> | 每轮 ms | 58.9 | **57.6 / 55.9** |
> | 32K / 128K prefill | 1952 / 999 | **2170 / 1356**（+11.1% / +28.6%） |
> | **256K prefill / TTFT** | 370 / 672 s | **895.93 t/s / 293 s**（+142% / -56%） |
> | 256K decode | 33.97 | **34.13**（未改善） |
> | 相对最初 | - | **+72% ~ +78%** |
>
> **已落地并提交（llama.cpp，7 个 commit，未 push）**
> - FA 表 Q_in_reg=false（Volta D=256 全 ncols）：prefill 32K/128K/256K = +11.1%/+28.6%/+41.8%，解码不变，greedy 逐位一致
> - DFlash2 CPU selector：gate 2.78 -> 1.00 ms（4 路部分和），selector 4.42 -> 2.63 ms，tg +3.3%
> - 4 个 env-gated 诊断量具：GGML_CUDA_AR_TIMING / GGML_CUDA_GRAPH_DEBUG / LLAMA_SPEC_TIMING 扩展 / GGML_SCHED_SPLIT_TIMING
>
> **已实测证伪并回退（附补丁，勿重做）**
> - push 式 allreduce：跑得起来但结果错（**根因已确证**：NCCL 路径归约前会清零非 COMPUTE 分片，push 内核漏了 —— 上游 #23480 同类）+ 慢 2.4x（flags 同缓存行）。已补修复待验；存档 patches/0002
> - 按批大小分槽的图 arena：无效果（单 scheduler + 连续 arena 守卫），存档 patches/0003
> - 形状感知 CUDA graph 缓存键：tg 无变化、capture +14%，已回退
> - NCCL 调参（LL128 / 单通道 / Tree）全部更差；TILE->MMA 分发改动（Volta nb<=8 本就该走 TILE）
>
> **方向纠正（重要）**：draft 侧 13 ms 的主因**不是**主机侧图管理（实测 reuse=0/rebuild=269，但主机侧只占 7.7 ms 且与 GPU 重叠）
> ⇒ **draft 瓶颈在 GPU 侧**（约 1 GB 权重读取 + MoE + GDN 顺序算子，有效带宽仅约 70-100 GB/s）=> 下一步做**算子融合**。
>
> **下一步（按证据排序）**：① push AR 修复验证（补清零已做，判据 [AR] ar_us_avg < 68.4 µs 且 greedy 不变）
> ② draft GPU 侧算子融合（MoE/GDN，老清单 P6）③ target 侧 AR（在生产路径实测 NCCL 68.4 µs/次 x 138 次/轮 = 9.4 ms）
> ④ 256K decode（KV 带宽受限，q8_0 下每 token 每卡约 5.7 GB ≈ 6.3 ms）
>
> **纪律新增（AGENTS.md §4 第 18-21 条）**：A/B 的 md5 必须覆盖四个库 + 二进制标记校验；上传与编译不可并发；
> 单 scheduler + 形状交替是结构性限制；长等待跑测交子代理（用后台作业/哨兵，不 sleep 轮询）。


> ## ★ 2026-09-20 晚：DSH 接手后的进展（新增，优先于下文 Qwen 期内容）
> **权威顺序**：本段 > `AUDIT-2026-09-20-dsh.md`（E1–E16 纠正 + F1–F8 事实）> `1CAT-PORT-BACKLOG.md`（§4.5–§4.13 全部实测与方案）> 下文。
>
> **已落袋（实机实测）**
> - **KV dtype**：`f16` 胜 `q8_0` —— 32K prefill +3.2% / decode +4.0%；**64K prefill +9.1% / decode +5.3%**（llama-bench，3 卡 tensor + P2P + f16，`-r 2`）
> - **零代码**：**TP4 + `-ub 2048`** 在 32K prefill **1520 → 2190 t/s（+44%）**；decode 侧仍以 **TP3 最优（98.84 t/s）** ⇒ **分场景选卡**
> - **量具**：`splits=2`、`enqueue` 17.5 ms/轮、**`sync_us` 33.9 ms/轮**（target 是 GPU-bound）；**`[AR]` 138 次集合/轮 × 123 µs**
>   ⇒ 已装 ccache、`build-instr` 增量编译；`GGML_CUDA_OP_TIMING` 的 per-op event 表**不可用**（多设备给负数）
>
> **已证伪（勿重做）**
> - **P5 前提**：llama.cpp 的 `internal` allreduce 管道在 2 卡下比 NCCL/butterfly **慢 2.5×**（37.8 vs 14.8/14.4 ms enqueue）⇒ 「把 internal 推广到 N 卡」必然更慢；要做得做**设备 IPC 版**
> - **M5 候选 #1**：Volta 风格 staging（nbatch_fa 64 / combine 64 / nstages 1）在 D=256 上 **−6.7%**（2044 vs 2190）⇒ 已回退；改这张表 **~15 分钟/候选**（几十个模板实例重编）
> - 旧的「每轮 ~21 ms 未归因」「1cat 4-bit/2.9× 字节」「head_dim=128/40Q/8KV」**均已作废**（见 AUDIT）
>
> **下一步（按序）**：① M5 继续扫（只动 ncols/nthreads/occupancy，保持 K2=V2=128/combine=128/nstages=2；候选已列在 §4.13）
> ② P2 draft（13.8 ms vs 1cat 3.32 ms）③ P1 GQA 打包（需 f16 KV，已具备）④ P5 仅在有设备 IPC 证据后
>
> ## ★ 2026-09-20 深夜（DSH 第二轮实测，覆盖上面若干条）
> **新权威文件**：`1cat-vllm-v100-study/SESSION-2026-09-20-measurements.md`（本轮全部实测 + 路线图 + 复现命令）；在它覆盖的范围内优先于本段与下文。
>
> - **AR 已实测（取代上面的「123 µs 推算」）**：**138.0 次/轮、94.9 µs/次（无偏，关图 arm 采样不丢）=> 13.1 ms/轮**。NCCL 走 P2P/direct pointer（8 通道 ring）；`LL128` / `MAX_NCHANNELS=1` / `Tree` 三个旋钮**全部更差**（101/110/109 vs 默认 74-84）。NV2 链路裸 copy 164 KB = 5.1 µs（边际 47 GB/s）。
> - **CUDA graph 是刚需**：关图 target 步 39.9 -> **62.2 ms**、enqueue 20.4 -> **57.8 ms/轮**、tg 89 -> 63。
> - **target M=8 孤立前向 = 30.8 ms**（llama-bench pp8，TP3+f16KV+ub2048）；pp1 = 24.3 ms => **读权重固定成本约 24 ms（391 GB/s/卡 = 峰值 43%）**，边际仅 0.93 ms/token。
> - **draft = 每轮 2 次图计算**（注入 `common/speculative.cpp:1383` + 块前向 `:1440`，同一批流串行），**13.12 ms 且几乎不随块大小变化**（n_max=1 时 12.73 ms）=> **权重/固定开销受限**，理想 2-3 ms，**头寸约 10 ms/轮**。1cat 的注入是「一次投影」，我们的是「一次完整 draft 前向」。
> - **已落地并验证（保留）**：D=256 FA 表 `Q_in_reg=false` -> **32K prefill +11.1%**（1952.7 -> 2169.9 t/s）、8K +2.9%、解码不变、**greedy sha256 逐位一致**（`f3edac19…`）。
> - **M5 候选 #1 的量法更正**：M5 第一轮微基准**全部无效**（FA 设备实例在 `template-instances/fattn-mma-f16-instance-*.cu.o`，只重编 `fattn.cu.o` 时配置怎么改都一样）；主线那次端到端 −6.7%（2044 vs 2190，走完整 cmake 构建）**仍然有效**，与本次 `Q_in_reg=false` 是不同维度（后者 +53% 微基准 / +11.1% 端到端）。
> - **下一步**：① draft 侧（先确认 draft 图是否被 CUDA graph 捕获）② 自写 push 式 allreduce（微基准已达 59 µs，固定开销 57 µs 疑为 `__threadfence_system`）③ target M=8 前向本体效率 ④ FA 第二/三批（occupancy 3/4、nbatch_fa 64/128）

日期：2026-09-20（周日）
交接原因：用户要把执行体从 Qwen Code 迁移到 DeepSeek 专用 dsh。
本文件是**唯一的权威交接文档**；开始工作前先读本文件，再读 `FINAL-REPORT.md`。
持久规则在**工作区根** `F:\vllm+llama.cpp\QWEN.md`（每次会话自动加载，含红线与度量纪律），冲突时以 QWEN.md 为准。

---

## 0. 30 秒速览

- **北极星（不变）**：让 `llama.cpp` 在 V100 上**追平 1cat-vLLM 的速度与效果**。目标是改 `llama.cpp/`，`vllm/`、`1cat-vllm/` 只是**抄作业的参考项目**。
- **对标实测值**：1cat 生产服务 `vllm-1cat.service` 实测 **221.6 / 230.8 / 263.2 tok/s**，AL 4.06-5.21，**17.463 ms/轮**（单流，短上下文，DFlash2）。⇒ 差距约 2.3 倍（我们现在 95.20）。
- **我们现在**：固定口径单流 decode 中位数 **95.20 tok/s**（正式、带 drop_caches），**NODROP 最好 96.60**（TP3 0,1,2）。起点基线 **55.95** ⇒ **+70%**。
- **三项已落地且已验证正确性的提速**：① 并行化 CPU selector（+26.8%）；② `GGML_CUDA_P2P=1`（+10.6%）；③ **把 NCCL 编进来**（+18.4%）。
- **已排除的头号假设**：把 MMQ 的 dp4a 换成 Volta FP16 HMMA。实测**这条路的天花板只有 ~1.36×**，够不到 110（详见 §6）。**不要再从零重做这条**。
- **第一步已做完**：实测 **GDN + attention 合计只占每轮 3-6%**（ncu 在本机封死，改用 `test-backend-ops`；数据见 §10 待办 1）。
  ⚠️ **2026-09-20 DSH 审计更正**：该 3–6% 是用 `test-backend-ops` 的**形状不匹配**微基准得出的（hsk=256/512/64、nh=1/2、kv≤49152），
  而本模型实际是 **`head_dim=256` / 24 Q 头 / 4 KV 头 / 16 层全注意力（`full_attention_interval=4`）** ⇒ **短上下文结论或仍成立，但不能外推到 256K**。
  另：**"每轮 ~21 ms 未归因"是 NCCL 之前的旧数**，当前最好构建只剩 **~6–8 ms**（见 §6.2 新账本）。
- **当前服务状态**：`vllm-1cat` 与 `llmscope` 已 **stop**（我操作腾卡的）。恢复：`systemctl start vllm-1cat llmscope`。

---

## 1. 北极星目标与验收口径

**最终目标（用户 2026-09-20 明确，不随迭代改变）**：
> 让 `llama.cpp` 在 V100 上追平 1cat-vLLM 的速度与效果。

推论（务必记住）：
- **1cat-vLLM 的实测数字是天花板参考**；`llama.cpp` 自己跟自己的 A/B 只是过程指标，不是终点。
- **交付物必须落在 `llama.cpp/` 内**（用户原话："我们要开的是llama.cpp分支而不是vllm"）。
- **用户已放宽"禁止搬运代码"**：原话"1cat-vLLM 的 csrc/sm70_turbombind 884 tile 是参考思路，但如果有直接可用的代码 可以直接搬运到llama.cpp"（注意：仍需逐行能理解、能维护，这是 AGENTS.md 的硬要求）。

**固定基准口径（不动这个口径去抬数字）**：
- 模型 `Qwen3.8-27B-Q8_0.gguf`（29,047,086,048 B，sha256 `a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348`，ModelScope `unsloth/Qwen3.8-27B-GGUF`）
- draft `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`（官方基座 draft），`--spec-type draft-dflash --spec-draft-n-max 7`
- ctx 8192，官方采样，seed 42，`--split-mode tensor`，卡数**动态决定但必须记录**
- 度量脚本 `/root/p60-ab-harness.sh`；正式数字必须带 `drop_caches`

---

## 2. 环境（服务器 / 编译 / 凭证）

**AC922 开发测试服务器**（所有编译 / 测试 / bench 都在这里，不是本地 Windows）

- 连接：`ssh -o BatchMode=yes root@192.168.50.235`（本机已授权 key，非交互）
- 架构 **ppc64le（IBM POWER9，176 核）**，AlmaLinux 8.10，kernel 4.18.0-553
- **6× Tesla V100-SXM2-16GB**，driver 550.54.15，**CUDA 12.4**（`/usr/local/cuda-12.4/`）；无 Rust、无 docker
- **无盘网络启动**（`/` 是 NFS `192.168.50.84:/mnt/IBM-AC922/rootfs`）⇒ **下载 / 加载模型 / 编译吃同一张网卡，不要并行**（用户明确说"可以多等等"）
- V100 HBM2 被映射进系统内存（PPC64LE 特性）⇒ 页缓存会吃显存，测前要 `sync; echo 3 > /proc/sys/vm/drop_caches`
- 性能工具**不在 PATH**，必须全路径：`/usr/local/cuda-12.4/bin/ncu`、`/usr/local/cuda-12.4/bin/nsys`

**编译配方**（系统 gcc 8.5 会失败，必须 gcc-toolset-12）：
```sh
CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc \
CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ \
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
  -DCMAKE_INSTALL_RPATH=/root/llm/llama.cpp/lib64
cmake --build build --config Release -j82     # 空载时从 -j82 起步
```
**编 NCCL 版必须额外加**（`find_library(NAMES nccl)` 找不到，因为包里只有 `libnccl.so.2`、没有 `libnccl.so`）：
```sh
  -DNCCL_INCLUDE_DIR=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/include \
  -DNCCL_LIBRARY=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/lib/libnccl.so.2
```

**源码 / 构建位置**：
- 本项目源码+构建：`/root/llm/test/v100-opt/llama.cpp`（build 目录 `build/`、`build-nccl/`）
- ⚠️ `/root/llm/llama.cpp/bin/*` 与 `/root/llm/test/llama.cpp/build/bin/*` 都是**上游 `0.4.0-dev 434ddbb`，不是 b11053** —— 别拿它们当本项目 binary
- 模型：`/root/llm/models/`（`Qwen3.8-27B-GGUF/`、`Qwen3.8-27B-DFlash2-GGUF/`、`Qwen3.8-27B-TurboFCFusion-gguf/`）

**红线（只两条，其他全放开了）**：
1. **不要改** `/root/llm/systemd/llama-server.service`（原生产 unit，用户要求原样保留）
2. **不要改** `vllm/`、`1cat-vllm/`、`vllm-forkpoint/` 三个目录的内容（只读对照）

6 张 V100 全部归本项目，可自主 `systemctl stop/start`、kill、抢占、重编，无需逐次同意。
（`vllm-1cat.service` / `llmscope.service` 恢复命令 = `systemctl start vllm-1cat llmscope`；`new-api.service` 不占显存，一直在跑。）

**拓扑（`nvidia-smi topo -m`）**：GPU 0/1/2 在 NUMA0，两两 **NV2**；GPU 3/4/5 在 NUMA8；**跨组是 `SYS`**（走主机内存）。
只有 3 张卡能全直连 P2P；NCCL 实测：TP3 `isAllDirectP2p 1`，TP4 `isAllDirectP2p 0`。卡号分配是随手填的，0,1,2,3 与 0,1,3,4 本身没有区别。

> ⚠️ **2026-09-20 DSH 审计更正（用户明确指示）**：**卡数 1–6 自由；跨岛不是瓶颈，直接排除这条可能性。**
> 用户原话："（我怀疑之前的代码中会引导你往3卡上引导）实际上我们的虽然是跨岛显卡，但这不是瓶颈问题，这个你可以直接排除可能性。"
> §5 表里"TP3 最优 / TP4+ 变慢"是 **butterfly+NCCL 早期、固定短上下文标尺**下的实测，**只作历史记录，不作选卡依据**；
> 反证：用户的 1cat 生产服务就是**跨岛 TP4（0,1,3,4）**且跑到 221.6–263.2 tok/s。
> ⇒ 选卡按实验目的与实测决定；**4 卡时我们 Q8_0 = 7.25 GB/卡，优于 1cat 的 8.68 GB/卡**（字节口径见 §7.1 更正）。

---

## 3. 代码现状（已提交什么）

### 3.1 `F:\vllm+llama.cpp\llama.cpp`（本项目 fork）

- **HEAD = `79504e72b`**，`git status --porcelain` **为空（干净）**
- parent = `1af554f8f` = 上游 **b11053**（base 钉在 b11053，不跟 master）
- 相对 b11053 的改动：**13 个文件，+587 / −33**，内容分四类：

| 类别 | 文件 | 说明 |
|---|---|---|
| **C4** Volta MMVQ 参数 | `ggml/src/ggml-cuda/mmvq.cu` (+35/−1), `mmvq.cuh` (+4) | `MMVQ_PARAMETERS_VOLTA`：V100（CC 700）ncols=1 → nwarps=2；`MMVQ_VOLTA_MAX_BATCH_SIZE_K 4`。tg128 36.23 → 37.52 |
| **C5** K-quant 交叉点 | 同上（`ggml_cuda_should_use_mmvq`） | V100 的 mmvq↔mmq 交叉点 = **4**（上游独缺 Volta，落到默认 8）。ne11=8 +2.6% |
| **PR #27858** DFlash2 CPU selector | `common/speculative.cpp` (+386), `common/speculative.h` (+16), `common/common.cpp` (+6), `src/llama-ext.h`, `src/llama-model.cpp/.h`, `src/models/dflash.cpp` | 修掉 `tensor + draft-dflash` 的硬崩（`ggml-backend-meta.cpp:543`）；**`build_dflash2_selector_cpu()` 已按 8 个 block 位置并行（8 线程）** → 23.45 → 4.29-4.41 ms/轮，逐位一致 |
| **计时量具**（`LLAMA_ROUND_TIMING` / `LLAMA_SPEC_TIMING` / `[RT]` 探针） | `src/llama-context.h/.cpp`, `common/sampling.cpp`, `tools/server/server-context.cpp` | env-gated、未设时零开销 |

⚠️ **树里目前留有 7 处 `[RT]` `fprintf(stderr, ...)` 调试探针**（`llama-context.cpp` 4、`server-context.cpp` 2、`sampling.cpp` 1）。下一步可以清理掉（另开一个 commit）。

### 3.2 `F:\vllm+llama.cpp\1cat-vllm-v100-study`（研究档案，独立的 git 仓库）

- 本会话开始时 HEAD = `8a18a5b`；本次交接新增提交见 §3.3
- 这是**所有结论的存档**：`FINAL-REPORT.md`（总交付）、`goal-phase1-findings.md`（§1-§18，最全）、`premise-check-1cat-vs-llamacpp.md`、`phase0-round-breakdown.md`、`1cat-sm70-gemm-tactics.md`、`c5-volta-crossover.md`、`same-source-ab.md`、`dflash-in-llamacpp.md` 等

### 3.3 本次交接的提交

见本文档末尾 §12 的提交清单（`archive : ...`）。

---

## 4. 实测成果

### 4.1 总账

| 阶段 | 配置 | 中位数 t/s | 备注 |
|---|---|---|---|
| 起点基线 | 原版 b11053，TP tensor | **55.95** | `/root/goal-baseline.txt`（layer 70.09） |
| + 并行 selector | 同上 | 70.92 | +26.8%，逐位一致 |
| + `GGML_CUDA_P2P=1` | 同上 | ~78.4 | +10.6% |
| + **NCCL** | 同上 | **95.20** | **正式口径（带 drop_caches）**；NODROP 94.18 / **96.60**（TP3 0,1,2） |

**总计 55.95 → 95.20 = +70%**（对照最初的未改动树）。

### 4.2 三项已落地提速（都验证过正确性）

**① CPU selector 并行化**（`common/speculative.cpp::build_dflash2_selector_cpu()`）
- 这是 PR #27858 引入的、**只在 `--split-mode tensor` 下启用**的 CPU 侧 selector：每轮对 8 个位置做**全词表 top-k**（`n_vocab = 248,320`，padded 词表）+ gate 矩阵乘，rank=72，**单线程 POWER9 上 23.4-23.5 ms/轮**。
- 改法：按 8 个 block 位置分 8 线程并行，每位置的算术完全不动。⇒ **23.45 → 4.29-4.41 ms/轮**，tensor 中位数 **55.95 → 70.92（+26.8%）**。
- 正确性：**AL 完全相同**，`temperature=0` 输出 **sha256 逐位一致**。

**② `GGML_CUDA_P2P=1`**（+10.6%）
- `ggml-cuda.cu:391` 的 `GGML_CUDA_P2P` **默认未设置**，而它才是真正调 `cudaDeviceEnablePeerAccess` 的开关。

**③ 把 NCCL 编进 `libggml-cuda.so`**（**+18.4%**，单次最大收益）
- 根因：`build/CMakeCache.txt` 里 `GGML_CUDA_NCCL:BOOL=ON`（默认就是 ON）但 `NCCL_LIBRARY-NOTFOUND`、`ldd` 没有 libnccl ⇒ 初始化链 **nccl → internal → none**（`ggml-cuda.cu:1208-1245`）直接掉到最后一级 **butterfly（经主机内存中转）**。
- 而 `allreduce.cu:399-403` 的快速内部 allreduce 在 `n_devices != 2` 时**直接 return nullptr** ⇒ 3 卡以上全部走 butterfly 并**逐层**付费。
- **本机其实已经有 NCCL 2.29.7**（就在 1cat 的 venv 里，见 §2 的路径），**无需下载**。
- 结果：**78.20 → 94.18（NODROP）**、**80.42 → 95.20（正式，带 drop_caches）**。

### 4.3 数值验收

- `llama-perplexity` 门：**3.7745（butterfly）→ 3.7751（NCCL）= +0.016% ≤ 0.1%** 通过
- 两侧 launcher shell **逐字节相同**（md5 `5005b9d3...`）；语料 `/root/ppl-corpus.txt`（336180 B，md5 `6737ebfc032119d7daa00ae5046252c2`）
- ⚠️ **NCCL 会改数值**：`ggml-cuda.cu:1000-1072`，3 卡时 `ne < 131072` 走 FP32、`>= 131072` 压成 **BF16** 归约；M=8 的 FFN `17408x8 = 139264` 正好越过门槛。所以正确性验收走 **AL + ppl**，不是逐位。

---

## 5. 已排除清单（带数字，**不要重复**）

| # | 尝试 | 结果 | 结论 |
|---|---|---|---|
| 1 | NCCL 调参（buffer / channel / algo） | Ring **93.40**、LL **93.44**、1ch **77.67**、TP2 **80.38**，默认 **94.18** | **默认最好**，别再扫 NCCL 旋钮 |
| 2 | 加卡（NCCL，TP tensor，NODROP） | TP3 0,1,2 **96.60**；TP4 0,1,2,3 **67.54**；TP4 0,1,3,4 **72.11**；TP5 **73.09**；TP6 **58.26** | 当时结论"TP3 最优"。⚠️ **已被审计降级为"历史实测"**：那是 butterfly/NCCL 早期 + 固定短上下文标尺下的结果，**用户已明确卡数 1–6 自由、跨岛不是瓶颈** ⇒ **需要重扫**（Phase B4），不要据此把方案限死在 3 卡 |
| 3 | Volta 上把 Q8_0 MMVQ 强推 MMQ | **−31%** | 已回退。`mmvq.cu:655` 区域判断在 V100 上 **MMVQ 是对的** |
| 4 | `--spec-draft-device CUDA0\|CUDA1`（配 tensor） | **两臂都 abort** | 根因**非 bug**：`src/models/dflash.cpp:172-175` —— 我们的 DFlash2 draft 是**全词表、借用 target 的 head**（GGUF 里没有 `output.weight`），所以它**必须**跑在 target 用的设备上 |
| 5 | 多形状 decode graph 缓存 | 主机侧合计 **2.10 ms/轮 = 2%**，复用率已 93%（reuse=94 / rebuild=7） | **收益上限 ~2%，放弃**。（外部 fork 说的 "TP alloc 38 ms" 是他们的配置，我们这里是 1.94 ms） |
| 6 | CPU 采样走回 GPU / 别的采样路径 | sampling ~1 ms/轮 | 不值当 |
| 7 | `GGML_CUDA_FORCE_CUBLAS` | 它**不 gate** `should_use_mmvq`；且会给每层每轮加 ~178 MB 反量化写出 | 放弃 |
| 8 | 手写 m8n8k4 fragment 映射 | 只产出**块对角**结果（rows0-3 × cols0-3 有值，rows4-7 全零） | 原因是合成后的 `(asup,bsup)` 对只允许两个对角 4×4 块 ⇒ **改走 WMMA** |
| 9 | WMMA HMMA 原型 | v1 **8.8 GB/s** → v2（smem staging）**73** → v3（双缓冲流水）**89.6-90.2 GB/s** | 对比在任实现 **555 GB/s** ⇒ **慢 6.2×**。**这就是 §6 的结论来源** |
| 10 | **ncu / nsys profiling 真实验证轮** | ncu：`Backing up device memory in system memory` → `==ERROR== UnknownError` → `Failed to profile "scale_f32"` → `No kernels were profiled`（Q8_0 29 GB × 3 卡，save/restore 崩）。nsys：两次 `Importer error`，侥幸写出的 `.nsys-rep` **不含 CUDA kernel 数据** | **两条路在本机都封死** ⇒ 改用 `test-backend-ops`（数据见 §10 待办 1） |
| 11 | `LLAMA_LOG_INFO` 在 libllama 里埋点 | 连试 3 次**无输出**（原因未查明） | 先用 `fprintf(stderr,...)` 证明函数被执行，再切日志 |
| 12 | `llama-bench` 验证库内埋点 | **一条 libllama INFO 日志都不打印** | 用 `llama-server`（会打印 `slot print_timing:`） |

---

## 6. 根因分析现状 + HMMA 结论（**最重要的修正，别再走错**）

### 6.1 差距在"每轮延迟"，不在草稿质量

| 栈 | AL | ms/轮 | tok/s |
|---|---|---|---|
| 1cat-vLLM FP8 + DFlash2（4×V100 TP4） | 4.06-5.21 | **17.463** | **221.6-263.2** |
| llama.cpp Q8_0 + DFlash2 n=7（3×V100 TP3） | 4.11-6.08 | ~55 | 95.20 |

⇒ **我们的 AL 不输甚至更好**；**我们只是每轮多花 3 倍时间**。抓手 = 每轮前向+验证的效率。

### 6.2 每轮成本分解（⚠️ **下表是 NCCL 修复之前的旧账本**，92.6 ms/轮那次实测，AL 5.21，55.49 tok/s）

> ⚠️ **2026-09-20 DSH 审计更正（重要）**：下表及"~22 ms 其余"都来自 **NCCL 之前**的 92.6 ms/轮构建；
> 当前最好构建（NCCL + P2P，见 §4.1）是 **58.9 ms/轮**，请用下面的新账本。
> 另外三处量具/口径更正：① `enqueue_us` **不是 GPU 时间**——`llama-context.cpp:1436` 的第二个参数是 `batched`，
> 同步在 `:1376` 且条件是 `cparams.pipeline_parallel`（本 harness 永远传 `--tensor-split` ⇒ 恒为假），
> 所以它是**异步提交窗口**；② 因此"**M>1 时同步、可信**""layer 模式数字是假的"**均不成立**；
> ③ 字节口径见 §7.1 更正（1cat 是 FP8，8.68 GB/卡，不是 4-bit 3.4 GB/卡）。

**新账本（当前最好：NCCL TP3 + P2P，Q8_0 + DFlash2 n=7，58.9 ms/轮 = AL 5.55 ÷ 94.18 t/s，NODROP）**

| 成分 | ms/轮 | 占比 | 量具 |
|---|---|---|---|
| target 整步（`decode+sync`） | **32.3–34.3** | 55–58% | `[RT] target decode+sync`（`server-context.cpp:3709`） |
| ↳ 其中 `enqueue`（**异步提交窗口**，非 GPU 时间） | 21.8 | 37% | `[RT] perf: enqueue_us`（`llama-context.cpp`，来自 `ctx_tgt`） |
| ↳ 其中主机侧图工作（build+alloc+setin） | 2.3 | 4% | 同上 |
| ↳ 其余（同步等待/尾部） | ~9 | 15% | 差减 |
| draft 前向 | **13.8–14.4** | 24% | `LLAMA_SPEC_TIMING` 的 `draft_decode=` |
| CPU selector | 4.3 | 7% | `LLAMA_SPEC_TIMING` 的 `selector=` |
| walk | 0.1 | 0% | 同上 |
| **未归因余量** | **~6–8** | 11% | 差减（**不是旧文里的 ~21 ms**） |

**旧表（仅作历史，勿据以排优先级）**

| 成分 | ms/轮 | 占比 | 量具 |
|---|---|---|---|
| target verify（M=8） | **29.87** | 32% | `LLAMA_ROUND_TIMING` 的 `enqueue_us`（~~M>1 时同步，可信~~ 见上更正） |
| **CPU selector**（现已优化到 4.4） | 23.44 → 4.4 | 25% → 5% | `LLAMA_SPEC_TIMING` 的 `selector=` |
| draft 前向 | **15.01** | 16% | `LLAMA_SPEC_TIMING` 的 `draft_decode=` |
| target 主机侧图工作（build+alloc+setin） | 2.10 | 2% | build 0.13 + alloc 1.94 + setin 0.03 |
| 候选走查 + 其余 | ~22 | 24% | 余量 |

**draft 前向偏高**：1.14 GB / 15.01 ms ⇒ 有效带宽仅 **~76 GB/s** ⇒ 是 **dispatch / 小 batch 受限**，不是带宽受限。**待查**。

### 6.3 ★ HMMA 假设被实测否定（天花板只有 ~1.36×）

**背景（1cat 的做法是真的）**：`1cat-vllm/csrc/sm70_turbomind/`（vendored lmdeploy SM70 GEMM，"884" = **HMMA m8n8k4** tile + QPN ops）；其设计文档 `docs/design/sm70_awq_small_n_hmma_operator.md` 记录 M=5 形状从 11.3419 → 9.0439 ms（**−20.26%**），收益来自 `CTA_N=32/64` 注册项 + 无冲突 A 共享内存布局 `SmemLayoutV2<8,64,8,64,Swizzle<3,3,3>>`（`p' = p xor ((p & 0x1c0) >> 3)`），**逐位一致**。

**但是**：我们自己的实测（`test-backend-ops perf -o MUL_MAT -b CUDA0`，单卡，日志 `/tmp/mb-mulmat.log`）显示——

| 形状 | dtype | n=1 | n=8 | 结论 |
|---|---|---|---|---|
| m=4096 k=14336 | **q8_0** | 82.40 µs / **757 GB/s** | 112.48 µs / **555 GB/s** | n=8 已是**自身 n=1 屋顶的 73%** ⇒ 真实余量只有 **~1.36×** |
| m=14336 k=4096 | q4_K | 46.73 µs / 706-729 GB/s | 97.09 µs / **340 GB/s（48%）** | 这个才是"低效"的，但它**走 Ampere MMQ config**（两回事） |
| m=4096 k=14336 | f16 | 137.45 µs / 854 GB/s | — | — |
| m=4096 k=14336 | q4_0 | 45.34 / 729 | 91.96 / 359 | — |

**我此前"2.1× headroom"的说法是错的**——它来自 q4_K（n=8 走 Ampere MMQ config）。**我们固定口径的 q8_0 n=8 已经在自身屋顶的 73%**。
⇒ **HMMA 这条路最多 1.36×（55.95 × 1.36 ≈ 76），够不到 110。** 而且我的 WMMA 原型只有 90 GB/s（慢 6.2 倍）。

**M 曲线（3 卡、同 prompt/seed/512 tokens）**：`none`(M=1) **32.35** / `dflash n=3`(M=4) **67.2** / `dflash n=7`(M=8) **92.6 ms/轮**
⇒ 边际 **6.4-11.6 ms/行**（**不是 0**，所以 M=8 不是纯带宽受限）；纯带宽理想每行 32.35/8 ≈ **4.0 ms** ⇒ 实测约为带宽下限的 **2.9×**。
**这 2.9× 里"超出带宽下限"的那部分，才是真正可攻击的计算量。**

### 6.4 ★ 实测出来的 m8n8k4 fragment 布局（省得下一个人再逆一遍）

一条 `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32` **每线程**取 A=4 个 half、B=4 个 half、产出 D=8 个 f32。

- A 的 lane `L` 的寄存器 `{d0,d1,d4,d5}` ← A-lane `4*(L>>2)+(L&1)`；`{d2,d3,d6,d7}` ← 上面那个 `+2`
- B 的 lane `L`：`c0 = 2*((L&15)>>1)`；`{d0,d2}`←c0，`{d1,d3}`←c0+1，`{d4,d6}`←c0+16，`{d5,d7}`←c0+17
- 32 个 A-lane × 8 槽 = 256 = 全部 D 槽且互斥 ⇒ **D 是 8 行 × 4 份私有复制**
- ⚠️ **合成后的对是块对角的**（rows0-3 只与 cols0-3 乘；rows4-7 只与 cols4-7）—— 这就是手写映射只得到 32/64 个输出的原因

**WMMA 在 sm_70 可用，且能绕开所有 fragment 记账**：
```cpp
wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> a;
wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::col_major> b;
wmma::fragment<wmma::accumulator, 16,16,16, float> c;
wmma::load_matrix_sync(a, ptr, ldm);  wmma::mma_sync(c,a,b,c);
wmma::store_matrix_sync(out, c, ldm_out, wmma::mem_row_major);
```
对 `[n][k]`-major 的激活数组，`col_major` 配 `ldm = K` 才是正确配对（我一开始配对了 row-major 的 k-stride，得到 `rel err 1.375e10 → MISMATCH`；改成 `load_matrix_sync(b, B + k, K)` 后 16×16×16 探针 `0.000e+00 → MATCH`）。
Q8_0 GEMM 探针：128/128 单元填满，误差只在 f16 反量化层（8.25834 vs 8.25975）。

---

## 7. 1cat-vLLM 参考要点

### 7.1 它的真实运行配置（只读读到的，非 README 数字）

`vllm-1cat.service`：`CUDA_VISIBLE_DEVICES=0,1,3,4`（**跨 NUMA 组**）、`--tensor-parallel-size 4`、`--dtype half`、`--kv-cache-dtype fp8_e5m2`、`--gpu-memory-utilization 0.905`、`--max-model-len 262144`、`--max-num-seqs 1`、`--attention-backend FLASH_ATTN_V100`、speculative-config dflash。
⇒ `1cat-runtime.env` **没有任何 NCCL 调参**（只有 PATH / 缓存 / `VLLM_NCCL_SO_PATH` 指向同一个 libnccl.so.2）。
⇒ **它 4 卡能跑，不是因为卡数或 NCCL 调参**。**是硬约束逼出来的**（2026-09-20 DSH 审计更正）：
- 实测该 FP8 目录 **30,866,866,928 B**（66 个 safetensors）+ draft **3,848,817,896 B** = **34.72 GB** ⇒ **8.68 GB/卡**（4 卡）；
- `config.json`：`head_dim 256`、`num_attention_heads 24`、`num_key_value_heads 4` ⇒ vLLM 要求 TP 整除头数 ⇒ **TP ∈ {1,2,4}**；
- TP2 需要 17.4 GB/卡 > 16 GB ⇒ **只剩 TP4**。
⇒ 原文"FP8 权重 27 GB / 4 卡 ≈ 6.75 GB/卡"**数字偏低**（把 30.87 GB 写成了 27 GB，且漏了 draft）。
⇒ **对我们不构成约束**：llama.cpp 允许任意切分；**4 卡时我们 Q8_0 是 7.25 GB/卡，已优于他们的 8.68**。
⇒ 并且 1cat 的 TP4 是**跨岛**（0,1,3,4）——再次说明**跨岛不是瓶颈**。

### 7.2 它自己实测过的开关（`docs/design/sm70_qwen38_default_fastpath.md`，可直接借鉴）

| 开关 | C=1 | C=4 | C=8 | C=16 | 结论 |
|---|---|---|---|---|---|
| `VLLM_SM70_TP4_PUSH_ALLREDUCE_CONCURRENCY` | — | — | — | **+3.4%** | 默认 on |
| `VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES` | — | — | — | **+1.0%** | 默认 on |
| 两者一起 vs 都关 | +7.8%* | 0% | −0.9%* | **+4.3%** | 已合并 |
| `VLLM_SM70_USE_BREAKABLE_CUDAGRAPH` | **−29.2%** | **−17.6%** | **−18.8%** | **−12.7%** | **保持关闭** |

`*` 标注的 C=1 / C=8 因为 off/on 范围重叠，不可分辨；只有 C=16 可分离。

**它自己的自研 allreduce** 在 `csrc/custom_all_reduce.cuh`（push-based），两个开关用 `std::getenv` 在 **kernel launch 时**读；`vllm/envs.py` 里的声明当时**没被消费**（所以我们一度以为它没生效）—— **一类和我们 NCCL 问题同源的 bug**。

### 7.3 它的 V100-native 清单（`sm70_qwen38_default_fastpath.md`）

5 个模型级默认开关（只在 Qwen3.8 Flash-Next + NVFP4 + FP16 act/KV + native FP32 SSM + 全 SM70 本地 TP4/PP1 时默认开）：
- `VLLM_SM70_QWEN38_FP16_GEMV`
- `VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16`
- `VLLM_SM70_QWEN38_FUSED_HC_FP16`
- `VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP`
- `VLLM_SM70_MOE_ADD_ALLREDUCE`

依赖项：dual compilation、hybrid PLE（CPU/磁盘 offload）、Model Runner V2、full+piecewise graphs、native RMSNorm。
历史 ~98 tok/s 基线 = FP16 act/KV + native FP32 SSM + NVFP4 experts + TP4/PP1 + 262144 ctx + 8192 prefill chunk + **无 MTP / 无 prefix cache**。

### 7.4 ★ 分数校准（别被 README 数字骗）

- 另一个把 V100 当第一目标的 fork（`github.com/anyei/llamacpp-v100`）**投入大量工程后单流 MTP 只到 66 t/s**，其自己的结论是：*"batch-1 tg headroom on Volta is now mostly exhausted at the orchestration/launch-config level. Remaining gains require sm70 kernel-algorithm work (MMVQ small-batch, fattn-vec) — high effort, uncertain payoff."*
- ⇒ **没有人用 llama.cpp 在 V100 上达到过 1cat 的 200+**。1cat 的优势来自 **decode GEMM 走 Tensor Core（TurboMind HMMA 884）+ NVFP4/FP8 + 融合管线 + 高接受 draft**。
- 上游官方（PR #12098，CUDA 维护者）：*"the MMQ kernels should only be compiled for batch sizes up to MMQ_DP4A_MAX_BATCH_SIZE if FP16 tensor core hardware is available but **int8 tensor core hardware is not (basically only V100s)**."* ⇒ **V100 是唯一"有 FP16 TC、没有 int8 TC"的 N 卡**；`mmq.cuh:190` 的 MMA layout 要 `turing_mma_available(cc)`（≥750），V100(700) 落 dp4a；`mmq.cu:334` 是 `ne11 < 64` 才用 MMQ。

---

## 8. 关键文件 / 脚本 / 快照 / 日志清单

### 8.1 服务器（AC922 `/root/`）

**二进制快照**（同源 A/B 必须整目录 `LD_LIBRARY_PATH` 切换，**不能只复制可执行文件** —— 壳的 `DT_RUNPATH` 指向 build 目录绝对路径）：
- `/root/libdir-rt` —— baseline butterfly 构建
- `/root/libdir-nccl` —— **NCCL 构建**（`libggml-cuda` md5 `faf9cc468a0a263997e15c7f885c75b9`）← **当前最好**
- `/root/libdir-nccl-q8mmq`（已否定的变体）、`/root/libdir-bf-ppl`、`/root/libdir-27858`

**度量脚本**：
- `/root/p60-ab-harness.sh` —— **主力验收脚本**。参数：`CARDS` / `SPLIT` / `TAG` / `NPRED` / `PORT` / `L` / `M` / `D` / `NODROP` / `P2P` / `ARENV` / `DEVD` / `NGLD`；输出 lib md5（含 `libllama-common.so.0.4.1`）、每 prompt 的 tg+AL、`MEDIAN_TG`、`[RT] perf:`、spec timing、greedy sha256
- `/root/p60-ab-harness2.sh` —— 超集（加 `LDEXTRA` / `ARENV2`，打印 allreduce-init 行与 `[RT] target decode+sync`）
- `/root/goal-baseline.txt`（layer 70.09 / tensor 55.95）
- `/root/ppl-corpus.txt`（336180 B，md5 `6737ebfc032119d7daa00ae5046252c2`）

**微基准 / 探针**（`/root/`）：`hmma_probe`、`hmma_layout`、`hmma_map`、`wmma_bench`、`wmma_perf{,2,3}`；源码 `hmma_probe.cu`、`hmma_layout.cu`、`hmma_map.cu`、`hm_q8_bench.cu`、`hm_dbg.cu`、`wmma_q8_bench.cu`、`wmma_perf{,2,3}.cu`
**日志**：`/tmp/p60-*.log`、`/tmp/mb-mulmat.log`（521 行，MUL_MAT perf）、`/tmp/cards2.log`、`/tmp/hmma_layout.out`、`/tmp/ncu-tgt.log`（**未读**）、`/tmp/nccl-ab-summary.log`、`/tmp/final-nccl.log`
**脚本**：`nccl-check.sh`、`nccl-find.sh`、`nccl-pkg.sh`、`nccl-plan.sh`、`build-nccl{,-inner}.sh`、`nccl-ab.sh`、`nccl-env-sweep.sh`、`nccl-knobs2.sh`、`final-nccl.sh`、`ppl-ab.sh`、`ppl-bf3.sh`、`mk-corpus.sh`、`q8mmq-ab.sh`、`revert-q8.sh`、`devd-retest.sh`、`profile-tgt{,2}.sh`、`ncu-tgt.sh`、`cards2.sh`、`collect2.sh`、`final-evidence.sh`

### 8.2 本地（`F:\vllm+llama.cpp\1cat-vllm-v100-study\`）

`FINAL-REPORT.md`（**总交付报告，先读这个**）、`goal-phase1-findings.md`（§1-§18，最全细节）、`premise-check-1cat-vs-llamacpp.md` + `premise-check.html`、`phase0-round-breakdown.md`、`phase1a-report.html`、`stress-test-256k.md`、`l3-acceptance.md`、`c5-volta-crossover.md`、`same-source-ab.md`、`mtp-sampler-cpu.md`、`dflash-in-llamacpp.md`、`dflash2-llama-cpp-research.md`、`baseline.md`、`WORKPLAN.md`、`core-changes.md`、`FlashAttention-V100-reference.md`、`fa-v100-verified.md`、`1cat-sm70-gemm-tactics.md`、`vllm-vs-1cat-vllm-diff.md`、`COMMIT-MSG.txt`

### 8.3 1cat 参考文件（只读）

- `1cat-vllm/docs/design/sm70_awq_exact_m5_batched_gemv.md` —— M=5 形状（`5x17408x5120` 等），每 rank 每验证轮 **236 次**调用
- `1cat-vllm/docs/design/sm70_awq_small_n_hmma_operator.md` —— M=5 被接受的 HMMA 路线（**−20.26%，逐位一致**）与**被否定的备选**（N32 无 swizzle 1.1%、N32+swizzle 比 N64 慢、gate/up 强拆 2 有 1-ULP 差、N64 用在 MLP down 会回退）
- `1cat-vllm/docs/design/sm70_qwen38_default_fastpath.md` —— **§7.2/§7.3 的来源**
- `1cat-vllm/csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/{mma_sm70.h,operand_sm70_s884.h,smem_copy_sm70.h}`、`gemm/kernel/sm70_884_{4,8,16}.cu`、`gemm/mainloop_sm70.h`、`ops/awq_sm70_gemm.cu`
- `1cat-vllm/csrc/custom_all_reduce.cuh` —— 自研 push allreduce
- `1cat-vllm/scripts/serve_qwen38_27b_nvfp4_v100.sh`

---

## 9. 度量纪律（**必守，本会话踩过的坑**）

1. **记录两侧 binary 的 `--version`**；A/B 必须**同一个 build dir、同一套 CMake 参数**，唯一变量是待测代码。
2. **每个配置至少跑 2 次并报告离散度**（本机 256k prefill 同配置能漂 8%）。
3. **投机解码的 tg 是"含接受的吞吐"**，随接受率一起动 ⇒ 归因 kernel 必须**关 MTP**（`--spec-type none`）或**固定 seed 让两侧生成同样文本**；**报 t/s 必须同时报 AL 与每轮 ms**（`ms/轮 = AL ÷ tg`）。
4. **投机解码对比必须 ≥3 prompt + 固定 seed**（单 prompt 结论不可信：接受率随 prompt 摆动 0.25-0.40，排序会翻转）。
5. **A/B 有效性自检**：`md5sum` 两边 `libggml-cuda.so` **和 `libllama-common.so`**（spec 代码在后者里），相同即 A/B 无效。
6. **`build/bin/` 整套目录切换 + `LD_LIBRARY_PATH`**；只复制可执行文件完全无效。
7. `LLAMA_ROUND_TIMING` 的 `graph_compute` 只在 **M>1** 时同步 ⇒ M=8 可信，**layer 模式下的该数字是假的**。
8. `LLAMA_SPEC_TIMING` 必须从**每轮都执行**的函数 tick（`draft()`）；`process()` 只 prefill、`accept()` 会提前 return。
9. **`llama-bench` 不打印库内 INFO 日志** ⇒ 用 `llama-server`；libllama 里 `LLAMA_LOG_INFO` 可能无效 ⇒ 先用 `fprintf(stderr,...)`。
10. **`nsys` 必须优雅退出才写 `.nsys-rep`**（`kill -9` 丢报告）。
11. 诊断可 `NODROP=1` 跳过 drop_caches（加载 15 s vs ~270 s），但**对外数字必须带 `drop_caches`**。
12. **无盘 NFS**：下载 / 加载 / 编译共网卡 ⇒ **不要并行**。
13. 跨架构：本地 Windows 树与服务器树 **md5 不同是因为行尾**（`core.autocrlf=true`，`i/lf w/crlf`）⇒ 用 `git ls-files --eol` 判定，别慌。
14. **`ncu -o /tmp/X` 配脚本里的 `rm -rf /tmp/X.*` 会删掉自己的日志**（glob `X.*` 命中 `X.log`，也就是 stdout 重定向目标）⇒ 文件被 unlink，脚本退出后彻底消失。本会话 `ncu-tgt.log` "从未存在" 就是这个原因。排查 ncu 失败**先 `ls -la /tmp/X*`**，别以为脚本没跑。
15. **`pkill -f '<pattern>'` 会匹配到承载它的远程 shell 自己**（远程命令行里含该 pattern）⇒ 自杀、后续输出全丢、`LAUNCHED` 之类也不打印。用括号模式：`pkill -f 'foo[ ]bar'`。
16. **`test-backend-ops` 只在 `build-nccl/bin/`，不在 `build/bin/`**（同 `LLAMA_BUILD_TESTS=ON` 但只有前者产出了它）⇒ 别用 `build/bin` 的路径去调它。

---

## 10. 下一步任务（按序，可直接执行）

### 已完成（本会话）
- [x] PR #27858 移植（修 DFlash2 × tensor 硬崩，selector 正确性验证）
- [x] 前提核实：1cat 实测 221.6/230.8/263.2（用户数字真实）；差距在每轮延迟
- [x] 三项提速落地：并行 selector / `GGML_CUDA_P2P=1` / **NCCL**（+70%）
- [x] 大量否定：NCCL 调参 / 加卡 / Q8_0 强推 MMQ / `--spec-draft-device` / 多形状图缓存 / HMMA（天花板 1.36×）
- [x] 存档 + 提交（本文件）
- [x] 1cat 自己的实测开关表 + V100-native 清单已抄出来（§7.2/§7.3）
- [x] **Goal 8dbd30ea 第 1 项（GDN vs attention 占比）已量出**：ncu 路封死，改用 `test-backend-ops`（在 `build-nccl/bin/`）；**两者合计仅占每轮 3-6%**，下一步转向 draft 前向与那 ~21 ms 未归因（详见"待办 1"）

### 待办 1 — ✅ **已完成**（2026-09-20 交接时）：ncu 此路不通 → 改用 in-tree op 微基准，结论见 ④

**① ncu 在 AC922 上 profile 不了这个负载（已封死）**
`/root/ncu-tgt.sh` 发起 `ncu --launch-skip 4000 --launch-count 60` 打**真实 M=8 验证轮**。它的日志被脚本自己的 `rm -rf "$OUT".*` 删掉了（见 §9 第 14 条），但 `/tmp/ncu-tgt-server.log` 留下了真相：
```
==WARNING== Backing up device memory in system memory. Kernel replay might be slow.
==ERROR== UnknownError
==ERROR== Failed to profile "scale_f32" in process 1322774
==ERROR== No kernels were profiled.
```
⇒ **Q8_0 29 GB 权重 + 3 卡，ncu 的 device-memory save/restore 撑不住 ⇒ ncu 这条 profiling 路在本机封死**（nsys 同样拿不到 CUDA kernel 数据）。
⇒ 要 kernel 级证据，只剩**在 `ggml_backend_cuda_graph_compute` 里加 env-gated 的按 op name 聚合计时**（需 `GGML_CUDA_DISABLE_GRAPHS=1`，因为 capture 的图里每节点 event 无效）。

**② 可用的替代（路径已验证，不是猜的）**
⚠️ `test-backend-ops` **不在 `build/bin/`，在 `build-nccl/bin/`**（两个 build 的 CMakeCache 都是 `LLAMA_BUILD_TESTS:BOOL=ON`，但只有 `build-nccl` 产出了它）。
```sh
cd /root/llm/test/v100-opt/llama.cpp
LD_LIBRARY_PATH=/root/libdir-nccl CUDA_VISIBLE_DEVICES=0 \
  ./build-nccl/bin/test-backend-ops perf -o GATED_DELTA_NET -b CUDA0
```
脚本 `opbench.sh`（本会话跑过）；原始输出已存档：`opbench-gdn-raw.txt`、`opbench-fa-raw.txt`、`opbench-mm-raw.txt`。

**③ 实测数据（单卡、隔离 op；**不是**真实图，口径必须标注）**

**GDN（gated delta net）**，`head_count=32, head_size=128`（= 本模型的 DeltaNet 形状）：

| n_seq_tokens | us/run | 带宽 |
|---|---|---|
| **1（= decode 形状）** | **5.50** | **721.76 GB/s**（贴近屋顶） |
| 64 | 99.21 | 78.91 GB/s |
| 256 | 393.02 | 49.87 GB/s |
| 512 | 790.70 | 44.66 GB/s |
| 1024 | 1577.06 | 42.34 GB/s |

⇒ **GDN 边际成本 ≈ 1.53-1.55 us/token**（64/256/512/1024 四点一致）⇒ **M=8 约 5.5 + 7×1.53 ≈ 16 us/层**。
（注意：GDN 在**大 n（prefill）掉到 42-50 GB/s** —— 那是 prefill 侧的低效点，与本轮 decode 目标不同。）

**FlashAttention（q8_0 K/V）**：

| kv | us/run |
|---|---|
| 128 | 35.79 |
| **512** | **25.68** |
| **1024** | **25.55** |
| 2048 | 37.69 |
| 4096 | 60.70 |
| 7680（hsk=64,nh=8） | 216.33 |
| 10000 | 203.11 |
| 20000 | 392.88 |

**④ ★ 结论（与层数无关 ⇒ 现在就成立）**
- 我们固定口径是**短上下文**（实际 prompt ~76-590 token）⇒ 单层 FA ≈ **26 us**、单层 GDN ≈ **16 us**。
- 顺带核实了模型结构：**65 个 block**（0..63 主体 + **blk.64 = MTP/NextN 块**；用外部 DFlash2 draft 时被忽略 —— 证据：加载日志里 `blk.64.nextn.*` 全部 `unused tensor ... ignoring`）；由 `ffn_gate.weight = 94699520 B` 反推 **n_embd = 5120 / ffn = 17408**（Q8_0 下 94699520×32/34 = 17408×5120），**与 1cat 的 `5x17408x5120` 形状完全对上**。
- **即使把 64 层全算成 FA：64 × 26 us = 1.66 ms；全算 GDN：64 × 16 us = 1.02 ms。** 而 target 的 M=8 前向是 **29.87-32 ms/轮**（`[RT] target decode+sync` 实测 32.4 ms @TP3，某 NCCL TP3 臂日志里 47.56 ms）。
- ⇒ **GDN + attention 合计只占 ~1-2 ms / ~30 ms（约 3-6%）。它们不是 3× 差距之所在。**
- ⇒ **Goal 8dbd30ea 的候选 ①（`FUSED_GDN_INPUT_FP16` / `FUSED_HC_FP16`）优先级下调**；attention 只在**长上下文**变大（kv=7680 时 64 层 ≈ 13.8 ms —— 那才是 FA 该优化处，与 §0.13-D3「长上下文 decode 是 FA 计算受限」一致）。

**⑤ ⇒ 钱在另外三处（按头寸排序）**
1. **draft 前向（最大待定项）**：`draft_decode` ≈ **14.2-15.2 ms/轮**，而 draft 权重只有 **1.14 GB**。
   - 若**一轮一次**前向 ⇒ 有效带宽仅 **~76-80 GB/s**（dispatch 受限，**头寸 ~8-12×**）
   - 若**一轮 8 次**前向（8×1.14 GB = 9.1 GB）⇒ ≈607 GB/s（67% 屋顶，头寸只有 ~1.5×）
   - **这两种解释差 8 倍，必须先定论**。做法：读 `common/speculative.cpp` 的 block-draft 路径一轮做几次 draft 前向；或跑 `--spec-draft-n-max 1/3/7` 看 `draft_decode` 是否随 `n_max` 线性变化（**线性 ⇒ 每 token 一次；不变 ⇒ 一次块前向**）。
2. ~~**我们每轮还有 ~21 ms 未归因**~~：**该数字已过期**（2026-09-20 DSH 审计）。它是 NCCL 之前 73 ms/轮的残差；
   当前最好构建（NCCL+P2P，58.9 ms/轮）的新账本是：target 整步 32.3–34.3 + draft 13.8–14.4 + selector 4.3 + walk 0.1 ⇒ **余量仅 ~6–8 ms**（§6.2）。
   **当前最大单一未知量已变成"target 整步里的 21.8 ms 异步提交窗口 + ~9 ms 同步/尾部"从何而来。**
3. **量化 matmul**：q8_0 n=8 已 555 GB/s（自身 n=1 屋顶 757 GB/s 的 73%）⇒ **上限 1.36×，已封顶**。只在 n≈5-12 这段还有故事（1cat 的 m5/small-N 算子正对此），但那最多 1.36×。

### 待办 2：落地下一步（**顺序已按待办 1 的结论重排**）
| 序 | 动作 | 理由 / 头寸 |
|---|---|---|
| **2a** | **定论 "draft 前向一轮几次"**：读 `common/speculative.cpp` 的 block-draft 路径，或跑 `--spec-draft-n-max 1/3/7` 看 `draft_decode` 是否随 `n_max` 线性 | 头寸 **8-12× vs 1.5×**，当前最大待定项 |
| **2b** | **攻 target 整步的 21.8 ms 提交窗口 + ~9 ms 尾部**（加 per-op 聚合计时或更细分段计时；注意 `GGML_CUDA_DISABLE_GRAPHS=1` 才好在图内埋点）。⚠️ 旧文案的"~21 ms 未归因"已过期，见 §6.2 新账本 | 约每轮 52%，是当前最大未知量 |
| 2c | 量化 matmul 的 **M≈5-12** 段（1cat 的 m5 / small-N HMMA 算子思路可借鉴） | **上限 1.36×，已封顶** |
| 2d | `FUSED_GDN_INPUT_FP16` / `FUSED_HC_FP16`（1cat `VLLM_SM70_QWEN38_*`） | **优先级下调**（GDN 只占 3-6%） |
| 2e | push-based allreduce（1cat `csrc/custom_all_reduce.cuh`；其 C=1 +7.8% / C=16 +4.3%），与所有现成通道都不同 | 先守住 NCCL 基线；**不要**引入 `VLLM_SM70_USE_BREAKABLE_CUDAGRAPH`（−13% ~ −29%） |
| 2f | 长上下文口径的 FA（kv≥4096 时 64 层 FA ≈ 6-14 ms，才是 FA 该优化处） | **当前短上下文口径不适用**，换口径才做 |

**明确禁止**：`VLLM_SM70_USE_BREAKABLE_CUDAGRAPH`（1cat 实测 −13% ~ −29%）。

### 待办 3：验收（Goal 8dbd30ea 的 Done when，逐条贴证据）
1. 贴 **GDN vs attention 成本占比**实测证据行
2. `/root/p60-ab-harness.sh` **带 drop_caches** 重跑，贴 `MEDIAN_TG=`（目标 **≥110 且 >95.20**）与所选**卡数 / split-mode**
3. 贴 `[RT] target decode+sync` 行
4. 数值验收：两臂 `llama-perplexity` 的 `Final estimate`（能逐位一致就改贴 greedy sha256，否则 ppl 相对变化 **≤0.1%**）
5. 贴 `git status --porcelain`，**改动只落在 `llama.cpp/` 内**

### 待办 4（清理）
树里 7 处 `[RT]` `fprintf` 调试探针可清掉（另开 commit）。

---

## 11. 红线 / 服务状态 / 恢复

**红线**：
- 绝不 `git push` / 代开 PR / 代写 upstream 提交描述（本项目是私有 fork，不做上游贡献）
- 不修改 `/root/llm/systemd/llama-server.service`
- 不修改 `vllm/`、`1cat-vllm/`、`vllm-forkpoint/` 内容
- `llama.cpp/` 源码只用 **ASCII**（禁 `—` `→` `×` `…`，用 `-` `->` `x` `...`）；不新增 `tests/*` 文件；大改动先问用户

**服务状态（2026-09-20 交接时）**：`vllm-1cat.service`、`llmscope.service` 已 **stop**（腾卡）；`new-api.service` 仍在跑（不占显存）。
**恢复命令**：`systemctl start vllm-1cat llmscope`

**要跑 1cat 对标**：`bash /root/p32-vllm1cat-bench.sh`（启动服务 + 3 请求 + 采 journal）→ 日志 `/tmp/vllm1cat-bench.log`

---

## 12. 立刻继续（复制粘贴区）

```sh
# 0) 看现状
ssh -o BatchMode=yes root@192.168.50.235 'nvidia-smi --query-gpu=index,memory.used --format=csv; systemctl is-active vllm-1cat llmscope'

# 1) ncu 路已确认封死（待办 1）；真要看真相看 server 日志，别看 .log（被脚本自己删了）
ssh -o BatchMode=yes root@192.168.50.235 'grep -aE "ERROR|Failed to profile|No kernels|Backing up device" /tmp/ncu-tgt-server.log'

# 2) op 级微基准（路径已验证：在 build-nccl/bin/，库用 /root/libdir-nccl）
ssh -o BatchMode=yes root@192.168.50.235 'cd /root/llm/test/v100-opt/llama.cpp && LD_LIBRARY_PATH=/root/libdir-nccl CUDA_VISIBLE_DEVICES=0 ./build-nccl/bin/test-backend-ops perf -o GATED_DELTA_NET -b CUDA0'

# 2b) 全量 opbench（GDN + FLASH_ATTN_EXT + MUL_MAT，约 10 min；脚本已存档 opbench.sh）
ssh -o BatchMode=yes root@192.168.50.235 'nohup bash /root/opbench.sh > /tmp/opbench.log 2>&1 < /dev/null & echo LAUNCHED'

# 2c) 待办 2a：用 harness 的 NPRED（= --spec-draft-n-max）分别跑 1 / 3 / 7，
#     看日志里 "draft: spec timing: ... draft_decode=" 是否随 NPRED 线性变化
#     （线性 ⇒ 每 token 一次前向；不变 ⇒ 一轮一次块前向）

# 3) 基线复现（正式口径，带 drop_caches）
# ⚠️ 审计更正：必须带 P2P=1（记录里的 94.18/95.20 都是带 P2P 跑的；harness 默认不开 P2P，也不默认用 nccl 库）
ssh -o BatchMode=yes root@192.168.50.235 'CARDS=0,1,2 SPLIT=tensor TAG=handoff-base L=/root/libdir-nccl P2P=1 bash /root/p60-ab-harness.sh'
# 卡数不设限（用户 2026-09-20 明确：1–6 卡自由，跨岛不是瓶颈），例如 4 卡：
#   CARDS=0,1,2,3 ... （4 卡时我们 Q8_0 为 7.25 GB/卡，优于 1cat 的 8.68 GB/卡）
```

**本地仓库位置**：`F:\vllm+llama.cpp\llama.cpp`（fork，HEAD `79504e72b`）、`F:\vllm+llama.cpp\1cat-vllm-v100-study`（档案）
**权威规则**：`F:\vllm+llama.cpp\QWEN.md`
