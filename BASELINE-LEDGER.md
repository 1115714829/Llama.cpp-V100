# BASELINE-LEDGER.md — 阶段成果账（对比基线）

> **规矩（用户 2026-09-21 明确）**：每拿到一个**阶段成果**，就把它**记进本文件并作为下一轮扩展的对比初值**；
> 后续所有实验都以**最近一次采用的阶段成果**为基准做同源 A/B；**比基准差的阶段成果一律不采用**（就地标注为灰色否证）。
> 报数纪律：**tg 与 ms/轮 必须并列**（`ms/轮 = AL ÷ tg × 1000`）；每配置 >=2 臂并报离散；greedy sha256 门必须一致。

## 采用链（从旧到新，每一行都是当轮的"初值"）

| # | 配置 | tg p1/p2/p3 | AL p1/p2/p3 | **ms/轮 p1/p2/p3（均值）** | 相对上一阶段 | 判定 |
|---|---|---|---|---|---|---|
| B0 | 上游 b11053 原版, TP tensor | 55.95（单值） | - | - | 起点 | 起点 |
| B1 | + 9 个自建 commit（NCCL TP3 + P2P + selector 并行 + V100 调优） | 99.83 / 80.59 / 115.78 | 5.55 / 4.22 / 6.38 | 55.60 / 52.36 / 55.10（**54.35**） | +78% | 采用（旧基准） |
| **B2** | **B1 + E15 FGC（`GGML_META_FULLGRAPH=1`，meta 整调用单图捕获）** | **100.95 / 88.30 / 129.12** | 5.55 / 4.22 / 6.38 | **54.98 / 47.79 / 49.41（50.73）** | **均值 -6.5%；p2 -8.4%，p3 -10.1%** | 采用（B3 之前） |
| **B3** | **B2 + CPU selector 向量化（R256 起为默认行为，无需 env；`GGML_SPEC_SELECTOR_VEC=0` 可回退）** | **102.09 / 89.22 / 130.72** | 5.55 / 4.22 / 6.38 | **54.36 / 47.44 / 48.48（50.22）** | **均值 -1.0%** | 采用（B4 之前） |
| **B4** | **B3 + T8 多槽 gallocr（R273/R274；**默认 ON**，`GGML_GALLOCR_SLOTS=0` 回退；库 `/root/libdir-t8b`）** | **107.24 / 95.88 / 141.19** | 5.55 / 4.22 / 6.38 | **51.76 / 44.01 / 45.19（46.99）** | **均值 -6.5%（-3.29 ms/轮）** | 采用（B5 之前） |
| **B5** | **B4 + FGC 关闭（R276 新鲜度反转；完全不设 `GGML_META_FULLGRAPH`）** | **113.28 / 94.32 / 145.65** | 5.55 / 4.22 / 6.38 | **49.18 / 44.99 / 43.63（45.94）** | **均值 -2.1%（-1.00 ms/轮）** | **采用（当前基准）** |

### R298-P ★★ **enqueue 税解剖（P6/P10 量具，256K 单请求）：巨兽不是 meta（7 ms/call），是 sched 每调用图规划（263.7 ms/call = 墙钟 69%）**（2026-09-24，前哨）

- **账（r=1.08，rep=1，505 调用）**：`enqueue 133.1 s = 263.7 ms/call`（墙钟 69%）｜`alloc 28.5 ms/call`（T8 已治）｜`setin 22.5 ms/call`｜`build 0.25 ms/call`。
- **[META] 对照**：`total=7.049 ms/call`、sub/call=15.8、ar/call=14.8、dev_hist 中位 25µs、85%<60µs ⇒ **meta 派发只是零头**；老 E 系列"meta 税"叙事在 256K 服务负载下让位于 **ggml_backend_sched 每调用图规划**（split/指派 O(大图)，prefill 图 5000+ 节点）。
- ⇒ **桶化已让图复现（rebuild 66）但 sched 规划仍逐调用重做** —— r=1.08 收益饱和的真因。
- ⇒ **下一刀 R298 = sched 规划缓存/免重做**（按图指纹缓存 split/assign 结果，复用轮直达 submit）。**天花板测算**：enqueue 264→~30 ms/call ⇒ TTFT 192→**~100-115 s（spec-off）⇒ 反超 BL1（152.5 s）在射程内**。
- 口径注记：[SCHED] P10 探针无输出（打印条件待查，不影响主账）；[META] calls 口径（4224）与 rounds（505）的映射未完全厘清，7 ms/call 下界结论不受影响。

### R297 ★★★ **桶化 r=1.08 采用（P-GRAPHTAX 收口）：256K 预填充 +38.4%（289.2→208.9 s），距 BL1 1.90x→1.37x；纯步 -0.3% 不回退**（2026-09-23/24）

- **采用值（env `GGML_KV_BUCKET_RATIO=1.08`，256 对齐桶）**：

| 口径 | BL3 基线 | **r=1.08 采用值** | 增幅 | 判据 | 判 |
|---|---|---|---|---|---|
| 256K TTFT（spec-on） | 289.2 s / 817 t/s | **208.9 s / 1131 t/s** | **+38.4%** | ≥+30% | ✅ |
| 256K TTFT（spec-off） | 269.6 s / 877 | **192.3 s / 1229** | +40.2% | — | ✅ |
| **纯步（spec-off 低噪声锚）** | 34.5 ms/token | **34.4 ms/token（-0.3%）** | 平 | 不回退 >2% | ✅ |
| 门值（四臂矩阵） | — | 全绿（f3edac19/69207026 逐字，含投机） | — | 原样 | ✅ |
| 显存 | 11.3 GB/卡 | 平 | — | 不升 | ✅ |
| rebuild/请求（机制自证） | 468 | **65（-86%）**，alloc 198→28 ms/call | | ≤30 | ⚠ 判据口径缺陷：29 桶估算基于 r=1.25 且漏多 n_tokens 形态；**目的（机制自证）已达成**，缺陷入档、判据不改 |

- **曲线三点（单变量 r）**：r=1.0 基线 / **r=1.08 = +38.4%（采用）** / r=1.25 = +35.1%（被 r=1.08 支配：padding/填充税 27s > rebuild 节省 14s）。**收益在 r≈1.08 饱和**——剩余 host 大头 = **enqueue 253 ms/调用的非形状税**（下一刀 = P-D1/Fix B 领地）。
- **tg 判据教训入档**：spec-on 的 tg 受生成内容 AL 波动污染（±25%），**"不回退"必须用 spec-off 纯步锚**判（本次即以 34.5 vs 34.4 ms 定案）。
- **距 BL1（256K 同尺）**：预填充 **1.90x → 1.37x**（152.5 vs 208.9）；spec-off 对 spec-off **1.21x**（159.0 vs 192.3）。吐字主刀仍为轮结构/纯步（1.44x 步差 + 2.65x 轮放大，见 R293/R294）。
- 双 rep 离散：TTFT 0.1-0.2%、纯步 0.1%。流程事故注记：一条远程命令内 pkill 自匹配再演（括号式被同命令裸名破功）→ 杀/启已分令执行（RULES 家族教训第 6 次）。

### R296b ★★★ **桶化 v3（256 对齐）四臂门全绿 = 全路径位级等价（含 DFlash2）；性能验收 r=1.25 点：TTFT +35% 达标、tg -11.9% 超标 ⇒ 按预登记不采用，转权衡曲线**（2026-09-23）

- **根因定案（破案链）**：v1/v2 崩溃 = **FA 内核实例化形状轨道**（`fattn-mma-f16.cuh:1824 flash_attn_ext_f16 has no device code for arch 700`，CUDA_LAUNCH_BLOCKING 点名）——llama 自带 256 padding 正是护轨，1.25 阶梯（298/373…）脱轨即崩（N8T/#28761 NO_DEVICE_CODE 家族）。compute-sanitizer 报告为假线索（NCCL P2P 不兼容，已排除）。
- **v3 = 256 对齐桶**：四臂门矩阵全绿——gb-on **`f3edac19…` 逐字复现**（含投机）、gb-on-nospec **`69207026…` 逐字复现** ⇒ **全路径位级等价 [S2] 完全成立**；v2 底层桶化保留为鲁棒化。
- **性能验收（[S3]，BL3 vs GBON 双 rep 离散 ≤0.5%）**：

| 指标 | BL3 | GBON(r=1.25) | 判据 | 判 |
|---|---|---|---|---|
| TTFT / 预填充 | 289.2 s / 817 | **214.0 s / 1104（+35.1%）** | ≥+30% | ✅ |
| 门值/显存 | — | 全绿 / 平 | 原样/不升 | ✅ |
| rebuild/请求 | 468 | **33（-93%）**（reps=2 计 66/1021 rounds） | ≤30 | ❌ 边缘 |
| tg | 32.7 | **28.8（-11.9%）**（33.7→35.6 ms/token） | 不回退 >2% | ❌ |

- ⇒ **按 [S3] 预登记：不采用（记档）**。机制归因：桶 padding 使每轮 FA 扫描 + mask 尾填膨胀 ~12% 直接打在 verify/draft 轮（tg -12% 与膨胀率吻合）。
- ⇒ **权衡曲线（单变量 r）**：r=1.25 点 = {TTFT +35%, tg -12%, rebuild -93%}；下一测点 r=1.08（桶数 29→78/请求，膨胀 12%→4%），外推交集可能在 r≈1.05-1.08；v4 候选 = 按 n_tokens 分段桶（prefill 粗桶 / decode 细桶）或 r 单点定夺。
- **流程事故双记**：① 构建与门臂同批并发发 ⇒ 构建被 BUSY_SERVER 拦、门臂跑旧库（RULES③ 第五次显灵，已串行重做）；② sanitizer/gate 孤儿 server 进程残留致构建互斥误触发（清理后干净）。

### R296 ★★★ **P-GRAPHTAX 形状桶化（GGML_KV_BUCKET_RATIO）v1：target 路径位级等价已证；draft 路径交互病灶待修**（2026-09-23）

- **实现**：`llama-kv-cache.h` 桶函数（几何比值 env，默认关）+ `llama-graph.cpp` mask 定尺/复现判据 + `llama-kv-cache.cpp` K/V 视图桶化 + mask 尾部 -inf。分支 `feat/graph-shape-bucket`（bb6d4a788）。构建 BUILD_RC=0 / 标记串三查绿（MARK_LIB_BUCKET=1）。
- **四臂门值矩阵**（p60 固定尺 harness，NPRED=512）：

| 臂 | sha256 | 判 |
|---|---|---|
| gb-off（无桶+投机） | `f3edac19…` | 门 ✓（B5 门值复现） |
| gb-off-nospec | `69207026…`（len 160） | 无投机谱系基准 |
| gb-on（桶+投机） | **CUDA `unspecified launch failure` @`common_speculative_impl_draft_dflash::draft`** | draft 交互病灶 |
| **gb-on-nospec（桶+无投机）** | **`69207026…` 逐字同对照** | **位级等价 ✓✓** |

- ⇒ **[S2] 数值论证成立**（target 路径 padding 全 mask 位级不变）；投机/无投机两谱系差 = DFlash2 概率采样改输出分布（len 158 vs 160），各自自洽。
- ⇒ **病灶锁定 draft 路径 × 桶化交互**（异步 CUDA fault 归因在 draft()；机制候选：dflash 注入图按真实 n_kv 定形的 gather/concat/k_idxs 与桶化视图错位——下一步读 `src/models/dflash.cpp` 注入图定修法）。
- 复盘注记：途中三次工具链笔误（判别臂参数未接线 / 调用点拆词 ×2）已修，均未污染数据；gb-off 门值三连复现稳定。

### R295 ★★ **ub 口径对齐臂（BL3UB2K）：host 税归因首战验证——调用数 462→116 = 预填充 +27%，但吐字 -20%**（2026-09-23）

- **臂**：BL3 同栈（t8b/TP4/spec on）仅 `--ubatch-size 2048`（对齐 vLLM 标准 chunked-prefill 2048 基本参数）+ `SLOTS=1`（SLOTS=3 在 ub2048×深 KV 下 OOM：**1715 MiB = [n_kv×2048] f32 mask**，R282/m1 同族）。双 rep 离散 0.1%。
- **结果**：预填充 **817 → 1038.5 t/s**（227.5 s，**+27%**，R223 预登记的 +27.8% 复现 ✓）；吐字 **32.7 → 26.2 t/s**（30.5 → 38.1 ms/token，**-20%**，大 ub 的 mask/图搬运压 decode 轮——R222"-2.2% 解码换预填充"的服务版放大）；AL 0.27/0.28 持平。
- ⇒ **R294 host 税归因成立**：host 调用 462→116（4x）直接兑现预填充 +27%；**距 BL1 预填充 1.90x → 1.49x**。
- ⇒ **ub 是部署层双刃剑**；根治仍须**形状稳定化**（n_kv 桶量化/mask 定尺——使 ub512 也吃调用数红利且不伤 decode；mask 定尺顺带解 1.7 GB OOM = P-M 白捡）。
- 口径注记：此臂为**口径对齐测量**（非成果主张，UBLEVER 纪律）；BL3 标准行维持 ub512/SLOTS=3 原口径。

### R294 ★★★ **256K 轮内分账（LLAMA_ROUND_TIMING + LLAMA_SPEC_TIMING，零改码）：host 图管理税 = 两大差距的同一根病**（2026-09-23）

- **target ctx（489 调用 = 462 prefill ubatch + verify 等）**：`rebuild=468/489（95.7% 逐调用重建）`；**alloc 97.0 s（198 ms/调用）+ setin 35.7 s（73 ms）+ enqueue 157.3 s（322 ms）≈ 290 s ≈ TTFT 285 s 的 102%** ⇒ **预填充是主机受限，GPU 藏在 host 影子下**。draft ctx 对照：3726 调用 reuse 3689（99% 复用），host 账仅 ~5.5 s。
- **decode 轮（16 轮）**：`draft_decode 13.21 + selector 2.52 + walk 0.10 ms/round`；target 段 ≈73 ms/轮 = **每调用 rebuild（R246 单次 23 ms+）+ alloc/setin/enqueue** 的放大——R293"轮放大 2.65x"的真身。
- ⇒ **tg 3.81x 与 pp 1.90x 同根 = 每调用图重建 + alloc/setin/enqueue 主机税**（meta 税在 256K 服务负载的统治级放大；kq_mask 逐 ubatch 变长 → 图形状逐调用变 → rebuild，REBUILD 节点成因①）。
- ⇒ **主刀重排（第二次）：host 图管理税根治**（R246 修法③ mask 定尺/统一分块 + 多形状 build 缓存 + P-D1/N6B 老线）；**P-P3 内核线降为二刀**（host 不清，内核收益被埋，Amdahl 否决内核优先）。
- 口径注记：本探测 gen 高接受（draft 162/接受 102，tg 45.3）⇒ **AL 随内容波动 2.9–6.4**；基线表按同 prompt 双 rep 口径不变。

### R293 ★★★ **吐字 3.81x 分解实验（spec-off 对照 + journal SpecDecoding metrics）：主敌 = 投机轮结构开销 2.65x，纯 decode 步差仅 1.44x**（2026-09-23）

- **臂**：BL3NS（B5 + `NO_SPEC=1`）/ BL1NS（vLLM 去 `--speculative-config`）——唯一变量 = 投机开关；同 prompt（236,313 tok / 90.14%）、同 gen 128、同 TP4。双 rep 离散 ≤1%。
- **分解表**：

| 256K 同口径 | B5 llama | vLLM BL1 | 比 |
|---|---|---|---|
| 纯 decode（spec-off） | **28.9 t/s（34.5 ms/token）** | **41.5 t/s（24.1 ms/token）** | **1.44x** |
| tg（spec-on） | 32.7（AL 2.89/2.95） | 124.6（AL 3.28/3.66，journal） | 3.81x |
| **投机轮放大（on÷off）** | **1.13x**（round 88.7 ms = **2.57× 单步**） | **3.00x**（round 27.3 ms = **1.10× 单步**） | **2.65x** |
| 预填充 off / on | 876.5 / 817.2 t/s（draft 吃 **6.8%**） | 1486 / 1549 t/s（spec-on 反 **+4%**，机制未查，标未验证） | 1.70–1.90x |

- ⇒ **tg 差 3.81x = 纯步 1.44x × 轮放大差 2.65x**（AL 差仅 1.17x 是小头）。**主敌 = 我方 DFlash2 轮的结构开销**：round ≈ 2.57 个单步（draft 2 次调用 + selector CPU 1.5–4.3 ms + meta 主机循环 = E16 轮账碎件）；vLLM 证明 draft/verify 可压到 ≈1.1 个单步（draft 近免费）。
- ⇒ **吐字线主刀重排：P-D 轮结构（P-D3 两 draft 合一 / P-D1 meta 容器 Fix B / P-D5 selector 上 GPU）>> 纯 decode 内核微调（1.44x 是次要战场）**。
- **基建（本轮起生效）**：vLLM 测试走本地 NVMe 快载（`bl1-run.sh`，`WITH_SPEC` 开关；模型同性已验证 config.json 逐字节同、29G 同容 = 口径零变化；无盘机 NFS 加载 18 min vs NVMe 2–3 min）；模型路径转 `/mnt/3.84t/**`（红线合规）。原服务单元未动（mtime 9/17 自证）。

### R292 ★★★ **三基线标准线测定（stress-256k，256K 90% 填充综合压测）：BL1 = 152.5s/1549/124.6；BL2 = 结构性不可用（D7 硬崩）；BL3 B5 = 289.2s/817/32.7**（2026-09-23）

- **口径**：ctx 262144 的 90.14% 填充（prompt 236,313 tok；每 rep 加盐防 prefix cache、句序号防投机刷分）；gen 128；thinking on；同 prompt 字节、同生成长度。**BL1** = `vllm-1cat.service` 原样（FP8+DFlash2+TP4 卡 0,1,3,4，chunked prefill 2048）；**BL2** = 官方 llama（q8_0+DFlash2+256K）；**BL3** = B5（`/root/libdir-t8b`，TP4 卡 0,1,2,3，`--parallel 1 --ctx-size 262144`，ub 512，DFlash2 n-max 7）。
- **BL1 标准线（双 rep 离散 0.37%）**：TTFT **152.26/152.82 s**、预填充 **1552.0/1546.4 t/s**、吐字 **118.0/131.1 t/s**（tpot 8.47/7.63 ms）。
- **BL2 = 结构性不可用（0 分基线）**：官方 434ddbb 与 b11053 原版库（libdir-pristine）**双双启动 0.05s 硬崩** `ggml-backend-meta.cpp:543 GGML_ASSERT(src_ss[0].axis != SPLIT_AXIS_0)`（core dumped）= **D7 实锤**（DFlash2 selector top-k × vocab 切分）；官方版另证 butterfly AllReduce 仅支持 2 卡。**B5 的 D7 修复 = 该负载能跑的入场券**。
- **BL3 B5（双 rep 离散 0.06%）**：TTFT **289.36/289.02 s**、预填充 **816.7/817.6 t/s**（服务端 prompt_ms 288.5/288.1 双源吻合）、吐字 **32.29/33.19 t/s**（tpot 30.9/30.0 ms）、**AL 0.271/0.289（mean len 2.89/2.95）**。
- ⇒ **追平山 = 预填充 1.90x、吐字 3.86x**（tpot 8.0 vs 30.5 ms）。显存包络：BL3 TP4 四卡各 11.3 GB ✓ 合格（TP3 深 KV OOM：双 KV 头 rank +578 MiB 分配失败 → meta:1726 断言）。
- ⚠ 解读注记：合成 prompt 对双方投机解码均有可预测性红利（句序号已压低：llama AL 0.28 vs R282 真实文本 0.41），**横向公平**；勿与 1cat 文档真实文本 256K decode 50 t/s 直接混比。vLLM 不报 AL。
- **压测工具战痕（stress-256k 客户端已内建对策）**：API key 连字符截断→401；thinking 流走 `reasoning_content`/`reasoning` 字段；prefix cache 吃掉预填充→每 rep 加盐；重复 prompt 刷高投机 tg→句序号；llama OAI 严格字段→变体降级重试（记录所用变体）；llama `--ctx-size` 按 `--parallel` 均分→单序列语义；llama stdout 块缓冲→flush。
- vLLM 测后已恢复停机（用户交付状态）；BL2/BL3 为测试实例（8082），测后停机。

### R291 ★★★ **P-P0 预填充 TP 扫描 + M1 失衡锚：加卡全线负收益、M1 吞吐杠杆证伪 ⇒ 部署 = 全场景 TP3，卡数只按显存包络定**（2026-09-23，llama-bench 同源臂，判读规则**预登记**于 `p0-scripts/P0-PREREG.md`）

- **口径**：llama-bench pp-only（`-n 0` ⇒ **greedy sha256 门不适用**，R271 先例）；库 `/root/libdir-t8b`（md5 30876545/19675f8b/031009ae/d3affdd2 = B5 账本值 ✓）；env 恒定 `LLAMA_SM70_D256=1 / GGML_GALLOCR_SLOTS=3 / GGML_CUDA_P2P=1`；**唯一变量 = `-ts` + CUDA_VISIBLE_DEVICES**；模型 `/mnt/3.84t/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf`，q8_0 KV + `-fa on`；**NODROP 诊断口径（页缓存热）**；镜像臂序消漂移；TP4 分片机制自证 ✓（运行中四卡各 ~10.5 GB 均载快照）。
- **(a) R223 扩展性扫描**（`-p 8192,32768 -b 2048 -ub 2048 -r 8`，镜像对离散 ≤0.28%）：

| 配置 | pp8192（两臂） | pp32768（两臂） | 每卡 pp8192 | **TP4/TP3** | **TP6/TP3** |
|---|---|---|---|---|---|
| TP3 (0,1,2) | 1114.27±2.70 / 1108.13±3.69（均 **1111.20**） | 963.89±2.14 / 961.87±1.60（均 **962.88**） | 370.4 | — | — |
| TP4 (0,1,2,3) | 1084.37±6.19 / 1078.32±3.48（均 **1081.35**） | 939.40±1.95 / 940.01±1.32（均 **939.71**） | 270.3 | **0.973 / 0.976** | — |
| TP6 (0-5) | 1042.21±3.64 / 1047.01±2.68（均 **1044.61**） | 900.95±1.82 / 899.89±2.25（均 **900.42**） | 174.1 | — | **0.940 / 0.935** |

- ⇒ **R223 预登记的「4/3 线性外推 pp32768(TP4)=2864」证伪**：比值远低于 85% 判据线（<1.13）；加卡**负收益**（TP4 慢 2.4~2.7%，TP6 慢 6.0~6.5%），每卡吞吐随卡数递减 27%/54%。
- **(b) M1 失衡锚**（`-p 131072 -r 3`；⚠ **首跑 ub2048 四臂全 OOM** `failed to decode prompt batch, res=-2`@深 KV——kq_mask [n_kv×2048] 约 1 GB/设备 + T8 槽逐深度换 key 重预留（R282 同族教训），**零数据产出** ⇒ 按预登记附注 §5 改 `-b 512 -ub 512` 重测，镜像对离散 ≤0.16%）：

| 配置 | pp131072（两臂） | 均值 | TP4/TP3 |
|---|---|---|---|
| TP3 | 550.28±1.12 / 552.05±0.40 | **551.17** | — |
| TP4 | 533.58±0.50 / 533.56±0.51 | **533.57** | **0.968** |

- ⇒ **预登记主判据（比值随 FA 占比单调上升 8K<32K<131K）：0.973 → 0.976 → 0.968 = 三档持平（<1%）⇒ M1 吞吐杠杆证伪**（辅助判据 ≥1.55 亦远未达）。KV 头 1/1/2 失衡在算术上真实（RESEARCH 4.6），但其吞吐效应被**非缩放主导成本**完全遮蔽（三长度 × 两种 ub 口径均无加卡收益 ⇒ E8 族「per-ubatch 固定成本在关键路径」假说获强化，待逐 ubatch 分解直接验证）。
- **部署建议（R275 冲突注记）**：**全场景 TP3**——decode（R275：TP3 最优）与预填充（本条）同向，无场景冲突；**卡数只按显存包络定**（用户 2026-09-23 验收线：≤4×V100-16GB，超 4 卡不合格）；256K 进 4 卡包络靠 **P-M 显存瘦身**（Path B 镜像消除 + 槽位预算），**不靠加卡**。
- 复现：`p0-scripts/p0-sweep.sh`、`p0-m1-rerun.sh`；原始日志 `/root/llm/test/p0-logs/`。

### B2 的证据（R197，同源四臂 ABBA，同一套库）
- ⚠️ 库 md5 后续因 R199 的诊断代码（env 关闭时逐位等价）变为 `libggml-base.so b1751a8b4fa3d99d336e560b4e9b00fd`；rbs0/rbs3 两臂（50.79/50.78）复现了本行 B2 的 50.81。
- 库（两侧完全相同，唯一变量是 env）：`libggml-base.so 23c1b9179a0f97f1ed8fa87174fd0945`、`libggml-cuda.so.0.24.0 ea09324fd031ee676c020d2552bcb13e`、`libllama.so.0.4.1 d8ed3262ad7862abea160cca161dccbc`、`libllama-common.so.0.4.1 4e12c98faa3d79e3a011c401c2d11394`。
- 四臂 `fga1 / fgb1 / fgb2 / fga2`（PORT 8321-8324，NPRED=512，NODROP=1，CARDS=0,1,2，SPLIT=tensor，P2P=1）：
  - 控制臂（env 未设）：`MEDIAN_TG 99.83 / 99.39`，均值 ms/轮 **54.35 / 54.14**
  - 特性臂（`GGML_META_FULLGRAPH=1`）：`MEDIAN_TG 100.95 / 100.83`，均值 ms/轮 **50.73 / 50.72**（两臂差 0.02%）
- **机制（[META] 稳态，calls=832）**：`sub/call 47.6 -> 3.2`、`ar/call 46.6 -> 3.2`、`total 9.64 -> 3.87 ms/call`、`dev 6.30 -> 1.40`、`ar 2.21 -> 0.33`；
  `[RT]` target `enqueue` **18.34 -> 8.68 ms/轮**，draft `enqueue+alloc` **5.42 -> 1.70 ms/轮**。
- **四臂 greedy sha256 全部 = `f3edac19446ef641447f8391c71cb1b25a74055d662e9887890fa4a98602ca34`**（门未破）。
- ⚠️ 未决：`MEDIAN_TG` 只涨 1.1%（中位数落在 p1 上），而 **p1 几乎没动**（55.60 -> 54.98），p2/p3 才是主收益。p1 是每个进程的第一个请求（含首轮 CUDA 图捕获/实例化与分配器冷启动）—— 待单独查。
- ⚠️ 口径：本表是 **NODROP 诊断口径**（热加载 ~16 s）。要对外报数需重跑 `drop_caches` 官方口径（加载 255 s/臂）。
- 复现 B2 的命令（GGML_META_FULLGRAPH 是环境变量，默认关；基线臂必须显式打开，控制臂不设）：
  `CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=<t> NPRED=512 NODROP=1 PORT=<p> GGML_META_FULLGRAPH=1 bash /root/p60-ab-harness.sh`
  （harness 内部用 `env ... nohup llama-server` 启动，所以 export 即可传递。）
### B2 的轮时拆解（E16，R198，零改码：LLAMA_ROUND_TIMING_SYNC=1）





### R284 **F3-v3（预热录制出窗）否证：p3 之税不是录制成本，F3 三连不采用、存档待解谜**（2026-09-23 凌晨）
- 热身实验：p60 harness 加 opt-in WARMUP=1（health 后一条 16-token 计时外请求吞掉子图录制；默认零变化，备份 .pre-warmup）。机制自证 WARMUP_DONE ✓、STATUS=OK ✓。
- t46（ABBA 0/1/1/0，全臂 WARMUP=1，门值/draft_n 逐位一致）：a1 对照 45.59；a2/a3 的 p1 = 47.21/47.25 vs 48.25（**-2.2% 提升，四连一致**）；p2 44.75/44.85 ≈ 平（三轮 -1.4/-1.4/-0.2 不稳）；**p3 46.56 vs 43.67（+6.5% 降低，四连一致）** => 终值均值 45.58 vs 46.13 = +0.55 降低（a1/a4 对照 48.21/44.78/43.75，a2/a3 特性 47.24/44.68/46.47）。
- **关键否证：热身出窗后 p3 税依旧（+2.9 ms/轮 确定性）=> 「一次性录制成本」假说死亡**；p3 之税 = 捕获路径对 p3 负载的真实每轮成本（疑点收窄至：draft 图逐调用重建致签名逐调用新（指针入签）=> 每调用重录其 12 子图；或 KV 视图逐位移致签名漂移；两者均待记录计数器验证）。
- 处置：**F3（子图捕获）系列三连不采用（R281/t43、t44、t46），转档案**；p1 的稳定 -2.2% 与 p3 的稳定 +6.5% 并存 = 解谜后可能净值转正，列为待解谜项（诊断 = 每臂 slot 记录数计数器）。

### R281 **F3 子图级捕获 v1：数值根治、均值 +0.2 不采用；p1/p2 稳态 -1.5~3.0% 的曙光记档**（2026-09-22 深夜）
- 机制自证：`[META] dev` 5.05（CAP=1）vs 5.02（CAP=0）≈平（水槽签名计算成本吃掉早期 -48% 的大半）；record-only 鉴别臂证明**记录语义无病、命中路径地址漂移**（确定性损坏，两臂 draft_n=3549 逐字相同）=> 水槽签名（张量对象指针+op+ne+nb+data+src 指针与数据+view_src）根治 => **四臂 draft_n/draft_acc 642/419、843/390、336/258 逐位全对**。
- 墙钟（ABBA 0/1/1/0）：p1 48.01 vs 49.51（**-3.0% 提升**）、p2 44.62 vs 45.31（**-1.5% 提升**）、p3 46.49 vs 43.69（**+6.4% 降低**）=> 均值 46.37 vs 46.17 = **+0.2 降低，v1 不采用**。
- 遗留两修点（v2）：p3 之谜（短负载 +6.4%，非记录成本，疑签名成本/异步结构与深 KV 轮干扰）；签名成本优化（缓存签名或按 graph 指针+容器代数记）。
- 两坑双记：**env 判存在 = 0 也开**（FULLGRAPH 之训原文在案却复读）；**捕获记录不执行、存槽必 launch**（FGC :2905 范式）。
- 代码存档：ggml-backend-meta.cpp 内 GGML_META_SUBGRAPH_CAPTURE 门控（默认关）+ GGML_META_SUBCAP_RECORDONLY 诊断；`.f3-v1.broken` 存坏版对照。

### R285 **F3 子图捕获线终局：六连不采用（v1-v6），净值 0±0.3% 关闭存档**（2026-09-23 凌晨）
- t52（v6 直映射槽 + 二见才录 + 内容签名，ABBA 0/1/1/0 + 预热）：门值/draft_n 四臂逐位绿；**recs=356（v2 的 12507 -> 356，二见+O(1) 全效）**；**p3 税根除**（43.92 vs 43.79 = +0.3% 免税）；但 p1 果实缩水至 -0.9%、p2 +0.3% => 均值 45.56 vs 45.49 = **-0.07 ms/轮（+0.15%，噪声级）= 不采用**。
- 六轮序列与判决：v1 指针签名=确定性损坏（draft_n=3549 逐位同）；v2 水槽签名=修复但 +0.3；v3 颭138ms 出窗=+0.55（一次性成本假说被杀）；v4 内容签名=门绿但 +0.0（recs 12507 地址漂移重录）；v5 二见=+2.4（512 线性双扫描税放大 p3）；**v6 O(1) 直映射+槽内 seen=净值 0**。
- 科学结论：早期 p1 -2~4% 含重录开销/回放收益交叉假象；**捕获的真实净值 ≈ 0±0.3%（dispatch 节省被回放/簿记开销抵平）**。
- 技术副产品入档（可复用）：内容签名正确性法（data+ne+nb+op+type+src 数据，零对象指针，门值验证）；二见才录政策；O(1) 直映射槽+槽内 seen 态；record-only 鉴别法；热身请求计量协议（harness WARMUP=1 opt-in）。
- 代码状态：GGML_META_SUBGRAPH_CAPTURE 默认关（=0 走原直发），env 门控保留供后人；SUBCAP 计数/RECORDONLY 诊断保留。

### R286 **W8VOLTA-MMA 内核战役终局：七轮迭代 + ncu 定案，mma 路线存档；缺口真因 = n=8 内存空转**（2026-09-23 凌晨）
- 内核已完整落码（mmvq-w8-volta.cuh ~230 行，m8n8k4 碎片 + 0x6400 双操作数解码 + 双缓冲单屏障）+ env GGML_CUDA_SM70_MMA_Q8 门控接入 wrapper 派发；**test-backend-ops test 全程 7/7 后端通过 = CPU 神谕位级验证**（数学正确的移植）。
- 七轮结构迭代（对齐修复/宽存/去屏障/1-warp-CTA/宽碎片/u16/u32）：411 -> 362 us（累计 -12%），n=8 同形仍 3.2x 慢于在位 MMVQ（111.66 us）=> **mma 路线判定不采用、存档**。
- **ncu 定案**：IPC 0.16（发射间隔 6.1 周期）、内存依赖停顿 53%、实测占用 9.9%；源码级采样 = smem 碎片重载（LDS.U.128 x8）与全局装载（LDG.U16）各占约半。根本洞见：**n<=8 matvec 是纯访存型，张量核 mma 的暂存+碎片重载 = 零收益纯开销，SIMT 点积即最优形态**（设计书 5 节的『结果不确定』实验落定）。
- **缺口真因（F4' 新方向）**：稳态 ncu 对照 = n=1 内存吞吐 95.6%（roofline）vs **n=8 仅 34.2%**（计算 16%、余为停顿）=> 111.66 vs 82.10 us 的 30 us 缺口 = **n=8 访问形态/发射链空转，可修**（设计书『结构性固有值』论半错）。
- 复用资产：神谕验证内核（回归对照可复用）、ncu 剖析方法（test-backend-ops 小缓冲绕开大模型封死）、对齐陷阱表（q8_0 块 34B 步长 = 奇偶块混对齐，只 u16 标量放行；宽读 8/16B 必陷）。

### R287 **n=8 调形微杆战役：外链断链否证（寄存器外溢 4.5x 恶化），内链杆未竟；F4' 路线图定格**（2026-09-23 凌晨）
- **外链断链（tmp 双累加器奇偶交替）否证**：n=8 112.5 -> 507.6 us（4.5x 恶化！）且神谕 6/7 FAIL；真因 = ncols x rows 双累加器 = 寄存器爆炸外溢（慢化随 ncols 单调增长实证）。已清源还原。
- **内链断链（vecdotq.cuh vdr 段双部分和）**：实现完成（+2 寄存器零外溢设计），因拷贝清单第 4 次陷阱（vecdotq.cuh 未入 t15 清单）+ static/extern 链接冲突未竟测量；env GGML_MMVQ_SPLIT_ACC 旗（器件全局 + 主机一次性注入）保留可续。
- **F4' 路线图定格（三杆已探明）**：① 预取杆 = L2 预取被编译期锁死 DGX_SPARK（mmvq.cu prefetch 段），sm_70 解锁（PTX prefetch 指令可用性需核）或寄存器假读版；② 内链断链（本轮未竟）；③ halve-iters 尾判版（GB10 专用锁，R237 近亲已否）。ncu 铁证回顾：n=8 内存吞吐 34%（vs n=1 的 95.6%）= 30 us 缺口本体。
- 环境教训三连记：getenv 禁用于 __global__（主机专属，须器件旗+主机注入）；static/extern 链接冲突（同 TU 内 extern 声明 + static 定义必炸）；**拷贝清单陷阱第 4 次**（vecdotq.cuh）——规矩升级：新触文件一律先 `grep -c` 清单核验再构建。

### R288 **F4' 预取杆否证：四连否证坐实『调参路已走死』；F4 线全案关闭**（2026-09-23 凌晨）
- 预取杆两版皆败：寄存器假读版 = 0.0%（死读被编译器消除）；**真 PTX prefetch.global.L2 版 = 全段 +4.5~30% 恶化**（n=8 117.54 vs 112.51 us；n=2..5 +20-30%）=> 提示发射开销 + 缓存污染 > 延迟遮蔽。指令本身支持 sm_70（编译通过 = 新知识点）。
- **四连否证合流**（R237 nwarps / R244 rows_per_block / 外链断链 / 预取）+ mma 路线（R286）=> DESIGN-W8VOLTA-MMA.md 的『526 GB/s 是该内核结构固有值、便宜调参路已全部走死』**获独立复证**；30 us n=8 缺口 = 真结构性（剩余希望 = 内链断链未竟 + halve-iters 尾判版，均低概率）。
- 终态：树/库归净（git checkout + 重建绿，B5 行为位一致）；w8v 内核 .cuh 留档为 R286 资产（弃引用惰性）。
- **R288b 内链断链构造性死亡**：`VDR_Q8_0_Q8_1_MMVQ = 2`（vecdotq.cuh:243）=> vdr 链深仅 2 个 dp4a，2 路拆分零增益（投资门 = 先查链深再动工，省 2 轮）。**F4' 六杆全灭**（nwarps/rows_per_block/mma/外链断链/预取/内链断链）=> n=8 的 30 us 缺口三重独立复证为结构性。对照：`VDR_Q8_0_Q8_1_MMQ = 8`（MMQ 路 vdr=8，但 C5 已否 ne11=8 强推 MMQ -31%）。

### R282 **256K 全量压测（Q8_0+DFlash2）vs 早期基线（Q4_K_M+MTP）：pp +48%、吐字 +75%，项目价值在 256K 深度成立**（2026-09-22 深夜）
- 我方（t51H，ctx 262144、80% 填充实测 prompt_n=199121、ub512、6 卡 TP6、GGML_GALLOCR_SLOTS=3）：**pp 887.53/890.93 t/s（两轮一致），TTFT 224.4/223.5 s；吐字 39.88 t/s（25.08 ms/token，128 tok，DFlash2 draft_n=231 acc=94，AL~3.9）**。
- 参照（llama-256k-tensor-numa.log，Qwen3.6 Q4_K_M + MTP，262144 满填充）：**pp 600.53 t/s（261910 tok / 436.1 s），吐字 22.80 t/s（43.86 ms/token，MTP acc 0.70、mean len 3.74）**。用户确认 3.6/3.8 同速。
- 对照：pp **+48% 提升**（深度归一到同带约 +27% 提升）；吐字 **+75% 提升**（深度归一约 +65~70% 提升）；TTFT 224 vs 436 s。口径差异注记：我方 76% 深度 + Q8_0 更重权重（E14：Q4 在 Volta decode 反而更慢，参照的 Q4 劣势约 5%）。
- **T8 多槽 gallocr 的显存代价首测（256K 新发现）**：8 槽 x 每形状 ~1 GB 计算缓冲在 256K KV 挤占下逐级 OOM（meta:1821 断言 + cudaMalloc 994-1007 MiB 失败，均死在预填充 90% 尾段形状切换处）=> 长上下文须限槽（GGML_GALLOCR_SLOTS=3 实测放行）。**备忘：大 KV 场景的槽位预算入 T8 账。**
- 工具链两坑：llama-bench 不支持 --model-draft（spec 只走 llama-server）；256K 压测须 ub<=512 + 限槽。

### R280 **F2-1c 调用序重排否证 + 「影子源尸体」大破案**（2026-09-22 深夜，双教训）
- **F2-1c（特征拷贝前置 + process() 入同步窗 + gate 前置，GGML_SPEC_EARLY_PROCESS 门控）：零效果不采用**。ABBA 0/1/1/0：49.41 vs 49.39（-0.02 噪声内）；机制自证 selector 段 3.02->2.99（-0.03，拷贝尾不藏 gate 下，1c 预研的 -0.8~1.5 预估证伪）。已全量还原（标记串 EARLY_PROCESS=0 三库核验）。
- **「影子源尸体」破案（R279 的补账，新教训）**：t33 起全部臂莫名 +3.6 ms/轮（draft_decode 5.3->7.8、alloc_us +18%），旧库 t8b 同机正常 => 定位为构建内容差。真凶 = **R279 还原只走了服务器树 cp，而 t15-build.sh 每次构建从 /tmp/t15/ 影子源重新覆盖** => GGML_SCHED_POOL 默认 ON 的池化回归一直活着（其 +3.6 与 R279 的 POOL=1 臂 49.65 完全吻合）。修复 = 影子源换回 .pre-froot（md5 39c342b2 双侧核验）；**还原纪律升级：树 + 影子源双通道还原，还原后必须标记串三查（strings|grep -c 含被撤 env 名）**。
- F2-1c 的零效果结论不受污染（ABBA 双臂同环境、偏移相消）。清洁态复验：t38clean = **45.99 ms/轮**（锚定 45.91 / B5 45.94 复原）、draft_decode 5.23、三库标记 0/0。
- 现行状态：B5 干净基线在役；B6 主攻 = F3 子图级捕获（预期 -2~4，NCCL 官方推荐形态，定价前置已达标）。

### R279 **Fix A（副本/依赖对象池化 GGML_SCHED_POOL）否证：反向 7.8%，立即还原**（ABBA 0/1/1/0、门值与 draft_n/draft_acc 四臂逐位一致、机制自证齐）
- 墙钟均值 ms/轮 46.06（POOL=0）vs **49.65（POOL=1）** => **+3.59 = 变慢 7.8%，不采用**。
- 机制自证：alloc_us 625k -> 750k（+20% 更差）；[META] prologue 0.56 -> **1.17 ms/call（翻倍）**；total 7.80 -> 8.65。墙钟增量与探针增量（约 +3 ms/轮）自洽。
- 判读：A 代理的「身份稳定 => 指针键缓存复活」**必要不充分**——meta 容器乒乓（stc_compute[2] 逐 compute 换容器重建镜像，A 代理 (c)）是缓存线真堵点；单独池化只加遍历局部性税。**缓存线全部转入 Fix B（停容器乒乓 + 内容键化，E6 雷区）后备。**
- 配套否证（同日）：F1 指针折印快路径（r_stamp=509/513，split 每调用 ggml_free+ggml_init 重建张量工厂 ggml-backend.cpp:1085-1087 + input_dep 无条件新生 :1514）；F1-v3 跳过 needs_realloc（仅省 7%，alloc 真身 = 指派遍历过 meta init 钩子）。
- 标定（同日）：spin 探针主机穿透 50%（1 ms/call ~ 1.50 ms/轮）；[FPD] 证实单槽指纹对三形交替 workload 结构性永不命中（skipped=0 的真因）；[META] dev_hist 65% compute <20us = F3 子图捕获定价前置达标。
- 已还原：ggml-backend.cpp md5 = .pre-froot 一致；代码存档 wip-froot-pool.patch（未生成则以 git diff 补档）。

### R278 ★★ **关键路径标定 + F1 两级否证**（2026-09-22 晚，同源 env A/B、门值 f3edac19 全绿、draft_acc/draft_n 四臂逐位一致）
- **标定（spin 探针，ABBA 0/3000/3000/0）**：注入 8.85 ms/轮主机 -> 墙钟 +4.45 ms/轮（46.11 -> 50.56）
  => **主机穿透率 50%，杠杆 1 ms/call ~ 1.50 ms/轮**（E8 旧值 1.8x 的 post-T8 复测）。到 37.0 需砍 ~6 ms/call 当量。
- **F1（指针折印快路径）否证**：`r_stamp=509/513` —— split_graph 每次调用 `ggml_dup_tensor_layout` 新建副本张量 => 指针身份不可行；墙钟 +0.03 = 零效果（机制未生效的 A/B 不作数，**新纪律：任何特性 A/B 必须带机制自证计数**）。
- **F1-v3（槽命中跳过 needs_realloc 复检 + op_params[0] 入键）否证**：机制生效（fast=513/513）但只省 7%（852 -> 792 us/call）=> **alloc 2.16 ms/call 的真身 = 指派遍历过 meta init 钩子**（新张量逐个重算分裂态/镜像），needs_realloc 不是大头。墙钟 -0.08 噪声内。**不采用**。
- 候补：F1-v4 = meta init 钩子内容键缓存（深手术，E6 脆弱区）；F2 = selector（段间 GPU 空窗 ~2 ms/轮）；F3 = meta 逐节点派发（19.93 ms/轮，穿透待测）。
- 诊断代码：`[GALLOC_FAST]` 打印 + `GGML_GALLOCR_FAST` 门（默认 ON，=0 走复检）；`/root/t8-orig/ggml-alloc.c.pre-f1` 备份。

### R277 ★★ **B2 图内 selector TP 化首次上机: 两臂 LAUNCH_FAILED -- 病因锁定 meta 分裂代数缺跨设备合并语义; 含 B1/A 残值评估**

- 实现（env `GGML_SPEC_SELECTOR_INGRAPH=1` 门控 4 处条件: 装载取消 TENSOR_SKIP / 图端放开 767+824 / 主机关 CPU 路径; diff 归档 `wip-sel-ingraph.patch`）; 构建入 `/root/libdir-sel`（ggml 层与 B5 同源）。
- ABBA on/off x2: **两个 ON 臂 LAUNCH_FAILED（启动即 abort）**; OFF 控制臂正常, **46.13 ms/轮 复现 B5（45.94）**, 门值 `f3edac19…` 未破, `draft_n/acc` 逐字相同。
- **病因（断言原文）**: `ggml-backend-meta.cpp:550 GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)` -- meta 分裂代数的 handle_per_row 拒绝沿轴 0 切分的源; draft 的 `t_logits` 恰好是 vocab（=ne0）切分, 图内 `ggml_top_k` 的归约轴 == 切分轴 => 每设备只能算分片 top-k, **全局合并（+id 基址、跨设备归并）是框架里不存在的语义**。`ggml_get_rows(sel_next, ids)` 是第二堵墙（沿切分轴选行）。
- => 这就是「TP 下 in-graph selector 不能跑」的准确机制; 完成 B2 必须给 ggml meta 分裂框架**新增跨设备合并语义**（新模式, 红线要求先问用户）。
- **三线残值评估（实测口径）**:
  - **B2 完整体**（图内 top-k/gate/scoring）: 预期 **-2~3 ms/轮**（8 MB 拷贝 + CPU topk 0.65 + gate 0.55 + 同步点收缩）; 成本 = meta 分裂代数扩展 + get_rows 权重放置; 风险中高。
  - **B1 轻装退化版**（gate+scoring 图内、top-k 留主机）: **前提已碎** -- 图内 top-k 撞同一堵墙; 残值仅 **约 -0.6~1 ms/轮**（8 MB 拷贝与 CPU topk 原样留下）。
  - **A 融合**: 残值 **-1.1~1.3 ms/轮**（inject 链实测 gather 0.10 + copy 0.08 + submit 1.10）; 成本 = llama.cpp ubatch 混合模态批（同级新模式红线）。
- => 待拍板: (a) 做 meta 分裂扩展完成 B2; (b) selector 线收兵转下一头寸; (c) 只做退化版。

### R276 ★★★ **新鲜度 A/B（用户提示「图可能过时」）: P2P 边际归零、SELVEC 复核成立、FGC 边际反转 => 采用 B5 = FGC 关闭**

- 动机：图上「已落地提速」是老库/老时代量的，在当前 B4 库（`/root/libdir-t8b`）逐一重验；`ms/轮 = pred_ms ÷ (draft_n/7)`。
- ⚠️ 踩坑重演：`GGML_META_FULLGRAPH` **只判变量存在、`=0` 也是开**（HANDOFF §6.4 早有记载）⇒ 首轮 8 臂的 FGC 开关无效（两臂同 ON，数据作废），改 `env -u` 重测。
- **P2P（`GGML_CUDA_P2P=1`，旧记 +10.6%）**：关 46.85/46.95（均 46.90）vs 参照 46.93/47.03（均 46.98）= **-0.17%（零效果）** ⇒ butterfly 时代收益已被 NCCL+FGC+T8 吃尽，**旧值过时**（保留开启，无害）。
- **SELVEC（向量化 gate，R232/R256）**：关 47.02/47.40（均 47.21）vs 参照 46.98 = **+0.49%**（开=快 0.23 ms/轮）⇒ 复核成立。
- **FGC（B2 旧记 -6.5% = 开更快）**：

| FGC | p1（两臂） | p2（两臂） | p3（两臂） | 均值 |
|---|---|---|---|---|
| 开（B4 现状） | 51.74/51.58 | 44.29/44.10 | 45.08/44.81 | **46.94** |
| **关（`env -u`）** | 49.21/49.16 | 44.96/45.02 | 43.73/43.53 | **45.94** |

- ⇒ **关闭反而快 1.00 ms/轮（-2.1%），边际反转**。逐 prompt：p1 -2.5、p3 -1.3（关更快），p2 +0.85（关更慢，唯一回退）；均值效应是臂间离散（<0.5%）的 4 倍。
- **机制线索**：关 FGC 后 `enqueue_us/轮` **8.3 -> 18.6**（主机提交翻倍）**墙钟反而快** ⇒ cudaGraphExec 整图回放已成串行化税；T8 拿掉了 FGC 当年瞄准的 alloc 慢路径，其收益基础消失而代价留下（E7 课的镜像）。
- 正确性门：两轮共 12 臂 greedy sha256 全 = `f3edac19…`，`draft_n/acc` 逐字相同。
- ⇒ **采用 B5 = B4 + FGC 完全关闭（不设 `GGML_META_FULLGRAPH`）**：均值 **45.94 ms/轮**，tg **113.28 / 94.32 / 145.65**；距 37.0 还差 **-8.94 ms（-19.5%）**。
- ⚠️ 引用纪律：B2 的 -6.5% 不作废（它记录 T8 之前的边际），但**必须注明「已被 R276 反转」**；P2P 的 +10.6% 同理注明「R276 实测归零」。

### R275 ★★ **TP 卡数重扫（Phase B4 欠账）: 加卡全线变差，TP3 保持最优 => B5 = B4 不变，方向关闭**

- 口径：B4 配置（`/root/libdir-t8b` + `GGML_META_FULLGRAPH=1` + P2P=1 + 多槽默认 ON），NODROP、NPRED=512、镜像臂序 3a/4a/6a/2a/2b/6b/4b/3b；`ms/轮 = pred_ms ÷ (draft_n/7)`（§4.27 权威口径）。

| 配置 | ms/轮 p1/p2/p3（两臂） | **均值** | 相对 TP3 | tg p1/p2/p3 | AL p1/p2/p3 | 接受率 p1 |
|---|---|---|---|---|---|---|
| **TP3 (0,1,2)** | 51.62/44.05/44.75、52.43/44.35/45.17 | **47.06** | - | 107.9/96.3/142.0 | 5.58/4.25/6.38 | 65.3% |
| TP4 (0-3) | 52.96/46.64/46.68、52.97/46.58/46.73 | **48.76** | **+3.6%（变差）** | 79.7/84.9/129.9 | 4.23/3.97/6.08 | 45.9% |
| TP6 (0-5) | 61.66/50.36/69.18、61.64/50.51/69.69 | **60.50** | **+28.6%（变差）** | 81.4/79.9/90.6 | 5.03/4.03/6.29 | 57.4% |
| TP2 (0,1) | 两臂 tg=0 / PARSE_FAIL | 不可用 | - | - | - | - |

- **正确性门**：TP3 两臂 = `f3edac19…`（回归锚 ✓）；TP4 两臂同 = `ccc284e4…`、TP6 两臂同 = `69207026…`（TP 改 FP 求和序故各配置自有门值，**臂内一致** ✓）。
- **TP2 实据**：`ggml-backend-meta.cpp:1726 GGML_ASSERT` + `cudaMalloc failed: out of memory`（device 0）——14.5 GB 权重/卡 + 运行时 > 16 GB。
- **机制（两重税，独立叠加）**：① 主机提交/AR 税：`enqueue_us/轮` **8.1 -> 10.5（TP4）-> 20.5（TP6）**，加卡把提交窗口翻倍还多，吃光权重流摊薄；② **TP 数改变 FP 求和序 => DFlash2 draft 接受率 65.3% -> 45.9%/57.4%**，tg 双重受损（每轮 token 少 + 轮时高）。
- ⇒ **H1 否证**（先验 TP4 约 41 ms/轮，实测 48.8 反向）；K4「meta 税不灭则加卡不买带宽」在 NCCL+FGC+T8 时代**复证并加强**（新增：加卡还伤投机接受率）。
- ⇒ **TP3 保持最优，B5 = B4（配置与数值不变）**；剩余路径 = 解码主机残账（selector 4.6 + enqueue 提交窗口 8.3）与 fused（每轮 2 合 1，约 -1.7）。

### R274 ★★ **`GGML_GALLOCR_SLOTS` 翻默认 ON（用户拍板）: 三臂验证 = 默认复现 B4、`=`0 干净回退 B3**

- 改动：env 解析翻转（默认 8 槽；`GGML_GALLOCR_SLOTS=0` 关闭；N>0 设槽数）；顺带统一三条路径（`reserve_n`/`reserve_n_size` 不再要求 ids 非空、`alloc_graph` 单缓冲自动 reserve 恢复、`alloc_graph_n` 关闭时返回 false 由调用方走原逻辑）⇒ **env=0 时与上游行为逐位等价**。
- 三臂（同库 `/root/libdir-t8b`，NODROP）：

| 臂 | 模式 | ms/轮 p1/p2/p3 | 均值 |
|---|---|---|---|
| t9on1 / t9on2 | **默认（不设 env）** | 51.78/44.03/45.21、51.89/44.03/45.16 | **47.01 / 47.03** |
| t9off | `=0` 回退 | 54.66/47.46/48.98 | **50.37** |

- 默认 vs 显式 `=8`（R273）：47.02 vs 46.99（+0.06% 不变 ✓）；`=0` vs B3：50.37 vs 50.27（+0.2% 不变 ✓）；三臂 sha256 全 `f3edac19…`；默认臂 `[GALLOC_SLOT] hit=513 miss=11`、`=0` 臂零输出（真回退 ✓）。
- ⇒ **采用：开箱即 B4，无需 env**；**B4 基准库更新为 `/root/libdir-t8b`**（`libggml-base 19675f8b…`、`libggml-cuda 30876545…`、`libllama 031009ae…`、`libllama-common d3affdd2…`）。
- ⚠️ 注脚：预填充回归检查经用户取消**未测**——多槽对多形状 prompt 的 pp 影响未验证（风险记录在案）。

### R273 ★★★ **T8 多槽 gallocr（GGML_GALLOCR_SLOTS）: 均值 ms/轮 50.27 -> 46.99（-6.5%），门值未破 => 采用为 B4**

- **改动**（3 文件，`git diff` 归档 `wip-galloc-slots.patch`）：`ggml-alloc.c/.h` 加按图结构指纹（节点形状+op+src 连接+buffer id）的 **LRU 8 槽分配方案缓存**（每槽自带 node_allocs/leaf_allocs/vbuffers）；
  `ggml-backend.cpp` 在 `backend_ids_changed` 时先试 `ggml_gallocr_alloc_graph_n`，命中即**跳过 sync x3 + reserve（旧约 5 ms/call）**。env `GGML_GALLOCR_SLOTS=N` 门控，默认 OFF（= 逐位等价路径）。
- **接手审查修正 3 处**（mimocode 交接的实现未上机前）：① plan_key 补 op/ne[3]/src 槽位模式/src 形状（同形状不同连线不得共享方案）；② `reserve_n_size` 原绕过槽记账会破坏活动槽，已套 activate/sync；③ 删死字段。
- **构建**：`BUILD_RC=0` / `error:` 0 / `Built target` 115；标记串 `strings libggml-base.so | grep -c GGML_GALLOCR_SLOTS` = **2**（代码确在被加载的库）；装 `/root/libdir-t8`（**未动 libdir-instr 基线**）。
  库 md5（本构建）：`libggml-base da761bae…`、`libggml-cuda da13ceb2…`、`libllama 031009ae…`、`libllama-common d3affdd2…`（= R257 记录值）。
- **四臂 ABBA 同源**（唯一变量 env；同库同 ub；`CARDS=0,1,2 SPLIT=tensor P2P=1 L=/root/libdir-t8 NPRED=512 NODROP=1`；两特性臂额外 `GGML_GALLOCR_SLOTS_DEBUG=1`）：

| 臂序 | 控制（env 不设） | 特性（`=8`） |
|---|---|---|
| 1 | 54.71 / 47.41 / 48.96（**50.36**） | 51.79 / 43.98 / 45.11（**46.96**） |
| 2 | 54.41 / 47.27 / 48.90（**50.19**） | 51.75 / 44.05 / 45.26（**47.01**） |
| **均值** | **50.27** | **46.99（-3.29 ms/轮，-6.5%）** |

- **tg（两臂均值）**：p1 101.73 -> **107.24**（+5.4%）、p2 89.15 -> **95.88**（+7.5%）、p3 130.40 -> **141.19**（+8.3%）；AL 5.55/4.22/6.38 不变。
- **正确性门**：四臂 greedy sha256 **全部 = `f3edac19446ef641447f8391c71cb1b25a74055d662e9887890fa4a98602ca34`**；且四臂 `draft_n/draft_acc` 逐字相同（642/419、843/390、336/258）⇒ 数值与投机行为均未变。
- **机制证据**：`[GALLOC_SLOT] hit=513 miss=11`（两特性臂计数完全相同）= **97.9% 命中**，miss 全是启动期各形状首见；`spec timing` 的 `draft_decode` **6.01/5.93 -> 3.37/3.40 ms/轮**（draft 每轮 2 次调用交替图 = 旧慢路径重灾区）。
- **复现命令**：`GGML_GALLOCR_SLOTS=8 GGML_META_FULLGRAPH=1 P2P=1 CARDS=0,1,2 SPLIT=tensor L=/root/libdir-t8 NPRED=512 NODROP=1 TAG=<t> PORT=<p> bash /root/p60-ab-harness.sh`（控制臂**完全不设** `GGML_GALLOCR_SLOTS*`）。
- ⚠️ 注脚：① 口径 NODROP（诊断），官方 drop_caches 口径待补（B2/B3 同）；② 控制两臂差 0.33%（特性两臂仅 0.11%；效应 -6.5% 是臂间离散的 20 倍）；③ `GGML_GALLOCR_SLOTS` 是否翻成默认 ON 属产品开关，**等用户拍板**（当前默认 OFF 保护旧口径）；④ 多槽的显存占用（每槽一套 vbuffer）未单独计量，本负载无 OOM。
- ⇒ **离硬指标（37.0 ms/轮）还差 9.99 ms/轮（-21%）**；下一项 = T10 LM head（预填充 14.1%）/ 继续找解码侧结构项。

### R272 ★★★ **sm70 Path A 256K 端到端（用户场景）: +17.5%，与 Amdahl 预测逐字吻合 —— 预填充内核收益坐实**

- 口径：`llama-bench -p 262144 -n 0 -r 2 -ub 2048 -ts 1/1/1 --flash-attn 1 -ctk q8_0 -ctv q8_0`，
  `L=/root/libdir-sm70`，3 卡，四臂 ABBA，**唯一变量 `LLAMA_SM70_D256`**（ub 两边固定，非杠杆）。
- **pp262144 t/s**：

| 臂序 | stock (`=0`) | sm70 (`=1`) |
|---|---:|---:|
| 1 | 598.00 ± 0.11 | 701.99 ± 0.13 |
| 2 | 598.04 ± 0.31 | 702.85 ± 0.04 |
| **均值** | **598.02** | **702.42（+17.5%）** |

- 臂内离散 **±0.04–0.31**（极稳）；sm70 臂探针 `ACCEPT`×10（两臂各 5 次 real prefill），`Ktype=8`。
- ⇒ **与预测吻合**：算子级 −22.3% × 256K 上 FA 占比 78% ≈ **+17.4%**，实测 **+17.5%**。
- ⇒ **三层证据链闭合**：算子级 1.29x → 32K e2e +4.0%（Amdahl 19%）→ **256K e2e +17.5%（Amdahl 78%）**；解码零回退。
- ⇒ **预填充第二缺口（2163 vs 3567–4069）中，FA 这块已拿到内核级现货**；用户 256K agent 场景直接受益。
- ⚠️ **仍待收口**：greedy sha256 **门值须重立**（新核改累加序）；是否将 `LLAMA_SM70_D256=1` 设为默认（现默认 OFF 保护 B3）——属产品开关，等用户拍板。
- 墙钟：四臂约 82 min（12:28–13:50），单臂 256K×r2 约 20 min。

### R271 ★★★ **sm70 Path A 端到端（q8_0 KV，3 卡，同库同 ub，唯一变量 env）: pp32768 +4.0%，decode 零回退**

- 口径：`llama-bench -m Q8_0 -p 32768 -n 0 -r 3 -ub 2048 -ts 1/1/1 --flash-attn 1 -ctk q8_0 -ctv q8_0`，
  `L=/root/libdir-sm70`，**`-ub` 两边固定 = 测量形状，不是杠杆**（用户红线：ub 是用户调参，不入采用链）。
- **Path A 的 q8_0 路径坐实**：sm70 臂探针 `real#1..5 ACCEPT: sm70 d256 kernel selected`，`Ktype=8 Vtype=8`（q8_0）⇒ to_fp16 槽接通。
- **pp32768 t/s**（两臂 ABBA）:

| 臂 | stock (`=0`) | sm70 (`=1`) |
|---|---:|---:|
| 1 | 1921.02 ± 2.19 | 1996.75 ± 0.69 |
| 2 | 1911.45 ± 1.10 | 1990.69 ± 1.05 |
| **均值** | **1916.2** | **1993.7（+4.0%）** |

- **decode 零回退**：`tg64` stock **24.69 ± 0.03** vs sm70 **24.67 ± 0.02**（decode 走 VEC，sm70 正确 REJECT n_q=1）。
- ⇒ **+4.0% 与 Amdahl 吻合**：32K 上 FA 约占预填充 19%，算子级 +22% ⇒ 0.19 x 0.22 ≈ 4%。
  **长上下文（FA 占 78%）理论上限约 +17%**，256K 端到端尚未跑。
- ⇒ **这是内核收益，不是 ub**；采用判定：**算子级 + 端到端均为正且 decode 不回退 ⇒ 路径 A 可采用（待 256K 确认量级）**。
  门值：本口径是 llama-bench 吞吐，**greedy sha256 未测**（新核会改累加序，采用前须补）。

### R270 ★★★ **sm70-attn Path A 已移植并算子级 A/B: 真实形状 FLASH_ATTN_EXT 370.3 -> 287.5 ms（-22.3%，1.29x）**

- **移植（用户批准路径 A）**: `fattn-sm70-d256.cu` + `fattn-sm70-d256-kernel.cuh` + `blackmagic.h` + `sm70-vendor/`（162 文件）拷入开发树；
  `fattn.cu` 加 `BEST_FATTN_KERNEL_SM70_D256` 与 Volta 分派钩子；`CMakeLists.txt` 加 vendor include；
  **Path A = 扩 `kv_ok` 放行 q8_0 等 to_fp16 可转类型**（`sm70_d256_kv_type_ok`），复用现成 `to_fp16` 槽。
- **默认 OFF**（`LLAMA_SM70_D256=1` 才启用）⇒ B3  stock 路径不受影响，A/B 只靠 env 同库对照。
- **构建**: `build-instr` `-j128`，`BUILD_RC=0`，`MARKER_COUNT=2`，`libggml-cuda.so = 8481a6ed5386dd14fd0c62bf6acb1fbf`；装到 `/root/libdir-sm70`。
- **算子级 A/B**（`test-backend-ops perf --test-file` 只含 op74 `node_222` 两行，单卡 CUDA0，同库）:

| 形状 | stock (`=0`) | sm70 (`=1`) | 差 |
|---|---:|---:|---:|
| n_q=1 / n_kv=262144 | 5189.13 us | 5177.89 us | -0.2%（decode 不走，REJECT 正确） |
| **n_q=2048 / n_kv=262144** | **370275.19 us** | **287499.56 us** | **-22.3%（1.29x）** |

- ⇒ **超过验收停手线（<15% 则停）**；目标 <250 ms 未达但 287.5 已是实质收益。
- ⚠️ **口径**: 该 test-export 的 K/V 是 **f16**（`Ktype=1`），不是生产 q8_0；Path A 的 q8_0→to_fp16 路径**尚未在算子级单独验证**（端到端用 `-ctk q8_0`）。
- ⇒ **下一步**: 端到端 pp32768 / pp131072（`-ctk q8_0 -ctv q8_0`，≥2 臂报 ±）+ 解码不回退；门值**必须重立**（累加序变了）。

### R269 ★★★ **FA-D256 口径更正（读码核算）: 「1.49 TFLOPS = 峰值 1.2%」漏乘了 24 个 Q 头 ⇒ 真实约 35.8 TFLOPS = 峰值 28.6%**

- **旧账（R264/R266/R267/HANDOFF-2026-09-22）**: `FLOPs = 2*2*256*2048*262144 = 550 GFLOP`，除以 0.369 s = **1.49 TFLOPS = 1.2%**。
  这个 550 GFLOP 是**单个 Q 头**的 QK^T + PV；算子输出 `ne=[256,24,2048]` 明确有 **24 个 Q 头**（GQA=6 对 4 个 KV 头）。
- **更正后**: `550 GFLOP x 24 = 13.2 TFLOP` / 0.369 s = **35.8 TFLOPS = FP16 峰值(125) 的 28.6%**（不是 1.2%）。
- **带宽对照（同一次核算）**: 唯一 K/V（f16）= `4 kv_heads * 262144 * 256 * 2 B * 2(K+V) = 1.07 GB`；
  因 `ncols2=2` 而 `gqa_ratio=6` ⇒ KV 重读 **3x** = 3.2 GB ⇒ 按 900 GB/s 只要 **3.6 ms**，实测 369 ms ⇒ **不是带宽墙**（与 R263 一致）。
- **分派再确认（`fattn.cu:199-217`）**: Volta 专路要求 `gqa_ratio % 2 == 0` 才开 `ncols2=2`；**6 不能被 4/8 整除** ⇒ 只吃到 2 路打包，KV 仍重读 3 次。
  `switch_ncols1`（`Q->ne[1]=2048`）落 **ncols1=32, ncols2=2, ncols=64**；Volta D256/64 = `nbatch_fa=32, Q_in_reg=false, nstages=2`（`fattn-mma-f16.cuh:124-128`）。
- ⇒ **机会重估**: 不是「1.2% 病态、可 10-50x」，而是「28.6% 峰值效率、可争取 1.5-2x」——与 JS4 **+13%**、sm70-attn **+40%** 的声明同量级。
  **绝对收益仍然最大**（FA 占预填充 78.4%），但**不要按 50x 口径排期**。
- ⇒ 机制候选（按占比）: ① `nbatch_fa=32` 过小 ⇒ 256K 要 8192 次 softmax 重标定 ② `Q_in_reg=false`（注释: D=256 在 Volta 寄存器溢出）③ Volta `m8n8k4` 本征低效率 ④ GQA 只打包 2/6。
- ⇒ 路径不变: 结构性换内核（`SM70-ATTN-PORT-EVAL.md` 路径 A）仍是主线；常量 A/B（nbatch_fa）是廉价探路。

### R267 ★★★ **FA 算子级复测（test-backend-ops --test-file /tmp/ops-ub2048.txt, 单卡 CUDA0）+ 参考/上游对照**

- **真实形状** `FLASH_ATTN_EXT(n_q=2048, n_kv=262144, D=256, 24头, f16 K/V)` = **371,322 µs/run = 371.3 ms**（与 R264 的 368.8 ms 差 0.7%，同噪声带）⇒ 单算子墙复现。
- 同文件 `n_q=1` 那条 = **5,186 µs、192.94 GB/s**（decode 形状完全另一回事）。
- 该 test-file 的 K/V **已是 f16** ⇒ **371 ms 是纯 MMA_F16 主核，不含 to_fp16**；q8_0→f16 整条转换按字节估算 <1 ms/次 ⇒ **to_fp16/combine 不是墙**（与 R266 一致）。
- **本地参考**：
  - `sm70-attn`（1Cat Split-D N32 D256 插件）：**prefill-only**（q≥256），KV 只收 **F16/Q4_0**（**拒 q8_0**）；宣称 176k prefill +39.9%（证据等级 C，见 PLAN-GRAPH SM1/SM2）。门控 `D==256 && GQA 任意%` + mask。
  - `sglang-V100` dense D256：`BLOCK_M=64 / BLOCK_N=32 / THREADS=256`；尾块 split-KV 阈值表（SG3）；**满 8K chunk 不切分**（SG4）。
  - `flash-attention-v100`：Volta 仅 `m8n8k4`，D=256 smem 压力大（FA2）。
- **上游**：
  - **#28761**（open, sm_75）：D=256 的 Q-tile 32 vs 64 — 寄存器溢出 + 长 KV 下 tile64 **+19~50%**；按 `n_kv>8192` 自适应。**非 sm_70 直接可抄**，但证实 **D=256 长 KV 要更大 Q-tile** 的方向。
  - **#28907**（open, HIP cdna）：大 batch 下 mma-f16 对 D>256 仍优于 tile — 旁证 MMA 路线正确。
  - #29255：sm_70 layer-split + 大 prefill 的 launch 崩溃（无关本题，记档）。
- ⇒ **下一步排序不变**：① 结构性换 FA 内核（JS2 / sm70-attn 移植 / 自写 D256 prefill）② 若走 sm70-attn 必须 **F16 或 Q4_0 KV**（或扩 q8_0 支持）③ Q-tile/nbatch 常量微调作低成本 A/B（X16 警告仍有效）。

### R266 ★★★ **FA-D256 零改码诊断（[FAK] + 读码）: 预填充走 MMA_F16 + 整条 KV 转 f16；split-KV/combine 不是主因**

- 命令（服务器，零改码）: `GGML_CUDA_FA_KERNEL_DEBUG=1 CUDA_VISIBLE_DEVICES=0,1,2 GGML_CUDA_P2P=1 LD_LIBRARY_PATH=/root/libdir-instr llama-bench -m Q8_0 -p 32768 -n 0 -r 2 -ub 2048 -ts 1/1/1 --flash-attn 1 -ctk q8_0 -ctv q8_0` → `/tmp/fak32k.log`
- **[FAK]**:
  - `kernel=MMA_F16 D=256 n_q=2048 n_kv=32768 kv_type=8 need_f16_K=1 need_f16_V=1`
  - `kernel=VEC D=256 n_q=1 n_kv=32768 kv_type=8 need_f16_K=0 need_f16_V=0`
- **pp32768 q8_0 KV = 1924.40 ± 4.71 t/s**（与 R263 的 1920.27 一致）
- **分派链（`fattn.cu`）**: Volta + gqa_ratio=6 → `switch_ncols1<ncols2=2>` → n_q=2048 → **case ncols1=32, ncols2=2**。
- **Volta D256/ncols=64 配置**（`fattn-mma-f16.cuh:128`）: nthreads=128, occ=2, **nbatch_fa=32**, K2=128, V2=128, combine=128, nstages_target=2, Q_in_reg=**false**。
- **split-KV/combine 假设: 否（作为主因）**。`ntiles_KV = ceil(n_kv/nbatch_fa)` = 1024@32K / 8192@256K；combine/fixup 是主核之后的小 kernel。墙是 **MMA_F16 扫全 KV + `launch_fattn` 每次 `to_fp16` 整条 K/V**（`need_f16=1`），合计 1.49 TFLOPS。
- **JS4 2-CTA 门控（jusko `fattn-mma-f16.cuh:2221-2250`）不直接命中**: 要求 `nbatch_V2==64`（我们 **128**）、`Q->ne[1]<1024`（我们 **2048**）；`small_tp_long_prompt` 还要求 f16 KV + n_kv≥64K + 128≤n_q<1024。
- 参照仍在: **JS2** `fattn-q8-volta.cuh`；**JS3 常量照抄 = X16 已否证**（-1.19%）。
- 机器: 诊断后 6 卡空闲，无 llama-bench 残留。

### R265 ★ **状态复位（服务器）: 源码回到纯净，但 libggml-cuda 的 md5 与 R257 记录值不同（需下一轮知情）**

- 已还原：`mmvq.cu = 19c984c953d365fa0848a2c389239abc`（= 本地/HEAD）、`vecdotq.cuh = 275e009da0d98d02eb3467387efe6e9c`、
  `GGML_CUDA_MMVQ_WIDE` 标记串在库里计数 **0**、`ggml-backend.cpp = 967af7b5…`、`libggml-base.so = a8de7c48…`（= R257 值，可复现）。
- ⚠️ **但重建后的 `libggml-cuda.so.0.24.0 = ad317f8b757a4540f533bd7ec59b47d2`，不是 R257 记的 `0fb36620…`**。
  源码 md5 一致、算子级行为一致（`q8_0 m=4096 k=14336`: n=1 **82.81** / n=5 **95.20** / **n=8 112.37** µs，与 R236 的 82.10/111.66 同噪声带），
  说明差异来自**重编译本身**（R258 那次被打断的全量 CUDA 重建把若干 `.cu` 对象重编过，之后链接出的 .so 与 01:45 那次不再逐字节相同）。
- ⇒ 规矩：**引用"基线库 md5"时必须写清是哪一次构建**；行为等价要用算子级/门值复核，不能只看 md5 相等。

### R262 ★ **MMVQ-WIDE（VDR 2 -> 8，只对 Q8_0/ncols=8）: 算子级反而慢 17% ⇒ 不采用**（env 门控 + 已还原）

- 动机：R259 量到每条 dp4a 要付 7.2 条线程指令，其中约 5 条是 scale 换算。把 vec_dot 加宽到"一线程吃完整 32 值块"后，
  8 条 dp4a 只付一次 scale。
- 实现：`vecdotq.cuh` 加 `vec_dot_q8_0_q8_1_wide`（VDR=8），`mmvq.cu` 加 `wide` 模板参数 + env `GGML_CUDA_MMVQ_WIDE`，
  只实例化 Q8_0/ncols_dst=8；默认不设 env 时**逐位等价**。
- 同库同会话算子级（`test-backend-ops perf -o MUL_MAT -p 'q8_0.*4096.*14336'`；正确性 `test -p q8_0` 两臂各 **50/50 通过**；
  `strings libggml-cuda.so | grep -c GGML_CUDA_MMVQ_WIDE` = 1 证明代码进了被加载的库）：

| n | 1 | 2 | 3 | 4 | 5 | **8** |
|---|---:|---:|---:|---:|---:|---:|
| 基线 µs | 82.39 | 85.21 | 88.42 | 89.03 | 92.75 | **112.40** |
| 宽版 µs | 82.64 | 87.62 | 88.51 | 90.68 | 94.68 | **131.86（+17.3%）** |

- ⇒ **指令数少了、时间反而多了** ⇒ 机制不是"指令数"，而是 **vdr=8 让每线程同时持有 `v[8]+u[8]` 的寄存器压力**（accumulator 之外多 16 个活寄存器）。
- ⇒ 与 R237(nwarps)、R244(rpb) 合起来：**MMVQ 在 Volta 上已经是该结构的局部最优**；解码侧（n≤8）没有便宜的 kernel 空间。
- 源码已还原（`mmvq.cu`/`vecdotq.cuh` 回 `.orig-wide`）。

### R263 ★★★ **长上下文预填充的"KV 每 ubatch 反量化"假设: 代码是真的，但代价不是主要的 ⇒ 证伪**（同源同库，只改 `--cache-type-k/v`）

- 代码事实（`fattn-common.cuh:1026-1088`）：只要选中 TILE/MMA_F16 内核（**n_q>1 必选**），就 `to_fp16(K_data, K_f16, ggml_nelements(K))`
  —— **K 是整条 KV cache 视图**，且这段在每个 ubatch 都跑 ⇒ 形式上 O(n_kv) 每次 = **O(N²/ub)** 总量。
- 零改码判别实验（3 卡 `-ts 1/1/1`、`-ub 2048`、`-r 3`、`GGML_CUDA_P2P=1`，同库 `libggml-cuda e7c082cc`）：

| 深度 | q8_0 KV | f16 KV（**无任何转换**） | 差 |
|---|---:|---:|---:|
| pp8192 | 1964.67 ± 2.24 | 1980.87 ± 0.45 | +0.8% |
| pp32768 | 1920.27 ± 0.80 | 1938.83 ± 0.47 | +1.0% |
| **pp131072** | **1271.21 ± 1.43** | **879.36 ± 0.06** | **−30.8%** |

- ⇒ **f16 KV 完全不做转换，却在 131K 慢 31%** ⇒ 那条 O(N²/ub) 的反量化**不是长上下文的墙**；
  长上下文是 **注意力本身读 KV 的字节数**在说话（q8_0 = 1.06 B/元素 vs f16 = 2 B/元素），但**也不是纯带宽**（否则 f16 不会慢这么多，见 R264 的真实占比）。
- ⚠️ 顺带更正一处**口径混用**：本账本记的 "32K prefill 2162.44"（R239）是 **`llama-bench` 默认 = f16 KV**；
  而 harness/生产跑 **q8_0 KV**（实测 pp32768 = **1920.27**）。以后引用要写清 KV 类型。

### R264 ★★★ **预填充逐算子表（带名字，单卡，`-ub 2048`）: 84% 在注意力 + LM head，全部 GEMM 只占约 5%**

方法：`test-export-graph-ops -m Q8_0.gguf -o /tmp/ops-ub2048.txt -ub 2048`（91 算子）+
`test-backend-ops perf --test-file`（89 个被计时，合计 **470.3 ms**）。⚠️ 名字与耗时在**同一行**（早先按行号配对的表是错位的，已作废）。

| µs | 占比 | 算子 |
|---:|---:|---|
| **368757** | **78.4%** | **`FLASH_ATTN_EXT(name=node_222, ne=[256,24,2048])`，KV 视图 262144** |
| 66518 | 14.1% | `MUL_MAT(name=result_output, ne=[248320,2048])`（LM head 对**全部 2048 个位置**算 logits） |
| 5191 | 1.1% | `FLASH_ATTN_EXT(ne=[256,24,1])`（n_q=1） |
| 4766 / 4737 | 1.0% / 1.0% | `MUL_MAT ffn_gate / ffn_out` |
| 3644 | 0.8% | `GATED_DELTA_NET` |
| 3419 / 2792 / 1753 / 1707 | 0.7/0.6/0.4/0.4% | `Qcur_full / node_13 / z / linear_attn_out`（其余全是 <0.5%） |

- ⇒ **一个算子（长 KV 的 FA）吃掉 78.4%**；加上 LM head 的 14.1% 就是 **92.5%**；**除 LM head 外全部 MUL_MAT 合计只有约 4.3%**（fn_gate/ffn_out/Qcur_full/node_13/z/linear_attn_out 一堆 0.4-1.0%）。
- ⇒ 折算：n_kv=262144 时该 FA 单层 369 ms；256K 预填充要跑 121 个 ubatch，Σ∝n_kv ⇒ 16 层合计约 **357 s**，
  与实测 256K 预填充 **672 s** 同量级 ⇒ **这就是"256K 预填充 370 t/s / TTFT 672 s"的主因**（不是 GEMM，不是 KV 反量化）。
- ⇒ 效率：该 FA 的 FLOPs = 2*2*256*2048*262144 = 550 GFLOP / 0.369 s = **1.49 TFLOPS**（V100 FP16 张量核 125 TFLOPS 的 **1.2%**）
  ⇒ 与 R203「FA 11.7 GB/s = 病态」一致；**这是本项目目前最大的单一 kernel 缺口**。
- ⇒ **下一手（已写进 PLAN-GRAPH 的金色节点 FA-D256）**：D=256 / GQA=6 / Volta 的长 KV 预填充 FA。
  jusko 的两条恰好门控在我们的形状上：**JS4「2-CTA 紧凑核 +13.11%，门控 `gqa_ratio==6 && D==256`」** 与
  **JS2 `fattn-q8-volta.cuh`（smem 内 q8_0 反量化 -> mma.m8n8k4，101k 2.674->1.419 ms）**；
  本地副本 `v100-refs/jusko-llama-volta-qwen3flash/ggml/src/ggml-cuda/fattn-q8-volta.cuh` 可直接对照。

### R258 ★★★ **基础设施陷阱（必须先记住）: `/mnt/3.84t/v100-opt/llama.cpp` 是一份**陈旧副本**，真正的开发树是 `/root/llm/test/v100-opt/llama.cpp`；且 harness 的 `L` 默认值指向**过期的 libdir**

- 触发：R258 的 `rows_per_block` 实验把补丁打到 `/mnt/3.84t/v100-opt/llama.cpp/ggml/src/ggml-cuda/mmvq.cu`，构建"成功"（`BUILD_RC=0`、`Built target ggml-cuda`），但**实测数字与控制臂逐位相同**。
- 根因（三重）：
  1. `/mnt/3.84t/v100-opt/llama.cpp/build-instr/CMakeCache.txt` 里 `CMAKE_HOME_DIRECTORY=/root/llm/test/v100-opt/llama.cpp`，
     `CMAKE_CACHEFILE_DIR=/root/llm/test/v100-opt/llama.cpp/build-instr` ⇒ **那棵构建树编译的是 /root 的源码、对象也落在 /root**。
     nvcc 命令行实测 `-c /root/llm/test/v100-opt/llama.cpp/ggml/src/ggml-cuda/mmvq.cu` ⇒ **打 /mnt 的源码不生效**。
  2. 两棵树的内容**不一致**：`mmvq.cu` 本地/`/root` 树 = `19c984c953d365fa0848a2c389239abc`（76201 B），`/mnt` 树 = `c51dc2a1…`（75866 B）。
  3. `/root/p60-ab-harness.sh` 的默认 `L=/mnt/3.84t/v100-opt/libdirs/libdir-instr`，其 `libggml-cuda.so.0.24.0 = 64317717…`、
     `libggml-base.so.0.24.0 = d283439c…` —— **都不是当前基准**（当前基准 `libdir-instr` = `/root/libdir-instr`：
     `libggml-cuda 0fb36620…`、`libggml-base a8de7c48…`，见 R257）。⇒ **不显式传 `L=` 的臂全部作废**。
- 规矩（新增，写进 AGENTS）：① 只在 **`/root/llm/test/v100-opt/llama.cpp`** 改码；② 构建后必须 `md5sum` 核对 `.o/.so` 的 mtime 与内容，
  并用 `strings <lib> | grep -c <新标记串>` 证明代码进了被加载的那个库（本次 `strings libggml-base.so | grep -c SCHED_RETRY` = 2 才证明生效）；
  ③ 跑臂一律显式 `L=/root/libdir-instr`。

### R259 ★★★ **ncu 把 n=8 的 MMVQ 瓶颈定死了: 不是带宽，是"每条 dp4a 7.2 条指令"+ LSU 队列**（`test-backend-ops perf` q8_0 m=4096 k=14336，单卡）

| 指标 | n=1 | n=8 |
|---|---:|---:|
| `dram__bytes.sum` | 62.91 MB | 63.19 MB（**相同**） |
| `lts__t_sectors.sum`（L2） | 65 MB | 111 MB（1.7x，不是 8x） |
| `l1tex__t_sectors.sum`（L1） | 505 MB | 1076 MB |
| `smsp__inst_executed.sum`（warp 指令） | 7.44 M | **26.16 M（3.5x）** |
| `gpu__time_duration` | 83.74 µs | 115.26 µs |
| issue（相对 490 G/s 上限） | 89 G/s（18%） | 227 G/s（46%） |

- ⇒ **每条 dp4a 要付 7.2 条线程指令**（837 M thread-inst / 117 M dp4a）；`dram` 两边一样 ⇒ **n=8 的额外代价不是权重流量，是逐 (row,token) 调用的固定开销**（y 载入 + 5 条 scale 换算 + 地址）。
- ⇒ **预填充侧同理**：`l1tex` 1076 MB / 63 MB = **17x 扇出**（每 lane 读一行、行间相距 15 KB）。
- ⇒ 这条数据同时**解释了 R237/R244 的两个否定**：改 nwarps / rows_per_block 都在"每 call 开销"这一层，没碰到真正的结构。

### R258b ★★ **`rows_per_block` 2 -> 4/8/16（Volta, ncols 5..8）: 算子级零变化 ⇒ 否证（补丁已还原）**

- 补丁（`wip-` 脚本: `kernel-lab/rpb-patch.py`）把 `calc_rows_per_block` 的 `case 5..8` 改成 Volta 专用 env 常量。
  因 R258 的陷阱，**第一次的三个变体全部作废**（改的是 /mnt 副本）。
- 事后用 ncu 一起把机制也算清楚了：**提高 rows_per_block 不减少 y 流量** —— y 的读取次数 = (row,token) 配对数，
  rpb 变大时 CTA 数同比变小，**总流量不变**（这解释了"为什么这个杠杆在结构上不可能有效"）。
- ⇒ 与 R237/R244 合并成一条**结构性结论**：MMVQ 的"每 call 固定开销"**只能靠换 kernel 结构降低**（见 R260）。

### R260 ★★★ **Volta MMA kernel（`DESIGN-W8VOLTA-MMA.md` 的 decode 版）: 已实现、结果正确，但 238 µs vs 现役 111 µs ⇒ 否证（机制性，不是"没做完"）**

- 独立实验台（`1cat-vllm-v100-study/kernel-lab/w8volta-bench.cu`，`nvcc -arch=sm_70`，约 2 分钟一轮迭代，不碰 llama.cpp 构建）：
  `mma.sync.m8n8k4.row.col.f32.f16.f16.f32` + `0x6400|u` 魔数解码（1 xor + 1 LOP3 + hsub2 + hmul2 每 4 值）+ 自写参考实现（fp64）对拍。
- 正确性：置换探针（PROBE=1）逐点核对 A/B 的 k 配对 ⇒ **maxabs 3.5e-2（ref 尺度 110.7），与设计预期一致**（f16 舍入，需重立门值）。
- 时间线（m=4096 k=14336 n=8）：**673 µs（lane=row）→ 320 µs（8 lane 协作 + smem 暂存）→ 238 µs（split-K=8 + 预取一块）**；现役 MMVQ = **111 µs**。
- ncu 证据：lane=row 版 `l1tex` = **33.0 M sectors = 1056 MB（16.8x 扇出）**、`No Eligible 92.7%`、`3.01 active warps/scheduler`；
  协作版 `l1tex` 降到 **10.5 M**，但**指令数升到 16.9 M**（协作装载+回读的代价），仍是 `No Eligible 86.7%`。
- **判决性机制（ptxas 实测，非推断）**：`mma.sync.aligned.m8n8k4.row.col.s32.s8.s8.s32`
  ⇒ `ptxas error: Illegal matrix shape '.m8n8k4' for instruction 'mma'`；`m16n8k16.s8` ⇒ `requires .target sm_80 or higher`。
  ⇒ **sm_70 没有 int8 张量核** ⇒ 走 mma 就必须付 int8->f16 解码（约 2 指令/值，1024 值/CTA-k-block = 每 warp-k-block 约 88 条），
  而 **dp4a 路径对权重编码是零解码**。这就是为什么"mma 每字节只解码一次、复用到 8 个 token"的理论优势**被解码成本吃光**。
- ⇒ **结论：在 V100 上，对 Q8_0 + n<=8 这个形状，现役 dp4a MMVQ 已经接近该架构的最优解；mma 路线在 sm_70 上不成立**（不是调参问题）。
  复现：`cd /mnt/3.84t/v100-opt/kernel-lab && nvcc -arch=sm_70 -O3 -o w8volta-bench w8volta-bench.cu && ./w8volta-bench 4096 14336 8 200`。
- 剩下的**唯一**未试的 kernel 方向（下一轮候选，成本低）：**把 MMVQ 自己的 vec_dot 加宽（VDR 2 -> 8）**，
  即一个线程吃完整 32 值块（8 条 dp4a 只付一次 scale），把 7.2 inst/dp4a 压到约 2.5（估算 26.16 M -> 约 9 M warp 指令）。
  代价：`vecdotq.cuh` + `mmvq.cu` 各约 20-30 行、只对 Q8_0、会改累加顺序（必须重立门值）。**这是 R259 数据的直接指向。**

### R261 ★★ **sched 快路径重试（`GGML_SCHED_RETRY_ALLOC`）: 实测 `ok=0/563` ⇒ 撤销 R253 的"便宜修法"，只剩多槽 gallocr**

- 改动：`ggml-backend.cpp` 里在 `backend_ids_changed` 为真时**先试一次** `ggml_gallocr_alloc_graph()`（并逐节点校验
  `ggml_backend_buffer_get_type(node->buffer) == sched->bufts[node_backend_ids[i]]`，不通过就回退），env 门控 + 计数器。
- 四臂 ABBA 同源（同一套库，唯一变量 env；`L=/root/libdir-retry`，其余 = B3 口径）：
  `c1 100.96 / r1 101.18 / r2 102.00 / c2 102.32`，**四臂 greedy sha256 全部 = `f3edac19…`**（门未破），ms/轮 都约 50。
  计数器：**`bic=563 retry=563 ok=0 bad=0`** ⇒ **一次都没成功**。
- ⇒ R253 的"99.3% 是 `backend_ids_changed`、短路导致 alloc 根本没被调用"**计数正确但推论不成立**：
  单槽 gallocr 里 `needs_realloc` 每轮都为真（一轮在 4-6 张**节点数不同**的图之间交替），慢路径（sync x3 + reserve）**是必需的**。
- ⇒ 结论：要拿这约 2-5 ms/call，只能做**多槽/按图指纹的 gallocr**（R253 修法② = E6 五次 abort 的那条线，触及容器与 arena 生命周期）。
  便宜修法**已穷尽**；本改动不采用（源码已还原为纯净 `967af7b5…`，重建后 `libggml-base.so` 需回到 `a8de7c48…`）。

### R256 / R257 ★★ **把 B3 的向量化 selector 变成"默认路径"（不再需要 env），并同臂复核两条路径**

**R256（代码改动，本地 + 服务器逐字节一致）**: `speculative.cpp` 里
`const bool sel_vec = getenv("GGML_SPEC_SELECTOR_VEC") != nullptr;`
→
`const char * sel_vec_env = getenv(...); const bool sel_vec = (sel_vec_env == nullptr) || (atoi(sel_vec_env) != 0);`
（+3 行；语义：**默认走向量路径**，`GGML_SPEC_SELECTOR_VEC=0` 才回退到标量路径做 A/B）。原因：**生产服务不会设这个 env**，env-gated 的改进对用户等于不存在。文件 md5 `e5cad016…` → **`018a4666…`**（本地与服务器一致）。

**R257（同库同会话两臂，门都 `f3edac19…`）**:

| 臂 | env | 内部探针（每次调用） | tg p1/p2/p3 | **ms/轮 均值** |
|---|---|---|---|---|
| `r257A` | **不设（新默认）** | fetch 0.06 + topk 0.58-0.60 + **gate 0.53-0.55** | 101.71 / 89.29 / 130.21 | **50.37** |
| `r257B` | **`GGML_SPEC_SELECTOR_VEC=0`** | fetch 0.06 + topk 0.57-0.61 + **gate 0.84-0.88** | 100.55 / 88.53 / 129.50 | **50.80** |

- ⇒ **默认路径确实拿到了向量化的收益**（gate -0.31 ms/call），两臂差 **-0.43 ms/轮（-0.85%）**，与 B3 当初的 -0.42 一致 ✓；**两臂 greedy sha256 完全相同** ⇒ 逐位一致 ✓。
- ⇒ **B3 从"需要 env 的变体"升格为"fork 的默认行为"** —— 这是这一步的真正意义（用户的 `llama-server.service` 不加任何 env 也能吃到）。
- 库: `libllama-common.so = d3affdd29d15cccdca3b1e9076f95cf2`（新）、`libggml-cuda.so = 0fb36620…`（不变）、`libggml-base.so = a8de7c48…`、`libllama.so` 见 R254；标记串 `GGML_SPEC_SELECTOR_VEC` 仍在（1）。
- **本地保护**: 重新生成了 `b3-selector-vec.patch`（**5 393 B**，73 insertions / 2 deletions，`git apply --check --reverse` 通过）⇒ 未提交的改动不会因为 `git archive HEAD` 同步而丢失。

### R255 ★★ **把已识别杠杆加起来算一遍（决策用）: 全落地约 tg 140（≈39.7 ms/轮），150 还差约 2.7 ms ⇒ 150 需要"结构性的一步"**

起点 **B3 = 50.4 ms/轮（tg 101.2-102.05，AL 5.55）**；每轮账（R235）: verify 约 36 + draft 5.9 + selector 1.5 + 主机暴露约 7。已识别、**都带实测价码**的杠杆：

| # | 杠杆 | 省（ms/轮） | 依据 |
|---|---|---:|---|
| 1 | Volta Q8_0 小批量 MMA 内核（n=8: 526 -> 715 GB/s） | **-4.9** | R236 逐算子 + R237/R244 把两个便宜方向否证 |
| 2 | sched 慢路径（`alloc_splits` 5 ms/call x 1.08，按 55% 穿透） | **-3.0** | R248 `[SCHED]` 实测 + R253 定因 |
| 3 | 24 次图重建（23.2 ms/次 x 24/272） | **-2.0** | R247 直接量出单价 |
| 4 | selector 再向量化（gate/topk） | **-0.8** | R230/R232 内部探针 |
| | **合计** | **-10.7** | ⇒ **39.7 ms/轮 = tg 140** |

- ⇒ **要到 37.0 ms/轮（tg 150）还差约 2.7 ms**，而且上面四项**必须全部落地且互不冲突**（2 与 3 甚至可能是同一枚硬币的两面）。
- ⇒ **结论（写给自己和用户）: 靠"小步叠加"的天花板约 tg 140；150 需要一条结构性路线**，候选只有两个：
  - **(a) 1cat 的 fullgraph 路线**: 把一轮里的 target 验证 + draft 注入 + draft 块放进**同一张图**（或至少让这几种图共享一套已 reserve 的方案）⇒ 同时吃掉第 2、3 项（约 -5），并可能把 draft 的 2 次调用并成 1 次（再省约 1-2 ms 主机 / 轮）。
  - **(b) 把 E13 量到的"GPU 忙碌上限 168-185 t/s"直接变现**: 那要求主机侧几乎完全消失（E8 标定: 1 ms/call ≈ 1.8 ms/轮）。
- ⇒ 两条都属**大改动/新子系统**，按红线**需要用户拍板**；在此之前，第 4 项（selector）是唯一不需要拍板就能做的正收益项。

**第 4 项的具体做法（已读码确认，可直接开工）**: `work_topk`（`speculative.cpp:1206-1237`）现在是对**每个位置**的 `n_vocab=151936` 个 logit 做插入式 top-k（k=256），实测 **0.60 ms/call**。改成**两级**：先一趟「每 32 个取块内最大」（纯比较、无算术、可向量化）得到约 4749 个块极大值，对它们做插入式 top-256 得到候选块，再只在候选块内精扫 ⇒ 工作量约降 30 倍，预计 **0.60 -> 约 0.15 ms/call（-0.45 ms/call ≈ 每轮 -0.6 ms）**。
⚠️ **正确性靠「构造上逐位一致」**: 全程没有浮点算术（值都是直接拷贝），当前插入式选择给出的就是**（值降序, 下标升序）**的 top-256；两级实现只要按同一规则排序，就选出**完全相同且顺序完全相同**的集合（等值不下沉、早下标优先）。⇒ 门 `f3edac19…` 必须仍然一致。

### R254 ★ **计数器撤除 + 状态复位**: `ggml-backend.cpp` 回纯净（`CMP_ORIG=SAME`、`SRC_MD5=967af7b5…`），重建后 **`libggml-base.so` 回到 `a8de7c48…`**（= 本会话最早记录的基线值 ⇒ 这个文件的构建是可复现的），门 `f3edac19…`、MEDIAN_TG 101.60、ms/轮 54.84/47.63/48.75（**均值 50.41**）、STATUS=OK。
- **服务器树最终状态（权威）**: 本地 HEAD + **B3 selector patch**（`speculative.cpp = e5cad016…`）+ **[SLOT] 探针扩容**（`llama-context.cpp`，备份 `/root/llama-context.cpp.orig`）。库: `libggml-base = a8de7c48…`、`libggml-cuda = 0fb36620…`、`libllama = 031009ae…`、`libllama-common = 0a27d31d…`（标记串 1）。

### R253 ★★★ **慢路径的根因定案: 99.3% 来自 `backend_ids_changed`，不是分配失败（`bic=556 / allocfail=4`）**

把 R250 的计数器升级成"两种成因分开计数"（`wip-realloc-cause.py`，env `GGML_SCHED_REALLOC_COUNT`，只打服务器树；备份 `/root/ggml-backend.cpp.orig`），一臂（门 `f3edac19…`、MEDIAN_TG 101.52）:

```
[REALLOC] total=560 bic=556 allocfail=4 size=659 nodes=659 leafs=113
```

- ⇒ **560 次慢路径里 556 次（99.3%）是第一条件 `backend_ids_changed` 命中**（`||` 短路 ⇒ 此时 `ggml_gallocr_alloc_graph()` **根本没被调用**），只有 **4 次**是分配真的失败。
- ⇒ 读码对齐（`ggml-alloc.c:1052-1098`）: `ggml_gallocr_alloc_graph()` 本身**只有一个失败返回**（`needs_realloc()` 为真且 `galloc->n_buffers > 1` 时，`:1065`）；R251 已证 `needs_realloc` 内部几乎从不触发 ⇒ 两条独立证据一致。
- **机制（完整链条）**: 每轮交替出现 4-6 种图（目标验证 659 节点 / draft 注入 / draft 块 / KV 拷贝 66 节点 / prompt 分块）⇒ `split_graph` 重算的逐节点后端编号向量与"上一张图"的几乎必然不同 ⇒ `backend_ids_changed = true` ⇒ 走 `ggml-backend.cpp:1614-1640` 慢路径: **对 3 张卡各做一次 `ggml_backend_synchronize()` + 重算整套 buffer 方案 `ggml_gallocr_reserve_n()`**（≈5 ms/call，且**每次都把 GPU 流水打断**）。
- ⇒ **这解释了 E13「投机臂 GPU 35-45% 空闲」与 E8「主机时间穿透 55%」** —— 不是主机慢，而是**每轮 2-3 次把三张卡全同步**。
- **修法（按最小改动排序，都动核心文件 ⇒ 按红线先问用户）**:
  1. **让同步只在真的发生重分配时才做**（现在是无条件同步；`reserve_n` 若判定"现有 buffer 够大"则不重新分配、也不搬地址 ⇒ 那就不需要同步）⇒ 可能拿回"流水"这一半，改动最小。
  2. **按图指纹缓存分配方案**（把"上一张图"的单槽比较换成"最近见过的 4-6 张图"各自记账）⇒ 消掉那 ≈5 ms/call 的方案重算。
  3. 把 4-6 种图**合并/统一形状**（例如统一 prompt 分块、mask 定尺）⇒ 从源头减少交替，但这属于 R246/R247 那条线。

### R252 ★ **仪表撤除 + 状态复位复核**: `ggml-alloc.c` 回到纯净（`CMP_ORIG=SAME`），重建后门与基准都复现

- 撤掉 R251 的 gallocr 探针（它让该臂慢 2.7%），重建 `libggml-base.so = 9ecd835b…`（与探针前那次构建**逐位相同**）、`libggml-cuda.so = 0fb36620…`、`libllama-common` 标记串 1。
- 一臂复核: **greedy `f3edac19…` 门一致、MEDIAN_TG 101.20、ms/轮 55.06/47.64/48.78（均值 50.49）**、STATUS=OK ⇒ B3 口径再次站住（本会话第 6 次复现，区间 50.25-50.49）。
- 服务器树最终状态 = **本地 HEAD + B3 selector patch + [SLOT] 探针扩容 + [REALLOC] 计数器**（两个探针都 env 门控、不设时零成本且不改数值；备份分别在 `/root/llama-context.cpp.orig` 与 `/root/ggml-backend.cpp.orig`）。

### R251 ★★ **gallocr 慢路径的根因（第一层）: 不是 "`needs_realloc` 的尺寸问题" —— 节点/叶子计数从不变化，512 次检查里只有 2-3 次真"变大"，肇事张量是 DFlash2 自己的 `inp_target_features`**

探针（`wip-galloc-diag.py`，只打在服务器树，`GGML_GALLOCR_DIAG` 门控；备份 `/root/ggml-alloc.c.orig`）在 `ggml_gallocr_needs_realloc()` 里按三类原因计数，一臂结果：

```
[GALLOC] calls=128 nodes_diff=0 leafs_diff=0 grow=0 | last=
[GALLOC] calls=256 nodes_diff=0 leafs_diff=0 grow=2 | last=grow src inp_target_features of inp_target_features (view)
[GALLOC] calls=512 nodes_diff=0 leafs_diff=0 grow=3 | last=grow src inp_target_features of inp_target_features (view)
```

- ⇒ **节点数/叶子数差异恒为 0**；"尺寸变大"全程只发生 2-3 次，且肇事者是 **`inp_target_features`（DFlash2 注入的目标特征缓冲）**，出现在 prompt 阶段。
- ⇒ 但 R250b 的 `[REALLOC] total=560`（≈ 每次 sched 调用一次慢路径）依然成立 ⇒ **慢路径不是 `needs_realloc` 触发的**：`ggml-backend.cpp:1614` 的条件是 `backend_ids_changed || !ggml_gallocr_alloc_graph(...)`；我的计数器就在 `needs_realloc` 内部且被调用了 512 次 ⇒ `backend_ids_changed` 大多为假 ⇒ **失败点在 `ggml_gallocr_alloc_graph_impl()` 的分块分配过程里**（下一步就在那里加同款探针，属于纯仪表，不改行为）。
- ⚠️ **该臂的 `MEDIAN_TG = 99.32`**（常态 101.8-102.0，-2.7%）⇒ **探针本身有开销（每次调用 getenv + 每 128 次 fprintf）**，该臂只能看诊断内容，**不能用于计时**。
- ⚠️ 编译踩坑（记一笔）: `ggml-alloc.c` 是 **C 文件** —— `nullptr` 与 `static const bool x = getenv(...)`（非常量初始化）都会编译失败；连吃两次 `BUILD_RC=2` 才过。

### R248 / R249 / R250 ★★★ **主机侧最大的一笔找到了: sched 每次调用都在走"慢路径" —— 重新 reserve + 对 3 张卡各做一次 device synchronize（约 5 ms/call）**

**R248（`GGML_SCHED_SPLIT_TIMING` + `GGML_SCHED_SPLIT_CACHE`，同库两臂，门都 `f3edac19…`）**:
- `[SCHED]` 实测: **BIG（n_nodes>200）split = 228-257 µs/call、alloc = 4770-5212 µs/call**；SMALL split 24-25 µs、alloc 437-492 µs/call ⇒ **`alloc` 才是钱（约 5 ms/次，且每次调用都付）**，split 本身只值 0.25 ms（所以"split cache"本来就救不了）。
- `GGML_SCHED_SPLIT_CACHE=1` 两臂: **50.27 -> 50.29 ms/轮（零效果）**，且探针显示 `cached=0`（`last=0` ⇒ 指纹从未命中）⇒ 与旧判决一致，只是原因不同：**贵的是 alloc 不是 split**。

**R249（`GGML_SCHED_DEBUG_REALLOC=1`，一臂）**: 直接 abort 并留下证据 ——
`ggml-backend.cpp:1625: ggml_backend_sched_alloc_splits: unexpected graph reallocation (graph size = 66, nodes = 66, leafs = 31)`
⇒ **`ggml_gallocr_alloc_graph()` 会失败，代码于是走慢路径**（`ggml-backend.cpp:1614-1639`）:
`for each backend: ggml_backend_synchronize()`（**3 张卡各一次全同步**）+ `ggml_gallocr_reserve_n()`（重算分配方案）。

**R250b（`GGML_SCHED_REALLOC_COUNT=1` 计数器探针，一臂，门 `f3edac19…`、MEDIAN_TG 101.76）**: 一臂内 **`[REALLOC] total=560 unexpected=4`** —— 探针每 16 次打一行，最后一行 total=560；同一臂的 `[SCHED]` 调用数是 262 BIG + 250 SMALL = 512 ⇒ **约每次 sched 调用就有一次 realloc**（并且每次都要同步三张卡）。
- ⇒ **含义（重要）**: 每一轮里 ~2-3 次调用各自把 3 张 GPU **全同步**一次 ⇒ **跨调用无法流水**，主机提交时间被完全暴露 —— 这正是 E13 看到"投机臂 GPU 35-45% 空闲"、E8 看到"主机穿透 55%"的**机制落点**。
- ⇒ **修法方向（都属结构性改动，先问用户）**: ① 给 gallocr 做**多方案缓存**（按 node/leaf backend ids + 张量尺寸指纹存 2-4 份 reserve 方案），让交替出现的几种图各自走快路径；② 找到"每次调用都在变的那个尺寸"（嫌疑: `self_kq_mask->ne[0] = n_kv` 随 KV 增长 —— R246 已证 mask 尺寸确实随 n_kv 走）并把它固定（例如按 `n_ctx` 定尺），这样方案不再被作废。
- ⚠️ **服务器树当前状态（更新）**: 本地 HEAD + **B3 selector patch**（`speculative.cpp = e5cad016…`）+ **[SLOT] 探针扩容**（`llama-context.cpp`，备份 `/root/llama-context.cpp.orig`）+ **[REALLOC] 计数器**（`ggml-backend.cpp`，备份 `/root/ggml-backend.cpp.orig`，仅 `GGML_SCHED_REALLOC_COUNT` 设置时打印）。`libggml-base.so = 9ecd835b…`、`libggml-cuda.so = 0fb36620…`、`libllama.so = 031009ae…`、`libllama-common.so = 0a27d31d…`。

### R247 ★★★ **图复用到底值多少：关掉它 = 每轮 +21.3 ms（+42%）；单次重建的代价被直接量出 = 约 23 ms —— 24 次/臂 ≈ 每轮 2.05 ms（约 4%）**

同库同会话两臂（NODROP，NPRED=512，PORT 8207/8208；两臂 greedy sha256 都是 `f3edac19…` ⇒ 数值完全不变）：

| 臂 | 图复用 | tg p1/p2/p3 | **ms/轮 p1/p2/p3（均值）** | `[RT]` reuse/rebuild | build_us | alloc_us | enqueue_us |
|---|---|---|---:|---|---:|---:|---:|
| `r247A` | 开（默认） | 101.74 / 89.27 / 130.39 | 54.76 / 47.53 / 48.73（**50.34**） | 270 / 24 | 41 032 | 634 571 | 2 499 083 |
| `r247B` | **`LLAMA_GRAPH_REUSE_DISABLE=1`** | 73.96 / 61.21 / 90.48 | 75.33 / 69.32 / 70.23（**71.63**） | **0 / 294** | **508 119** | **4 595 986** | 3 817 654 |

- ⇒ **复用一次的轮 = 48.45 ms，重建一次的轮 = 71.63 ms ⇒ 单次重建代价约 23.2 ms**（反解：`50.34 x 294 = 270 R + 24 x 71.63`）。
- ⇒ **代价构成（先更正探针口径，2026-09-22）**：`[RT]` 的 `build_us`/`alloc_us` **只在"重建"分支里累加**（`llama-context.cpp:1421-1432`，与 `rebuild` 计数在同一个 if 块内）⇒ **复用轮的 build/alloc 根本不计数**，我先前按"每次调用平均"算出来的数字是错的。正确的**每重建**代价：
  - 复用开（24 次重建）：build 41 032/24 = **1.71 ms**、alloc 634 571/24 = **26.4 ms** ⇒ **28.1 ms/次**
  - 复用关（294 次重建）：build 508 119/294 = 1.73 ms、alloc 4 595 986/294 = **15.6 ms** ⇒ 17.3 ms/次
  ⇒ 钱几乎全在 `alloc_splits`（15.6-26.4 ms/次；§4.20 记的"约 5 ms/次"是乐观口径）；**两条臂单价不同**说明"形状交替时的重建更贵"（要先释放上一张图的缓冲再分配）。
  - `enqueue`（= `graph_compute` 全程）**每次调用都付**：复用轮 2 499 083/294 = **8.50 ms/call**、重建轮 3 817 654/294 = 13.0 ms/call ⇒ **这一项（约 8.5 ms/轮）比 rebuild（摊薄后约 2.5 ms/轮）更大**，是主机侧下一个要拆的东西（R248 用 `GGML_SCHED_SPLIT_TIMING` 去量其中的 split 份额）。
- ⇒ **本臂 24 次重建 = 557 ms / 13.1 s = 4.3% 的墙钟 ⇒ 每轮 2.05 ms（约 4% tg）**。
- ⇒ **推翻旧账**：HANDOFF §5 第 5 条把"多形状 decode graph 缓存"判为"收益上限 ~2%，放弃" —— 现在**直接量到 4%，而且它的前提（复用率已 93%）反而说明剩下那 7% 的形状切换就是全部代价**。**这条从"放弃"改回"候选"**。
- 与 R246 合并 ⇒ **重建来自两处**：① **prompt 分块/收尾的形状各异**（42/30/60/38/4…，每次都换形状 ⇒ 单槽缓存必 miss，约 12 次/臂）；② **验证轮里 9 次 `can_reuse` 因 `kq_mask->ne[0] == n_kv` 越界失败**（约 209 ms/臂）。
- ⚠️ 修法都要动 `ggml_backend_sched` 的单槽结构（E6 五次尝试都 abort）**或**把 mask 按 `n_ctx` 定尺/把 prompt 分块统一 —— 属**中等偏大改动**，按红线**先问用户**；但这一次它背后有 **23 ms/次** 的实测价码，不再是"看起来不大"。

### R246 ★★ **[SLOT] 探针扩容后的完整调用图: "24 次 rebuild" 不是每轮形状抖动，而是"非 8-token 形状的调用"（prompt 分块 + 部分验证）**

把探针上限从 24 提到 400 并加上 `n_tokens`（脚本 `wip-slot-probe.py`，只打在**服务器树**，备份 `/root/llama-context.cpp.orig`；env 门控、不设时零成本）。一臂（PORT 8206，NODROP，`MEDIAN_TG=102.04`、STATUS=OK）：

| 形状 | 次数 | 说明 |
|---|---:|---|
| `n_tokens=8, n_outputs=8` | **279** | 投机验证轮（= 272 轮 + 少量） |
| `n_tokens=42/30/60/38, n_outputs=0` | 8 | **prompt 分块前向（无 logits，但要 hidden states 供 draft 注入）** |
| `n_tokens=4, n_outputs=1` | 4 | prompt 最后一块（带 logits） |
| `n_tokens=6, n_outputs=6` / `4,4` | 2 | 请求末尾的**部分验证**（草稿块不满） |
| 其它（2/1 token 等） | 2 | warmup / 收尾 |

- ⇒ **`[RT]` 里那 24 次 "rebuild" ≈ 这 23 个非 8-token 形状的调用** ⇒ **不是"每轮形状都在变"**，我原先把"消除 rebuild"估成 4 ms/轮是**错的，作废**。
- **判据修正**: 探针的 `hit` = `!graph_reuse_disable && prev_active==res && res->can_reuse(gparams)`；验证轮里 `prev_active` 与 `res` **恒等**，所以 9 次 miss（call 5/41/85/100/132/205/224/250/275）**全部来自 `can_reuse` 返回 false**。
- **机制（读码）**: `can_reuse_kq_mask`（`llama-graph.cpp:48-65`）要求 `kq_mask->ne[0] == mctx->get_n_kv()` —— **mask 张量的第一个维度必须精确等于当前 KV 长度** ⇒ 每当（padding 后的）`n_kv` 越过被缓存 mask 的边界，就必须重建图。9 次 miss / 285 轮 ≈ 3%，且间隔 25-45 轮，与"KV 每轮涨约 5.5 个 token、跨过 padding 台阶"吻合。
- ⇒ **可能的修法（中等改动，未做）**: 把 mask 一次性按 `n_ctx` 分配（131072 x 8 x 2 B = 2 MB/卡），使 `ne[0]` 恒定 ⇒ 复用永不因 n_kv 失效；代价是注意力读 mask 的跨度变大。**收益上限要用 R247 的实测来定**。
- ⚠️ **服务器树当前状态（新增差异，必须记住）**: 本地 HEAD + **B3 selector patch**（`speculative.cpp = e5cad016…`）+ **[SLOT] 探针扩容**（`llama-context.cpp`，备份 `/root/llama-context.cpp.orig`）。`libllama.so = 031009ae…`、`libggml-cuda.so = 0fb36620…`、`libllama-common.so = 0a27d31d…`（标记串 1）。

### R244 ★★ **V100 MMVQ `rows_per_block` 2 -> 1（ncols 5..8）: 更慢 40%（算子级）⇒ 否证并还原；两次参数实验合起来证明"当前 MMVQ 参数已是局部最优"**

动机: R237 否证了 nwarps 后，剩下唯一没试过的参数是 **`calc_rows_per_block`（ncols 2..8 恒为 2）**；若 n=5 拐点来自**寄存器压力**（每线程 `tmp[8][2]` = 16 个累加器），把它降到 1 应该变快。

同源对照（同库同会话；算子级 = `test-backend-ops perf -o MUL_MAT -b CUDA0` 单卡；批级 = `llama-bench -p 8,16 -d 8192 -r 8`）:

| 口径 | rpb=2（基线） | **rpb=1** | 差 |
|---|---:|---:|---:|
| q8_0 n=1 单算子 | 82.10 µs | 82.87 µs | +0.9%（漂移量级，符合预期：该改动只影响 ncols 5..8） |
| q8_0 n=3 | 87.82 | 88.60 | +0.9% |
| **q8_0 n=8** | **111.66 µs（526 GB/s）** | **156.43 µs（375 GB/s）** | **+40%** |
| q8_0 n=512 | ~1090 µs | 1086.68 µs | 不变 |
| pp8 @ d8192 | **142.49 ± 7.00** t/s | **112.15 ± 4.02** | **-21%** |
| pp16 @ d8192 | 194.55 ± 6.44 | 193.31 ± 9.34 | 噪声内 |

- **判定: 不采用（负结论），已还原**（R245：`cp` 干净副本 + `cmp` + 重建）。
- 机制: rpb=1 让 **每个 block 只算 1 行**，于是**同样的 y（8 列激活）要被两倍的 block 各读一遍** ⇒ y 重读代价超过了寄存器压力的收益。⇒ **n=8 的 526 GB/s 不是"参数没调好"，而是这个内核结构（靠 2 行分块摊薄 y 重读）的固有结果**。
- ⇒ **合起来（R237 + R244）: MMVQ 在这两个方向都已到局部最优** ⇒ 要拿那 526 -> 715 GB/s 的约 30%（≈ 一次前向 -4.9 ms）**只能换内核结构**（mma.sync.m8n8k4 + 融合反量化 = 节点 HMMA 的 decode 版，参考 `v100-refs/ninfer-v100/.../w8_volta_mma_gemm.cuh`）。**已把设计要点写成 `DESIGN-W8VOLTA-MMA.md`，等用户批准后再动手**（红线：新子系统先问）。
- ⚠️ **过程教训（新增，与 §4.13 同族）**: 我用 `grep -a -e '^>' <file>` 去数 diff 行，**内层单引号被 Windows->ssh 路径吃掉后，`>` 变成了 shell 重定向** ⇒ 把 `/tmp/r244.log` **截断成 0 字节**（我自己的检查命令毁了日志；测量本身没事，数字从 `/tmp/r244-perf.log`/`/tmp/r244-bench.log` 找回）。⇒ **规矩: 远端命令里不许出现 `>`、`<`、引号这三种字符**（要过滤就用不带引号的正则，例如 `grep -c -e ^... 不行` 时改用 `awk` 的数法或另存文件再取）。

### R243 ★ **图槽探针（`LLAMA_GRAPH_SLOT_DEBUG=1`，同源一臂）: 目标 ctx 在稳态是 hit=1，形状不抖**

- 一臂（`GGML_META_FULLGRAPH=1 GGML_SPEC_SELECTOR_VEC=1 LLAMA_GRAPH_SLOT_DEBUG=1`，PORT 8204，NODROP）→ `P60_DONE STATUS=OK`
- 探针在前 24 次调用内的分布（探针自身有上限）: **`n_outputs=8 hit=1` 19 次**；miss 只在开头（call=1 `n_outputs=1`、call=2/3 `n_outputs=0`、call=4 `n_outputs=1`、call=5 `n_outputs=8` 首次）
- ⇒ **稳态下目标图稳定复用**（每轮 verify 都是 8 个输出位置）⇒ `[RT]` 里那 24/294 次 rebuild **不是"每轮形状都变"**，更像是"每 prompt 开头 + 少量边界"；⇒ **它不是可持续优化的 4 ms/轮**（我原先的估计作废）。
- ⇒ 结论: 目标侧主机成本（10.6 ms/call）**不能靠"消除形状抖动"来降**；要降只能动 FGC 命中路径本身（sched 提交 / 图启动）或那 2.04 次/轮的 draft 调用（E4 已判定结构性不可复用）。

### R241 / R242 ★★ **预填充 ub 阶梯定案（32K）+ R219 的两点模型命中（64K 预测 1819 vs 实测 1805）**

同一套库（`libggml-cuda.so = 1394b4f045f61d0edc0f23977f554c15`，已还原干净 mmvq）上的预填充阶梯（`llama-bench -ngl 999 -sm tensor -ts 1/1/1 -n 0 -r 8 -ctk q8_0 -ctv q8_0 -fa 1`）:

| n_prompt | `-ub 512` | `-ub 1024` | `-ub 2048` | 备注 |
|---|---:|---:|---:|---|
| 32768 | 1691.81 ± 1.52 | **1951.03 ± 1.94** | **2173.44 ± 4.63**（R239 2162.44） | ub512->1024 = **+15.3%**，1024->2048 = **+11.4%**，合计 **+28.5%** |
| 65536 | 1129（R222 历史） | - | **1805.29 ± 1.06** | **R219 两点模型预测 1819 ⇒ 差 0.8%** ✓ |

- ⇒ **R219 的预登记被命中**：`pp65536@ub2048 ≈ 1819`（实测 1805.29）—— 这同时坐实 R218 的翻案：**256K 那个 370 t/s 的点是 ub 口径差，不是"长度换机制"**。
- ⇒ **解码代价只与 ub>=2048 有关**：同库同会话解码 ms/轮 = **50.25（ub512）/ 50.22（ub1024）/ 51.36（ub2048）** ⇒ **ub1024 是"零解码代价"的中间档**（预填充已拿 +15.3%）；ub2048 拿 +28.5% 但要付 -2.2% 解码。
- ⚠️ `-b 8192 -ub 8192`（R223 预登记、一直没跑的那条）**直接 abort**：`ggml-backend-meta.cpp:1726: GGML_ASSERT(bufs.back() != nullptr) failed`（= 大 ubatch 的中间缓冲分配失败/显存不足）⇒ **大 ub 有硬上限，不会再往上扫**。

#### 预填充采用链（第二缺口，与解码链 B 分开记）

| # | 配置 | pp32768 | pp65536 | 相对上一阶段 | 判定 |
|---|---|---:|---:|---|---|
| U0 | 默认（无 `-ub`，即 512） | 1691.81 ± 1.52 | 1129（历史） | 起点 | 起点 |
| **U1** | **`-ub 1024`** | **1951.03 ± 1.94（+15.3%）** | - | +15.3% | **采用（零解码代价）** |
| **U2** | **`-ub 2048`** | **2173.44 ± 4.63（+28.5%）** | **1805.29 ± 1.06** | +11.4% | **采用（代价 -2.2% 解码）** |

⚠️ 两条都**不改源码**，只改运行参数；生产 unit（只读红线）里没有 `-ub` ⇒ 需要用户自己加 `Environment=` override 或 `ExecStart` 追加。推荐按场景选：**256K agentic（预填充主导）用 U2；要保解码就用 U1**。

### R240 ★★★ **干净状态恢复 + B3 复现（差 0.06%）+ `-ub 2048` 的解码代价（同库同会话两臂）**

- **还原**: `cp /root/mmvq-c4.cu.orig -> mmvq.cu`，`CMP_ORIG=SAME`，md5 `19c984c953d365fa0848a2c389239abc` ✓；重建（`BUILD_RC=0`/`ERR=0`/`BUILT=115`）⇒ 装好的 `libggml-cuda.so = 1394b4f045f61d0edc0f23977f554c15`、`libllama-common` 标记串 `COMMON_MARKER=1` ✓（R239b 的污染已清除）
- 两臂同库同会话（NODROP、NPRED=512、PORT 8201/8202、`GGML_SPEC_SELECTOR_VEC=1`）:

| 臂 | ub | tg p1/p2/p3 | **ms/轮 p1/p2/p3（均值）** | 门 |
|---|---|---|---|---|
| `r240A` | 默认 512 | **101.56 / 89.12 / 130.63** | 54.65 / 47.24 / 48.87（**50.25**） | `f3edac19…` ✓ |
| `r240B` | **2048** | 99.18 / 87.42 / 128.00 | 55.96 / 48.28 / 49.84（**51.36**） | `f3edac19…` ✓ |

- ⇒ ① **B3 复现**: r240A 的 **50.25** vs B3 的 **50.22**（差 0.06%）⇒ 采用链初值今天再次站住，库状态可信。
- ⇒ ② **`-ub 2048` 不改变数值**（两臂门都是 `f3edac19…`）—— 与 R219 的担心相反（短 prompt 下 ub 根本不改计算图）⇒ 这同时证明 **R239 那次门破是坏内核造成的，不是 -ub 造成的**。
- ⇒ ③ **但 `-ub 2048` 让解码慢 2.2%**（三条 prompt 方向一致：+2.4% / +2.2% / +2.0%；`[RT]` enqueue 从 2.545 s 涨到 2.653 s = +4.3%）⇒ **预填充 +27.8% 换解码 -2.2%，是 trade-off 不是纯赚**。
- ⇒ **判定: 记为预填充成果 U1（对用户 256K 场景净赚）；解码侧不采用**（解码基准仍是 **B3**，其命令**不带** `-ub`）。
- ⚠️ 生产 `llama-server.service`（只读红线）没有 `-ub` ⇒ 要生效需用户自己加 override，我不碰。
- ⚠️ 过程教训（本节第二次）: r240 第一次启动被**自己的 `pgrep -f r24[0].sh` 守卫**杀掉（`ABORT_SELF`）—— 因为该正则**匹配脚本自身的 cmdline**（§4.32 的同族错误）。⇒ **不要给"作业自身"加 pgrep 守卫**；要防重复启动就用锁文件。

### R239 ★★ **`-ub 2048` 的预填充杠杆在 32K 上同会话实测确认 +27.8%（这条 A/B 天然不受 MMVQ 影响）**

同库同会话 A/B（`llama-bench -m Q8_0 -ngl 999 -sm tensor -ts 1/1/1 -p 32768 -n 0 -r 8 -ctk q8_0 -ctv q8_0 -fa 1`）:

| 臂 | pp32768 | 每 token |
|---|---:|---:|
| `-ub 512` | **1691.81 ± 1.52** | 19.37 ms |
| `-ub 2048` | **2162.44 ± 1.78** | 15.15 ms |

- **+27.8%**；且 `-ub 2048` 的 2162.44 与权威基线 `2164.72 ± 3.19` 吻合（差 0.1%）⇒ 两条独立证据一致。
- **这条 A/B 不受当时坏内核的污染**：n=32768 的 matmul 全在 `ne11 >= 64 -> mul_mat_cublas`，**根本不走 MMVQ** ⇒ 内核参数改不到它。
- ⇒ **判定: 采用（预填充阶段成果 U1）**：`-ub 2048` 零改码、只改一个启动参数；按 R222 的表在 64K/128K/256K 是 1.25× / 1.21× / 约 1.18×。
  - 约束①显存（峰值待补测）；②门（R240 在干净库上验：短 prompt 下 ub 不改变计算图 ⇒ 应与 B3 的门完全一致）；③解码不回退（R240 验）。
- ⚠️ **生产 `llama-server.service`（只读红线）里没有 `-ub`** ⇒ 这一条只能由用户自己改 unit 或加 override，我不碰；同时它并不改变 llama.cpp 源码（属于运行配置）。

### R239b ⚠️⚠️ **我自己的补丁脚本 bug：锚点搜索没有上界 ⇒ 改到隔壁函数 + 源码状态被污染（已定位、已还原）**

- `wip-mmvq-volta-nwarps.py` 的实现是「从 `MMVQ_PARAMETERS_VOLTA) {` 往后找**第一个** `case 5..8 / return N;` 块」。当目标块**已经是**目标值时，它会继续往后命中 **`calc_rows_per_block`** 里同形状的块（`mmvq.cu:611-615`）并改它。
- 实际轨迹：run1（补丁脚本路径写错、且脚本**没 exit**，继续跑完）→ run2 打补丁（`calc_nwarps` 488 行 2→4，md5 `c09c8af9`）→ r237b 再"打"一次（这次改到 **574 行的 `calc_rows_per_block`** 2→4，md5 `32a7d634`）⇒ **r237b 实测的其实是 nwarps=4 且 rows_per_block=4**（"更慢"的结论仍成立，但变体标注必须更正）。
- r239 想还原成 2，脚本又在**更后面**找到另一处 `return 2` 并报 `ALREADY_AT_TARGET`，**没还原成功**（`ORIG_MATCH=NO`）⇒ 紧接着那个 harness 臂跑在坏内核上：`MEDIAN_TG=93.45`（-8.8%）且**门变成 `69207026…`**（= 无投机口径的值）⇒ 说明这个改动**改变了数值路径**（k 切分顺序变了）。
- ⇒ **规矩（新增，与 §4.13/§4.15 同族）**：**任何"找第一个匹配就改"的补丁脚本必须给搜索加窗口上界**（如 `b.find(cand, i, i+800)`），找不到就 `exit`；改完**必须 `diff` 原文件并核对 diff 行数**——本次若核对就会当场发现"应该 1 行却改了 2 行"。
- ⇒ **推论（重要）**：MMVQ 的 `nwarps`/`rows_per_block` **会改变数值结果**（累加顺序），所以任何这类内核调参都**必须重建门值**，不能沿用 `f3edac19…`。

### R237 ★★ **V100 MMVQ `ncols=5..8` 的 nwarps 调参: 4 比 2 更慢（算子级 -7.3% / 批级 -3.3%）⇒ 否证并还原**

动机: R234 的曲线在 **n=5** 处有拐点，而 `calc_nwarps` 的 **VOLTA 分支在 5..8 用 2**（2..4 用 4）—— 这个值是从上游 GENERIC 表继承来的，**从没在 V100 上调过**。

同源对照（同一套库、同一次会话；算子级 = `test-backend-ops perf -o MUL_MAT -b CUDA0` 单卡，批级 = `llama-bench -p 8,16 -d 8192 -r 8`）:

| 口径 | nwarps=2（基线） | nwarps=4 | 差 |
|---|---:|---:|---:|
| q8_0 n=1 单算子 | 82.10 µs | 83.07 µs | +1.2%（漂移量级） |
| q8_0 n=2 | 84.18 | 85.19 | +1.2% |
| q8_0 n=4 | - | 88.91 | - |
| **q8_0 n=8** | **111.66 µs（526 GB/s）** | **119.83 µs（490 GB/s）** | **+7.3%** |
| pp8 @ d8192 | **142.49 ± 7.00** t/s | **137.80 ± 6.12** | **-3.3%** |
| pp16 @ d8192 | 194.55 ± 6.44 | 194.93 ± 9.65 | +0.2%（噪声内） |

- **判定: 不采用（负结论）**；⚠️ **但当时写的"已还原、逐字节相同"是错的** —— `patch-nwarps.py 2` 实际**没有还原成功**（`ORIG_MATCH=NO`），根因与更正见下面 **R239b**。真正还原发生在 R240（`cp` + `cmp`）。⇒ **n=5 拐点的机制不是 nwarps**（r237b 实测的变体应记为 **nwarps=4 且 rows_per_block=4**）。
- 机制推断（下一轮据此换方向）: ncols=8 时每线程要持有 **8 组 y 向量 + 16 个累加器**（`tmp[ncols_dst][rows_per_block]` = 8x2），加重 warps 只会放大共享内存归约（`tmp_shared[3][8][2][32]` = 6 KB）而**不减每线程压力** ⇒ 更像**寄存器压力/溢出**。要吃掉那 **526 -> 715 GB/s 的约 30% 差额，只能重写内核**（参考 `v100-refs/ninfer-v100/src/ops/linear/w8/w8_volta_mma_gemm.cuh`：mma.sync.m8n8k4 + 融合反量化，W8G32 == Q8_0）。
- ⚠️⚠️ **过程事故（新增规矩）**: 第一次 r237 启动时补丁脚本路径写错（`/root/patch-nwarps.py` 不存在），而脚本**在 PATCH_RC=2 之后没有 exit**，于是它带着**未打补丁**的源码继续走完整条流程；我随后又启动了一次 ⇒ **两个 r237 同时跑**（一个在 build+install，一个在 measure）⇒ `PERF_RC=139`(SIGSEGV) + `BENCH_RC=135`，**正是 §4.37 描述的情形**（替换共享库杀死正在跑的测量）。已用 r237b 干净重做。
  ⇒ **规矩: 关键前置步骤（打补丁 / 校验 / 备份）失败必须 `exit`，不许"继续往下跑"；同一作业启动前先 `pgrep` 查同名脚本。**

### R235 ★★★ **决定性拆账（同源两臂）: 一轮 = 一次 8-token 目标前向 + draft 5.9 + selector 1.2 + 主机暴露约 10 ms；并发现 llama-bench 的 `-p` 绝对值被高估约 17 ms/次**

同源两臂（NODROP，NPRED=512，CARDS=0,1,2，SPLIT=tensor，P2P=1，L=libdir-instr，GGML_META_FULLGRAPH=1）:
- **`r235NS`**（`--spec-type none`，只跑目标）: tg **41.61 / 42.78 / 42.76** ⇒ **23.4-24.0 ms/token**（与 E13 的 22.07、Z1 的 22.18 一致 ⇒ **服务路径 n=1 没有退化**；sha256 = `69207026...` 是无投机口径的正常值，**不是门破**）
  - `[RT]` rounds=1436、**enqueue 2.85 ms/call**、alloc 0.30、build 0.02 ⇒ 主机 3.2 ms/call
- **`r235SP`**（dflash n-max 7 控制臂，带 `GGML_SPEC_SELECTOR_VEC=1`）: tg **102.51 / 89.44 / 131.08**、AL 5.55/4.22/6.38、**ms/轮 54.36 / 47.44 / 48.48（均值 50.09）**、门 `f3edac19...` ✓
  - `[RT]` target rounds=294、**enqueue 8.36 ms/call**、alloc 2.14、build 0.14 ⇒ 主机 **10.6 ms/call**（vs 无投机 3.2 ⇒ **多出的约 7.4 ms 几乎正好是 24/294 次重建摊出来的**：270 次约 4 ms + 24 次约 50 ms = 2.28 s vs 实测 2.46 s）
  - `[RT]` draft rounds=556（**2.04 次/轮**）、enqueue 1.65 + alloc 1.81 + build 0.12 = 3.58 ms/call ⇒ **7.3 ms/轮**

**★ 口径更正（重要）**: **llama-bench 的 `-p N` 绝对值不能直接当"一次 N-token 前向的成本"** —— 同一库同一天，`pp1 = 40.3 ms` 而服务路径无投机 = **23.4 ms/token**，差 **约 17 ms/次**（固定项，不随 n 缩放；若按 n 缩放则 pp8 会是 187 ms，与实测 52.6 矛盾）。⇒ **只有同框架内的差分可用**。

**⇒ 由此得到的每轮账（p1，54.36 ms）**: 8-token 目标 verify ≈ **36-43 ms**（两种推法: ① pp8 52.6 − 17 ms 偏移 = 35.7；② 轮时减去 draft 5.94 与 selector 约 1.5 与主机暴露约 10 = 约 37；③ 若把主机全部算作不重叠则 43）＋ draft 5.94 ＋ selector 约 1.5 ＋ 主机暴露约 10。
**⇒ 结论: 一轮的绝对主体 = 一次 8-token 目标前向**（而它比同模型 n=1 的 23.4 ms 贵 1.5-1.9 倍），其次才是主机（10 ms 暴露）与 draft（5.9）。

### R234 ★★ **n-token 前向成本曲线（llama-bench `-p 1,2,4,8,16,32,64` × `-d 0,8192`，同源 `-r 8`，CUDA_VISIBLE_DEVICES=0,1,2，`-ts 1/1/1`）**

| n | d0 t/s | **d0 ms** | d8192 t/s | **d8192 ms** |
|---:|---:|---:|---:|---:|
| 1 | 24.81 ± 1.06 | 40.3 | 23.70 ± 1.25 | 42.2 |
| 2 | 47.46 ± 2.03 | 42.1 | 45.88 ± 2.34 | 43.6 |
| 4 | 90.92 ± 3.76 | 44.0 | 86.66 ± 4.22 | 46.2 |
| 8 | 152.08 ± 5.69 | 52.6 | 142.49 ± 7.00 | 56.1 |
| 16 | 202.21 ± 10.00 | 79.1 | 194.55 ± 6.44 | 82.2 |
| 32 | 318.91 ± 12.04 | 100.3 | 299.13 ± 9.69 | 107.0 |
| 64 | 271.71 ± 3.46 | **235.6** | 264.29 ± 4.04 | **242.2** |

- n=1..8 拟合 **ms ≈ 38.5 + 1.8n**（pp2 预测 42.1 vs 实测 42.1；pp4 45.7 vs 44.0；pp8 52.9 vs 52.6）⇒ 每多一列只多 **1.76 ms（= 满权重流的 8%）**，**"MMVQ 每列都要重读权重"的担心远小于原先估计**（8 列总共只比 1 列贵 30%）。
- **★ n=64 处的分派拐点（首次直接量到，但量级要按每 token 算）**: n=64 时 235.6 ms vs n=32 的 100.3 ms（每次调用 2.35 倍），但**每 token 只有 3.68 vs 3.13 ms = +17%** —— 这正对应分派链在 `ne11 >= 64` 落到 `mul_mat_cublas`（每次调用全量反量化整张权重成 F16）。⚠️ 我一开始写成"断崖 2.35x"是**按次而非按 token 比**，已更正：它是**局部 +17% 的拐点**，到 n=512 又回落到 0.396 ms/token。（与 R216 的"反量化在带宽墙上、只能少做几次"一致。）
- **深度项极小**: 同 n 下 d8192 − d0 只有 +1.9 ~ +3.9 ms ⇒ 8K 深度上整条注意力/深度只值 3.5 ms/次（与 R229 的 FA 5.25 ms、R175 的 0.84 ms/token 封顶一致）⇒ **KV/注意力不是这条路的目标**。
- **★ 内核政策对照（同会话，同一 `-p 8,16 -d 8192 -r 8`）**: 默认 MMVQ `pp8 = 142.49 ± 7.00`（56.1 ms）vs **强制 MMQ（`GGML_CUDA_MMVQ_MAX_BATCH=4`，即 ne11>=5 走 MMQ）** `111.12 ± 6.68`（**72.0 ms，-22%**）vs **全 MMQ（`=0`）** `132.66 ± 16.86`（60.3 ms，-7%）⇒ **n=8 走 MMVQ 是对的**（老的 -31% 结论在当前基线上复现）；n=16 时 MMQ 略优（198.10 vs 194.55，+1.8%，噪声内）⇒ **调优只能在 MMVQ 内核内部做，换内核是负收益**。
- **★ 逐算子基线（R236，`test-backend-ops perf -o MUL_MAT -b CUDA0`，单卡，q8_0 m=4096 k=14336 = 58.7 MB）**: n=1 **82.10 µs（715 GB/s）**、n=2 84.18、n=3 87.82、n=5 94.55、**n=8 111.66 µs（526 GB/s）** ⇒ **n=8 只比 n=1 贵 1.36×**（与整模型 pp 曲线的 1.30× 一致）⇒ **"MMVQ 每列都要重读权重"不成立**（8 列总代价 +36%，不是 ×8），R234 曲线在 n=5 的小拐点与 VOLTA 分支把 nwarps 从 4 降到 2 重合（⇒ R237 正在试 nwarps=4）。
  ⇒ **但服务路径的整模型有效带宽只有 438 GB/s（22.07 ms/token），逐算子单卡却能到 715 GB/s** ⇒ 差额约 39% 来自"非矩阵乘"部分：**约 130 次 NCCL allreduce（每次约 50 µs ⇒ 估约 6.5 ms，占一次前向 30%）+ 注意力（8K 约 3.5-5 ms）+ 逐节点启动间隙**。⇒ **这才是"一次 8-token 前向"里除权重流之外最大的一块**（也解释了为什么"删主机 direct 调用"没用：它不是 AR）。
- ⚠️ 绝对值含 R235 那约 17 ms/次的 llama-bench 固定项 ⇒ **只引用同框架内的差**。

### R232 / R233 ★★ **selector 向量化: 内部探针 -0.33 ms/次（-37%）确认；端到端 -0.35 ms/轮 小于臂离散，未分辨 ⇒ 采用代码但不记采用链**

改法（服务器树 + 本地树均已落盘，env `GGML_SPEC_SELECTOR_VEC` 门控；补丁脚本 `wip-selector-vec.py`，逐位一致设计）:
① `work_topk` 的每位置 `std::vector` 换成定长数组；② 块嵌入转置成 `embd_T[d*n_tokens+i]`；③ `work_gate` 里 **8 条 lane = 8 个位置**的手工展开累加（每 lane 保持原有累加顺序与 `(s0+s1)+(s2+s3)` 组合）。

四臂（R232 顺序 A→B；R233 顺序 **B→A** 交叉验证；全部 `sha256 = f3edac19446ef641447f8391c71cb1b25a74055d662e9887890fa4a98602ca34` 门一致 ✓）:

| 臂 | 变体 | tg p1/p2/p3 | **ms/轮 均值** |
|---|---|---|---|
| R232A | 标量 | 100.40 / 88.14 / 129.43 | **50.91** |
| R232B | 向量 | 101.66 / 89.08 / 130.35 | **50.40** |
| R233B | 向量（先跑） | 102.09 / 89.14 / 130.74 | **50.17** |
| R233A | 标量（后跑） | 102.00 / 88.75 / 129.87 | **50.36** |
| R235SP | 向量 | 102.51 / 89.44 / 131.08 | **50.09** |

- **函数内部探针（可靠口径）**: scalar `gate = 0.84 / 0.84 / 0.89 / 0.88` vs vec `0.55 / 0.55 / 0.56 / 0.55` ⇒ **-0.33 ms/次（-37%）**；selector 三段合计 **1.52 → 1.21 ms/次（-20%）**；topk 0.71 → 0.60（在噪声内）。
- **整轮口径**: 标量均值 50.64（2 臂）vs 向量均值 50.22（3 臂）⇒ **-0.42 ms/轮（-0.8%）**，而**臂内离散 0.55 ms（1.1%）** ⇒ **方向一致但未分辨**。
- **判定: 采用 = B3**。（用户 2026-09-22 纠正过我一次：**"只要有一点点进步，这都算成功，都算记一笔"** —— 不能拿"离 150 还差得远"或"小于臂离散"当否掉的理由。此处原判"不记采用链"是错的，已改。）
  **B3 = B2 + 本改动**：tg **102.09 / 89.22 / 130.72**（三臂均值）、AL 5.55/4.22/6.38、ms/轮 **54.36 / 47.44 / 48.48（均值 50.22）** ⇒ 相对 B2（50.73）**-1.0%**；最好单臂 `r235SP` = 50.09。
  复现命令 = B2 的命令行（**R256 之后不再需要任何 selector env**）；要复现“旧标量路径”才加 `GGML_SPEC_SELECTOR_VEC=0`。服务器树保留 `/root/speculative.cpp.orig`；本地树补丁未提交（等用户批准），另有 `b3-selector-vec.patch`（5 393 B）做保护。
- ⚠️ 本地 Windows 树的补丁与服务器**逐字节相同**（两侧 md5 都是 `e5cad016df913513a5bee39656a66fa5`，orig = `99fd1e9f5975c80c79394d05ca8b8e64`）⇒ 将来用 `git archive HEAD` 同步会**丢掉它**（尚未提交，等用户批准）。

### R231 ★★ **分相括号不是稳定桶：同配置两臂的 `draft_decode` 相差 2.3 倍而 tg 相同 —— 选优化目标不能依据分相**

把同一配置两臂的**所有**分相探针并排（模型/配置/tg 都相同）：

| 相 | c0a | dev1 | 倍数 |
|---|---:|---:|---:|
| `draft_decode` | **6.03** | **13.67** | **2.27x** |
| `selector` | **5.19** | **2.78** | **1.87x** |
| `walk` | 0.07 | 0.09 | — |
| 中位 tg | 约 100 | 约 100 | 1.0 |

⇒ **时间在这些括号之间搬家**（大概率取决于异步工作落在同步点的哪一侧），**括号边界不是稳定量**。
⇒ **E16 那笔「target 36.86 + draft 6.13 + selector 5.16 + 其余 2.5」的分解，桶间边界不可靠** ⇒ **从分相里挑「最大的一块」来优化是不可靠的方法**（我在 R230 正打算这么做）。

**对照：函数内部的直接探针是稳定的** —— selector 内部三段在三个臂里为 **1.74 / 1.89 / 1.89 ms/次**。
**本轮另外读到两个内部探针**：
- `process: inject timing: n=256 | gather=0.36-0.78 copy=0.07 submit=1.44 wait=0.00 ms/call (layers=5 tok=8.73)` ⇒ **注入阶段约 1.9-2.3 ms/次，且只跑 5 层**；
- `draft ctx: n_eval=1 n_reused=0 t_eval=0.0 ms` ⇒ **该 perf 计数器在图上路径不累计，不可用**。

**⇒ 新增规矩（口径类，本会话第六次同族）**：**选优化目标只能依据「函数内部的直接探针」或「整轮总时长的差」；不得依据分相括号。**
⇒ 直接后果：R230 对 selector 的**单次 1.8 ms 仍然可靠**，但「每轮省 1.1-3.3 ms」的折算**必须用整轮总时长来验**（不能看 selector 那一格是否变小）。

**判定**：不涉及采用/否证；**不改动采用链**（基准仍是 **B2**：tg 100.95/88.30/129.12、AL 5.55/4.22/6.38、ms/轮均值 50.73）。
### R230 ★★ **selector 的 5.16 ms/轮 拆开了：每次调用稳定 1.8 ms，差别在「每轮调几次」；两段热点都是未向量化的标量代码**

**① 计时括号已定位**：`speculative.cpp:1514`（`st_t1`）→ `:1535`（`st_t2`）之间**只有一行** `build_dflash2_selector_cpu(cand, unary, gate)`（`:1533`）。
⇒ **`[SPEC] selector=` 就是这个函数的墙钟**，而该函数自带内部三段探针（每 32 次调用打一行），于是可以交叉核对。

**② 同臂交叉核对（关键：内部合计 ≈ 1.8 ms 恒定，而 `selector=` 逐臂不同）**

| 臂 | `[SPEC] selector=` | 内部三段合计 | 差 |
|---|---:|---:|---:|
| c0a | **5.19** | 0.06 + 0.67 + 1.01 = **1.74** | 3.45 |
| t5b | **2.04** | 0.11 + 0.71 + 1.07 = **1.89** | 0.15 |
| dev1 | **2.78** | 0.07 + 0.74 + 1.08 = **1.89** | 0.90 |

⇒ **差值 = 「每轮调用次数」的差异，不是单次成本差异**：单次稳定 **1.8 ms**；c0a 每轮约 **3 次**（3 x 1.74 = 5.2 ✓），t5b 约 1 次。
⇒ **E16 账里那笔「selector 5.16 ms/轮」= 约 3 次 x 1.8 ms** ⇒ 以后引用必须带「次数」，否则会把结构差异误读成实现开销。

**③ 单次 1.8 ms 的成分（`rank=256`, `n_tokens=8`, `n_embd_dec=5120`）**：
- **`gate` 约 1.05 ms**：`rank x n_tokens x n_embd_dec` = **10.5M FMA**，8 线程 ⇒ **约 1.2 GFLOPS/线程** ⇒ **明显是标量、未向量化**；
- **`topk` 约 0.71 ms**：`n_tokens x n_vocab` = **2M 次扫描**（且每个 token 在循环内 `std::vector` 分配 `ids`/`vals` 各一次，`:1214-1215`）；
- `fetch` 0.06 ms（`llama_get_logits_ith` x 8）—— **便宜，不是问题**。

**④ 安全约束（**做优化前必须守住**）**：源码注释明确写了这两段**刻意保持累加顺序以做到逐位一致**（`:1203-1205` *same order, same operands*；`:1261-1262` *bit-identical to the position-major form*；`:1268-1269` *reduction order is fixed*）。
⇒ **正确的向量化方向不是「点积内并行」，而是「跨 token 并行」**：`gate` 的 `n_tokens`(=8) 个点积共享同一行 `row_k` ⇒ 用 8 宽向量让**每条 lane 各自按原顺序累加自己那个 token** ⇒ 每个点积的累加顺序不变、**逐位一致**，同时把 `row` 的加载摊薄 8 倍。
⇒ 预期：`gate` 1.05 -> 约 0.2-0.3 ms；`topk` 去掉内层分配 + 阈值扫描向量化 ⇒ 约 0.4 ms ⇒ **单次 1.8 -> 约 0.7 ms** ⇒ 按每轮 1-3 次计，**每轮省 1.1-3.3 ms（2-6.5%）**。⚠️ 全部为估算，必须实测。

**判定**：不涉及采用/否证（尚未改码）；**不改动采用链**（基准仍是 **B2**：tg 100.95/88.30/129.12、AL 5.55/4.22/6.38、ms/轮均值 50.73）。
### R229 ★★ **拿到「验证批（n=8）」的逐算子账 —— 并且它推翻了 E16 的「非矩阵乘 9.4 ms」**

**① 新能力（本轮建立）**：`tests/test-export-graph-ops.cpp:198` 用 `n_tokens = min(n_ctx, n_ubatch)` ⇒ **`test-export-graph-ops -m <gguf> -o /tmp/ops8.txt -ub 8` 就能导出验证批形状**，配 `test-backend-ops perf --test-file /tmp/ops8.txt` 得到 n=8 的逐算子耗时。
（导出文件同时含 **n=1 与 n=8** 两套形状；`perf` 输出里**名字与耗时在不同行**，解析要按行配对 —— 我第一版解析器就是错在这里。）

**② n=8（验证批）的账**

| 类别 | 合计 | 备注 |
|---|---:|---|
| **MUL_MAT** | 2867 µs | 其中 **`result_output`（LM head，q8_0[5120,248320]×f32[5120,8]）= 2165.60 µs**（1.27 GB 权重 ÷ 2.17 ms ⇒ **约 586 GB/s**，带宽受限） |
| **FLASH_ATTN_EXT** | **5249.82 µs** | **单个** op（`ne=[256,24,8,1]`，n_q=8；n_kv 未知） |
| MUL / GET_ROWS / ADD / RMS_NORM / CONT / SILU / SSM_CONV / SIGMOID / SET_ROWS / SOFTPLUS | **合计约 40 µs** | **基本为零** |

**③ 结论一：E16 的「非矩阵乘 9.4 ms」是错误归因。** 真实账是：**逐元素类算子总共约 0.04 ms**；目标阶段**矩阵乘主导** —— 按层数乘开 FFN（65 层 ×(gate+up 162 + down 170 µs) ≈ **21.6 ms**）+ 注意力/GDN 投影约 8 ms + LM head 2.2 ms ≈ **32 ms**，与 E16 实测的 **28.44 ms** 同量级（差 12%）。
⇒ **选项 (a)「去拆那 9.4 ms 非矩阵乘」没有靶子**：那 9.4 ms 并不存在（它是把 FA 与逐元素混在一起的反推残差）。

**④ 结论二：目标阶段剩下的唯一未知量是 FA 的真实份额。** 导出里 5249.82 µs 是**某个未知 n_kv**下的单例；若乘 16 层得 **84 ms**，与实测 28.44 ms 矛盾 ⇒ **该绝对值不可用于每轮账**（R203 的警告在这一格上成立）。
⇒ 要定它，必须先知道真实轮内的 `n_kv` 与 n_q=8 时 FA 走哪条路径（[FAK]：`n_q>1` ⇒ MMA_F16 或 TILE），这需要在**真实 KV 长度**下量 —— 而 `-ub 8` 的导出不带 KV 填充。

**⑤ 顺带两条可入账的硬数**：
- **LM head 每轮 2.17 ms**（1.27 GB 权重、586 GB/s）—— 它是单卡上一次 1.27 GB 的读取，**占每轮 4.3%**；除非减少需要 logits 的位置数，否则没有空间。
- **`ffn_gate`/`ffn_out` 在 n=8 上是 162/170 µs**（权重 89 MB → 约 550 GB/s）⇒ 与「n=8 时矩阵乘 559 GB/s」一致，**再次印证解码侧矩阵乘已贴带宽**。

**判定**：不涉及采用/否证；**不改动采用链**（基准仍是 **B2**：tg 100.95/88.30/129.12、AL 5.55/4.22/6.38、ms/轮均值 50.73）。
### R228 ★★★ **判决（同库同形状，内核级）：宽 ncols 走 MMVQ 不是更快，而是慢得多 —— R224 的假设被证伪，AL 路线彻底关闭**

补丁已上机并通过**单元门**：`GGML_CUDA_MMVQ_MAX_BATCH=16 test-backend-ops test -o MUL_MAT -p type_a=q8_0` ⇒ **n=9 确实被跑到**、`50/50 tests passed`、`Backend CUDA0: OK`；不带 env 同样 50/50（**默认路径无回归**）。
随后用**同为 `m=4096, k=14336` 的固定形状**做 env 开/关对照（`perf --test-file` 走不通 —— perf 用的是专用列表 `make_test_cases_perf()`；已临时把该列表的 `bs` 从 `{1,2,3,4,5,8,512}` 扩到含 `9,12,15,16`）：

| n | MMQ（env 关） | MMVQ（env 开） | 差 |
|---:|---:|---:|---:|
| 1–8 | （两臂都是 MMVQ） | — | ±1–2% = **噪声本底** |
| **9** | 162.75 µs | 174.74 µs | **+7.4%** |
| **12** | 158.49 µs | 228.40 µs | **+44.1%** |
| **15** | 166.93 µs | 276.50 µs | **+65.6%** |
| **16** | 159.58 µs | 294.04 µs | **+84.3%** |

**机制（与源码注释一致）**：**MMQ 的耗时几乎与 n 无关**（162.75 / 158.49 / 166.93 / 159.58 µs —— 权重只读一遍、带宽受限），而 **MMVQ 随 n 线性涨**（174.74 → 294.04 µs —— *mvq redoes that per column*）。

**⇒ 三条结论**：
1. **R224 的假设证伪**：那条 AL 路线**不是**因为「缺实例化」才关的 —— 即使实例化补齐、MMVQ 可用，它在 n=15 上仍比 MMQ **慢 1.7 倍**。
2. **我上一轮说 R202「关错了」是错的**：R202 的结论在**效果**上正确（只是机制归因不完整 —— 真正症状是 `GGML_ABORT`，但即使不 abort 也没有收益）。**BLK14 的「路线关闭」现在有了第二个独立理由。**
3. 因此**不必**再做草稿语义侧的旁路（不必绕 `speculative.cpp:1018-1031` 的 `block_size-1` 夹取）—— 上游验证阶段本来就已经在更快的那条路上（MMQ）。

**处置：不采用，全部还原（R228b 已验证）**：`mmvq.cu` = `19c984c9…`、`mmvq.cuh` = `52288c1e…`、`tests/test-backend-ops.cpp` = `0d2da827…`（三者均回到 `.orig` 的 md5）+ `rm` 对应 `.o` + 重建；`PATCH_MARKER_LEFT=0`、装好的库里 `LIB_MARKER_R224=0`、`LIB_MARKER_HMMA=0`、`BASE_MD5` 未变。
（顺带第三次印证：**该 lib 的构建不是逐位可复现的** —— patch 前 `ad6d7fe6…`、还原后 `1690d33c…`，两次源码相同而 md5 不同。）

**★ 两条方法教训（本轮连续踩到，已固化）**：
- **「单元门通过」不等于「假设成立」**：补丁数值上完全正确（n=9 对数通过、默认路径无回归），但**收益方向是反的**。⇒ **正确性门与收益前提必须分成两次独立测量**，不能因为前者通过就投入后者。
- **`test-backend-ops` 的 `perf` 与 `test` 用两套 case 列表**（`make_test_cases_perf()`）⇒「`test` 里有的形状」在 `perf` 里**可能根本不存在**。我先后两次 perf A/B 都是空转（第一次 n≤8 两臂同为 MMVQ，差异全是噪声；第二次根本没有 n≥9 的 case）。⇒ **做 perf 对照前，先确认目标形状真的出现在输出里。**

**判定**：**不采用**（负结论，已还原）；**不改动采用链**（基准仍是 **B2**：tg 100.95/88.30/129.12、AL 5.55/4.22/6.38、ms/轮均值 50.73）。
### R226 **验收门的覆盖度核查：单元测试只能覆盖到 case 9，case 10..16 必须靠端到端**

在上机之前先把「先对数」这一步的有效性查清（否则会得到一个**空转的门**）。查 `tests/test-backend-ops.cpp`：

| 位置 | 覆盖 | 能否验证新代码 |
|---|---|---|
| `:9912` `#if 0` 内的 Q8_0 n 扫描（n=32..4096） | **被禁用** | ✗ |
| `:9900` `for (int64_t n : {1,7,8,9,16,127,128,511,512})` | 只生成 **F32 x F32**（`:9901`）⇒ 不走 MMVQ | ✗ |
| **`:9865` `for (int i = 1; i < 10; ++i)`**（在 `for type_a : all_types` 内） | **Q8_0 覆盖 `n = 1..9`** | **✓ 覆盖 case 9** |
| Q8_0 且 `n` 在 [10,16] | **无任何用例** | ✗ |

⇒ **两层门（这就是最终验收方案）**：
1. **单元**：`GGML_CUDA_MMVQ_MAX_BATCH=16 test-backend-ops test -o MUL_MAT -p type_a=q8_0` —— 注意**必须带 env**，否则 `n=9` 会因 `9 <= 8` 为假而走 MMQ，等于没测；覆盖 **case 9**（验证分发路径 + 断言放宽）。
2. **端到端**：harness 的 BLK14 配置（`n_max=14` ⇒ 15 token 验证）—— 这是**唯一**覆盖 `n ∈ [10,16]` 的门，且该配置的门值必须重新确立。

⚠️ **必须写进结论的限制**：`case 10..16` **没有单元测试覆盖**，只有端到端一条证据链。若端到端出现数值异常，定位会退化到「二分关闭 case」而不是直接指向某一列。

**判定**：不涉及采用/否证（尚未上机）；**不改动采用链**（基准仍是 **B2**）。
### R225 **MMVQ 宽 ncols 补丁：已备好并通过「副本干跑」验证（四处 bug 在上真树之前就被抓出）**

补丁脚本 `wip-mmvq-wide-ncols.py`（字节模式读写、CRLF 感知、锚点唯一性自检）。三处编辑：
1. `mmvq.cuh`：新增 `MMVQ_MAX_BATCH_SIZE_WIDE 16`（保留原行尾注释）；
2. `mmvq.cu:1110`：`GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE)` → `..._WIDE`（**不加这条，新 case 一样 abort**）；
3. `mmvq.cu`：在 `case 8` 块之后、`default:` 之前插入 `case 9..16`，每个用 `if constexpr (type == GGML_TYPE_Q8_0)` 收窄以避免 8x20 个内核实例化。

**干跑（在 `/tmp/mmvq-dryrun` 的副本上，未碰真树）抓出四处问题，全部是我自己脚本的 bug**：
| # | 症状 | 根因 |
|---|---|---|
| 1 | `ANCHOR_MISS found=2` | `default: GGML_ABORT("fatal error");` 在文件里出现 **2 次** ⇒ 脚本**正确拒绝执行**而不是乱改 |
| 2 | diff 显示 **1592 行全变** | 文本模式读写 + 通用换行 ⇒ **整个文件 CRLF 被改成 LF** |
| 3 | 生成的 `GGML_ABORT(wide ...)` **没有引号** | 我为躲转义把引号丢了 ⇒ **编译必然失败** |
| 4 | `.cuh` 里原行尾注释被劈到新行 | 锚点只取了 `#define ... 8`，没含尾部注释 |

**修复后干跑验收**：插入顺序正确（`case 8` → `case 9..16` → `default:`）、`GGML_ABORT` 引号正确、原 `default` 未被触碰、**两个文件都仍是 CRLF**、其余内容 md5 不变。
⇒ **补丁就绪，等机器空出即可上**（先备份 `.orig` → 打补丁 → `rm -f` 目标 `.o` 再编 → **先 `test-backend-ops test -o MUL_MAT -p type_a=q8_0` 在 ncols∈{9,15} 对数** → 再 harness A/B）。

**★ 可固化的方法（新增）**：**任何源码补丁在上真树之前，先在 `/tmp` 的副本上干跑一遍** —— 本轮四处 bug（锚点不唯一、行尾被改写、字面量丢引号、注释被劈）**没有一处会被「读一遍脚本」发现**，但干跑全部暴露，而且零风险。
配套的三条细则：① 脚本必须**字节模式**读写（否则 Python 通用换行会静默改写 CRLF）；② 锚点必须**自检唯一性**，不唯一就退出而不是取第一个；③ 插入的新行必须用 **`\r\n`**，与文件一致。

**判定**：不涉及采用/否证（尚未上机）；**不改动采用链**（基准仍是 **B2**）。
### R224 ★★★ **「MMVQ 不支持 9-16 批量」是错的：它不是原理性限制，而是一个缺失的 `switch` 分支 —— 被关闭的 AL 路线应当重开**

**① 根因（读码，可核）**
`ggml/src/ggml-cuda/mmvq.cu:1199` 的 `mul_mat_vec_q_switch_ncols_dst` 只实例化了 **`case 1..8`**，而：
```cpp
default:
    GGML_ABORT("fatal error");
    break;
```
⇒ 一旦 `GGML_CUDA_MMVQ_MAX_BATCH=16` 让 `ncols_dst` 走到 9..16，**进程直接 abort**。
⇒ **这正是 R202 当时看到的「tg=0 / PARSE_FAIL」**（我当时把它记成「内核不支持该批量」并据此关闭了整条 AL 路线）。**实际是 abort，不是算错、也不是不支持。**

**② 补 case 所需要的一切都已存在**（两张元数据表对 ncols>8 都有合理默认）：
- `calc_rows_per_block`（`:603`）：`default: return 1`；
- `calc_nwarps`（`:459`）：GENERIC 分支 `default: return 1`；**Volta 分支（`:475`，即 C4 调优那处）也是 `default: return 1`**。
⇒ 新增的 `case 9..16` 会得到 `nwarps=1, rows_per_block=1` 的合法配置（未调优，但合法）。

**③ 这个改动「默认零行为改变」**：`use_mul_mat_vec_q` 要求 `src1->ne[1] <= MMVQ_MAX_BATCH_SIZE`（`ggml-cuda.cu:1941-1942`），而 env 不设时上限恒为 **8** ⇒ **新分支默认不可达** ⇒ 补上它是**构造上安全**的。

**④ 为什么值得做（这可能是唯一还能抬 AL 的杠杆）**
E16 已定论：**GPU 串行地板 42.2 ms/轮 ⇒ AL 5.55 下 tg 上限 132**，所以 **150 必须靠抬 AL**；而 AL 的两条路当时都被关闭（BLK14 加长掩码块、R202 放宽 MMVQ 上限）。现在第二条的关闭理由是**错的**。
粗估（⚠️ 用 BLK14 的实测边际代价 1.24 ms/verify-token 外推，非测量）：若 15-token 验证回到 MMVQ 的每 token 代价，target 阶段从 `37.01 + 18.1` 变成 `37.01 + 8.7` ⇒ 轮时约 59.9 ms，配 BLK14 实测的 AL 6.47/6.06/8.24 ⇒ **tg 约 108 / 101 / 140**（现基准 100.95/88.30/129.12）。**p3 一路已接近硬指标**。

**⑤ 实施要点（若采纳）**
- 在 `mul_mat_vec_q_switch_ncols_dst` 补 `case 9..16`（每例约 7 行，照抄 `case 2`/`case 8` 的形状）；
- ⚠️ **编译代价**：该 switch 在**每个量化类型**上各实例化一次 ⇒ 8 个新 case × ~20 个类型 = 上百个新内核实例化。**缓解**：在新 case 里用 `if constexpr (type == GGML_TYPE_Q8_0)` 包住 launch（被丢弃的分支不实例化），只给我们真正需要的类型付代价；
- 用 env `GGML_CUDA_MMVQ_MAX_BATCH` 做同源 A/B（该 env 已存在于我们的树，R202 加的）；
- 验收：先 `test-backend-ops test -o MUL_MAT -p type_a=q8_0` 对数（ncols ∈ {9,15} 必须与 CPU 参考一致），再上 harness 走 BLK14 的 n_max=14 配置；**门必须在该配置上重新确立**。

**⑥ 这条同时是对 R202 的更正**：R202 的结论「内核不支持该批量 ⇒ AL 抬升的两条路都关闭」应改为「**内核缺实例化 ⇒ 可修**」。

**判定**：不涉及采用/否证（尚未改码）；**不改动采用链**（基准仍是 **B2**）。**待用户批准后实施**（涉及 `mmvq.cu` dispatch 与内核实例化规模，按项目红线「大改动先停下问用户」）。
### R223 **定量推断：每 ubatch 固定开销 C ≈ 79.7 ms；并预登记「`-b 8192 -ub 8192` 只值 4-5%」与预填充 TP 扩展性的预期**

**① 从 R222 的两条同日律反解 `C`**（模型 `a(ub) = a_∞ + C/ub`，因为固定开销总次数 = n/ub ⇒ 每 token 承担 `C/ub`）：

| 量 | 值 |
|---|---:|
| `a(512)` | 4.906e-4 s/token |
| `a(2048)` | 3.738e-4 s/token |
| **`C`（每 ubatch 固定开销）** | **79.7 ms** |
| `a_∞`（渐近线性代价） | 3.349e-4 s/token |

**② 这个 79.7 ms 里已知的成分只有一部分**：R216 实测的全量反量化 = 每卡 27.9 GB/ubatch ⇒ 约 **32 ms**。
⇒ **剩下约 48 ms 成分不明**（候选：cuBLAS workspace 的 alloc/free churn —— 每 ubatch 约 390 次 `src0_alloc`/释放；或 FA 在 `n_q=512` vs `2048` 下的 tile 效率差）。**这是一条尚未被任何节点覆盖的线索**，但**没有独立证据，不下结论**。

**③ 有界预测（预登记）**：即使把 `-b` 也提上去让 `-ub 8192` 真正生效，收益上限 = `C·(1/2048 − 1/8192)·n`：
- n = 65536 ⇒ **+1.91 s / 36.6 s = +5.2%**
- n = 131072 ⇒ **+3.83 s / 97.4 s = +3.9%**
⇒ **`-b 8192 -ub 8192` 只值 4-5%**，属「便宜就做」而不是「头寸」。
⚠️ 前提：把 Δa 全部归因为「每 ubatch 固定开销」。若它其实来自注意力在 `n_q=512 vs 2048` 下的效率差，则此预测不成立 —— 但**无论机制如何，`-ub 2048` 相对 512 的 1.246×/1.213× 是实测**。

**④ 预登记：预填充 TP 扩展性的预期**（即将跑 TP3/TP4/TP6 的 `-p 8192,32768`）
- **线性外推**：pp32768(TP4) = 2148 × 4/3 = **2864**；pp32768(TP6) = 2148 × 6/3 = **4296**。
- **我的预期：明显低于线性**，理由：预填充时每个 PARTIAL 边界的 AR 要搬 `n_tokens × hidden` 的激活（`-ub 2048` 下每次约 2048×5120×4 B = **42 MB**），而 46 个边界/ubatch ⇒ **约 1.9 GB 的 AR 流量/ubatch**；且 4/6 卡里必然有跨 NUMA 岛（0/1/2 在 NUMA0，3/4/5 在 NUMA8）⇒ 跨岛走 `SYS`。
- **判读**：比值 ≥ 1.25 ⇒ 预填充**值得加卡**（与解码的「加卡不买时间」相反，因为预填充能摊薄 AR）；比值 ≤ 1.05 ⇒ 加卡对预填充也无用。

**判定**：不涉及采用/否证（数字待测），**不改动采用链**（基准仍是 **B2**）。
### R222 ★★ **ub×长度全表（同日同库，只差 ub）+ ub 杠杆的真实量级：64K 1.25× / 128K 1.21× / 256K 约 1.18×**

全部在**清库重建后的库**上、TP3、显式 `CUDA_VISIBLE_DEVICES=0,1,2`、升序、同一进程外各自加载：

| n | `-ub 512` | `-ub 2048` | `-ub 8192` | **ub 效应（512→2048）** |
|---|---:|---:|---:|---:|
| 65536 | 1436.46 ± 0.71 | **1790.23 ± 0.88** | **1792.23 ± 0.08** | **1.246×** |
| 131072 | 1109.18 ± 0.36 | **1345.26 ± 0.55** | （运行中；预期≈2048，见下） | **1.213×** |

**① `-ub 8192` 与 `-ub 2048` 逐位相同（+0.11%，噪声内）—— 因为 `n_ubatch` 被 `n_batch` 钳制。**
llama-bench 的 `-b` 默认 **2048**，而 `n_ubatch ≤ n_batch` ⇒ **`-ub 8192` 被静默钳到 2048**。
⇒ 这既解释了「没有额外收益」，也指出下一步：**要再往上必须连 `-b` 一起提**（待测：`-b 8192 -ub 8192`）。
⇒ 同时提醒：**任何『把 ub 加大』的建议都必须说明 `-b` 的钳制**，否则在生产上会以为设了 8192 就生效。

**② 用同日数据重拟合两条律（与 R219 的预登记版本对比）**

| 律 | a（s/token） | b（s/token²） | 出处 |
|---|---:|---:|---|
| `-ub 2048` | **3.738e-4** | **2.819e-9** | 由 **65536 + 131072** 两点（R222） |
| `-ub 2048`（旧） | 3.742e-4 | 2.678e-9 | 由 8192 + 32768 两点（R219 预登记） |
| `-ub 512` | **4.906e-4** | **3.136e-9** | 由 65536 + 131072 两点（R222） |
| `-ub 512`（旧，已作废） | 4.544e-4 | 6.583e-9 | 由 **Sep 20 的 p0.log** —— **陈旧口径，不得再用** |

⇒ **两条独立拟合（8K/32K 与 64K/128K）给出的 `-ub 2048` 律几乎相同（a 差 0.1%、b 差 5%）** ⇒ **同一条律从 8K 一路成立到 128K**（这是 R221「无拐点」的第二次独立确认）。
⇒ **ub 的影响主要落在线性项：a(512)/a(2048) = 1.312，而 b(512)/b(2048) = 1.112** —— 与「每 ubatch 一份固定开销（含 R216 那 32 ms/ubatch 的全量反量化）」的机制方向一致。

**③ 外推到用户的 256K（⚠️ 仍是外推，见④）**
- `-ub 2048`：t(248901) = **267.7 s → 930 t/s**
- `-ub 512`：t(248901) = **316.4 s → 787 t/s**
⇒ **ub 在 256K 上的效应只有约 1.18×**（比 64K 的 1.246× 更小），因为 ub 敏感的那部分随长度被二次项稀释。
⇒ **所以「提高 ub」是一个真实的、零改码的杠杆，但量级是 1.2× 量级，不是我 R218/R219 一开始说的 1.6×。**（那次是拿今天的数据去比 Sep 20 的 p0.log，已作废。）

**④ 一个必须解决的矛盾：那个「256K = 370 t/s」的旧点比今天的 `-ub 512` 律低 2.1×**
今天的 `-ub 512` 律外推到 248901 给 **787 t/s**，而已记录的实测是 **370 t/s** ⇒ **该旧点与今天的数据不可比（同 `p0.log` 一样是 Sep 20 时代的口径）**。
⇒ **正在跑重测（R222 已排队）**：`-p 262144 -r 1` × `-ub 2048 / 512`，用今天的库直接替换那个陈旧点。**在结果出来之前，不得再用「370 t/s」做任何推理。**

**⑤ 顺带修正 R220 的「同库离散 0.8%」**：那是 **8192/32768 两个短点**上的离散；**长点极其可复现** —— `pp65536@ub2048` 我这次 **1790.23** vs 子代理上次 **1790.92**（**0.04%**）、`pp131072` **1345.26** vs **1343.31**（0.15%），而且那两次用的还是不同的库（清库前/后）。⇒ **短点松散、长点紧**，引用离散要带上长度。

**⑥ 适用性说明（避免误用）**：本表全部是 `-n 0` 的**纯预填充**测量 ⇒ **`-ub` 是预填充专用杠杆，不影响解码**（解码批只有 1-8 个 token）。⇒ 对生产而言它是「改一个启动参数、只动 TTFT、不动 decode」的低风险项 —— 但**显存与门值两条仍须在生产配置上实测**（R219 的约束不变）。

**判定**：不涉及采用/否证（`-ub` 的收益尚未在 harness 端到端上验），**不改动采用链**（基准仍是 **B2**）。
### R221 ★★ **长度扫描跑到 131072：无拐点，两点模型全程只差 1.6-2.6%（R214 的机制变化彻底出局）** + ffn_down 的 cuBLAS 新头寸

**① 长度扫描（TP3、`-ub 2048`、`-r 3`、升序、显式 `CUDA_VISIBLE_DEVICES=0,1,2`）**

| test | 实测 t/s | 反推 s/pass | 两点模型（M1）预测 | 偏差 |
|---|---:|---:|---:|---:|
| pp8192 | 2503.29 ± 5.68 | 3.273 | — | — |
| pp32768 | 2147.72 ± 1.74 | 15.257 | — | — |
| pp65536 | 1790.92 ± 0.83 | 36.594 | 1819 | **−1.6%** |
| **pp131072** | **1343.31 ± 0.37** | **97.57** | 1379 | **−2.6%** |

⇒ **8192→131072（16 倍）吞吐只降到 1/1.864，log-log 斜率恒为 −0.224，两点模型全程只高估 1.6-2.6%** ⇒ **没有拐点，与「平滑幂律」一致** ⇒ **R214 的「32K→256K 机制变化」彻底出局，主因就是 ub 口径差（R218/R220）**。
⚠️ 口径：这批数是用**还原源码但未真正重编**的库跑的（md5 `89fe01bd…`，与对照行为等价性由 R215 的 ABBA 实测 −0.30%/−0.19% 背书）；重建后如需严格口径可复核。

**② ★ 新头寸（无任何节点覆盖过）：`ffn_down` 形状在 cuBLAS 里比其它形状差 ~33%**
子代理的独立 CUDA 微基准（`cublasGemmEx`，C32F/C16F 两次都给）：

| 形状 | TFLOPS |
|---|---:|
| W[17408,5120]×X[8192,5120]（ffn_gate/up） | **103.0 / 101.3** |
| W[6144,5120]×X[8192,5120]（QKV） | 101.5 |
| W[4096,14336]×X[512,14336] | 96.2 / 82.0 |
| **W[5120,17408]×X[8192,17408]（`ffn_down`）** | **只有 67.8** |

⇒ **`ffn_down` 形状比其它形状差约 33%**，而它占 GEMM FLOPs 约 **21%** ⇒ 折算端到端约 **4%**。**这是一个还没被任何节点覆盖过的头寸**，且属于「换 cuBLAS 算法/heuristic」类（不是内核改写）⇒ 成本低。
⇒ 已立为待查项（节点 `FFNDOWN`）。⚠️ 注意口径：这是**单形状微基准**，尚未进真实图。

**③ 两条已固化进 §4 的坑（来自子代理自查）**
1. **还原源码后 `cmake --build` 会静默跳过该 TU**：`cp -a`（保 mtime）还原 ⇒ `.cu` 比 `.o` 旧 ⇒ make 什么都不编，而 `BUILD_RC=0 / ERRORLINES=0 / BUILTTARGET=115` 全都「正常」。**这是 §4.28 的反方向变体**（§4.28 是 mtime 刷新导致多编，这里是 mtime 保留导致**漏编**）⇒ **规矩：还原后必须 `rm -f` 目标 `.o`；验收看二进制标记串 + libdir md5，不看 BUILD_RC。**
2. **发现负结论后必须立刻停掉「自动续跑」的长链**：那条链设计成「还原→重建→四个扫描串行」，结论一出后面三段全是浪费 + 污染风险。**规矩：长链在每个阶段前重查一次「这个阶段还有意义吗」，或不要把无关键串成一条。**

**④ 污染窗口已核，本轮的 R221 数据不受影响**：子代理那条链的窗口 = `build-restore.log` mtime **23:01:01** → `restore-scan.out` mtime **23:11:27**。
我的 R221 链条 23:07:55 启动后先 `pgrep llama-bench` 等到空（`IDLE_AFTER_TICKS=15` ⇒ 约 23:11:40 才通过），**之后才重建并启动扫描**（rebuild 完成、sweep 于 23:14 左右启动）⇒ **全部落在窗口之外** ✓。
而 `p0.log` 的 ub=512 点（1492.27 / 1128.98）是 **Sep 20 22:10** 的，同样不受影响。

**⑤ 子代理交付状态**：改动只落在 `ggml-cuda.cu`（md5 轨迹 `e9e82864`(原始) → `9881f8bc`(有转义 bug，编译失败) → `392469784b`(修好) → **已还原 `e9e82864`**）；**唯一残留动作是清库重建，已由主线完成**（见下）。

**⑥ 主线清库重建的结果（R221 执行）**：`rm -f` 那个 `.o` 后重建 —— `BUILD_RC=0`、`ERRORLINES=0`、**`OBJ_MARKER=0`**、**`LIB_MARKER=0`**、`libggml-cuda.so` md5 = **`ad6d7fe6975981b9ae191a8360ba118a`**、`libggml-base.so` 仍 `a8de7c48…`（未变）。
⚠️ **注意：md5 没有回到 `dfb838b3199e7cd1e9f4fab861e11557`** —— 说明该 lib 的构建**不是逐位可复现**的（同一份源码两次构建得到不同 md5）。⇒ **规矩：以后「对照库」的判据用「四库 md5 + 标记串 + 同源构建流程」，不能指望 md5 逐位复现。**

**判定**：不涉及采用/否证；**不改动采用链**（基准仍是 **B2**：tg 100.95/88.30/129.12、AL 5.55/4.22/6.38、ms/轮均值 50.73）。
### R220 ★★★ **判决命中：ub 归因成立（R214 的「机制变化」作废）** + 一次「空操作重建」的根因

**① 预登记的判决点打中了**：`llama-bench -p 8192,32768,65536,131072 -ub 2048 -r 3`（3 卡 tensor）实测

| n | 实测（`-ub 2048`） | M1 预测 | 同长度 `-ub 512` 实测 | **ub 效应** |
|---|---:|---:|---:|---:|
| 8192 | 2503.29 ± 5.68 | 2524 | 1967（M5 预测） | — |
| 32768 | 2147.72 ± 1.74 | 2165 | **1492.27**（p0.log） | **1.45×** |
| **65536** | **1790.92 ± 0.83** | **1819（偏 −1.6%）** | **1128.98**（p0.log） | **1.59×** |
| 131072 | （运行中） | 1379 | 759（M5 预测） | — |

⇒ **M1 在 65536 上预测 1819、实测 1790.92（−1.6%），而 M5 预测 1129 与实测的 ub=512 点 1128.98 完全吻合** ⇒ **两个 ub 各自的经验律都成立，R214 那个「32K→256K 有机制变化（注意力拐点）」作废** —— 它就是 **ub 口径差**。
⇒ **同长度上 ub=512 vs 2048 的效应随长度增长：32768 是 1.45×、65536 是 1.59×。** 而**生产服务与 harness 都没有 `-ub`（默认 512）** ⇒ 这条对用户真实的 256K 场景是**零改码、只改一个启动参数**的杠杆（R219 的三条约束照旧：显存 / 必须在**该配置上重新确立门值** / 解码不回退）。

**② 口径警告：同库同配置的 run-to-run 离散约 0.8%，不是 0.04%**
这一次同一个 `89fe01bd…` 库给出 pp8192 = 2503.29、pp32768 = 2147.72，而 R215 那次同库给的是 2523.66 / 2163.16、官方基线 2524.55 / 2164.72 ⇒ **本轮的 −0.8% 是过程/状态漂移**（可能是「一次进程里连跑 4 个长度」的影响）。
⇒ **判决仍然有效**（1819 vs 1129 差 58%，远超 0.8% 噪声），但**以后引用「同库复现」时要按 0.8% 这个尺度读，不要再用 0.04% 当离散**。

**③ 一次「空操作重建」的根因（mtime 家族的新一面）**
子代理从 `.orig` 还原 `ggml-cuda.cu` 后重建并安装，输出 `BUILD_RC=0 / ERRORLINES=0`，但：
- **目标文件 `.o` mtime = 22:54，而还原后的 `.cu` mtime = 21:37**（保 mtime 的复制把时间戳带回来了）⇒ **make 认为 .o 已是最新，直接跳过重编**；
- 证据：`strings <那个 .o> | grep -c GGML_CUDA_SM70_HMMA_Q8` = **2**，且 `libdir-instr/libggml-cuda.so.0.24.0` md5 仍是 **`89fe01bd…`**（= 特性构建），不是对照的 `dfb838b3…`。
⇒ **规矩（新增）**：**「还原源码」必须同时保证它比目标文件新** —— 要么别用保 mtime 的复制（`cp -p`/`cp -a`），要么还原后 `touch`，要么**直接 `rm -f` 那个 `.o`**（§4.25 的老办法）。
  这是 §4.28（「`git archive` 解包刷新 mtime ⇒ 全量重编」）的**同一枚硬币的另一面**：**mtime 被刷新会导致多余的重编，被保留则会导致缺失的重编 —— 两个方向都会静默出错。**
⇒ **修复（待办，必须在扫描跑完后做，§4.37：替换库会杀死运行中的测量）**：`rm -f` 该 `.o` -> 重建 -> `cp -a build-instr/bin/. /root/libdir-instr/` -> 核对 `strings lib | grep -c` == **0** 且 md5 回到 **`dfb838b3199e7cd1e9f4fab861e11557`**。

**④ 对已有测量的影响**：本轮的 1790.92 等数字取自仍带特性的库，但**该特性只在 env 设置时生效**（R215 的 ABBA 实测 off/on 差 −0.30%/−0.19%，即零效果）⇒ **数字有效**；但**在库真正清干净之前，不要把 `89fe01bd…` 当作「与 `dfb838b3…` 等价的对照」记入 md5 台账**。

**判定**：不涉及采用/否证（`-ub` 的端到端收益仍待 harness 验证），**不改动采用链**（基准仍是 **B2**）。
### R219 ★★ **预登记：两个 ub 的经验律 + 「生产侧默认 ub=512」这个可能的大杠杆**（测前写下）

**① 两条经验律（各自锚定在自己 ub 的实测点上，不是外推）**：

| 模型 | 参数（由两点定） | 锚定的实测点 |
|---|---|---|
| **M1（`-ub 2048`）** | `t = 3.742e-4·n + 2.678e-9·n²` | pp8192 = **2524.55**、pp32768 = **2164.72**（官方） |
| **M5（`-ub 512`）** | `t = 4.544e-4·n + 6.583e-9·n²` | p0.log 的 pp32768 = **1492.27**、pp65536 = **1128.98** |

| n | M1 预测（ub2048） | M5 预测（ub512） | 实测 |
|---|---:|---:|---|
| 8192 | **2524** | 1967 | **2524.55** ✓ M1 |
| 32768 | **2165** | **1492** | **2164.72** ✓ M1；p0.log **1492.27** ✓ M5 |
| 65536 | **1819**（待测） | **1129** | p0.log **1128.98** ✓ M5 |
| 131072 | **1379**（待测） | 759 | — |
| 248901 | **961** | 478 | 实测 **370**（ub512，仍差 29%） |

**⇒ 可证伪的判决点（子代理正在跑）**：**pp65536 @ `-ub 2048` 应约 1819 t/s**（同一长度 ub512 实测 1129）。落在 1819 附近 ⇒ **ub 是主因**，R214 的「机制变化」作废；落在 1129–1400 ⇒ **ub 不是主因，长度确实有额外代价**。
**⇒ 同长度 32768 上：ub=512 实测 1492 vs ub=2048 实测 2165 = 光 ub 就差 1.45×** —— 这是**已经测到的**，不是外推。

**② 新发现：生产与 harness 都跑在默认 ub=512 上**
- `/root/llm/systemd/llama-server.service`（**只读**，未改）：`--ctx-size 131072 --n-gpu-layers 999 --split-mode tensor --tensor-split 1,1 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 2 --spec-type draft-mtp --spec-draft-n-max 4` —— **没有 `-ub` 也没有 `-b`** ⇒ **n_ubatch 用默认 512、n_batch 用默认 2048**（且 `--parallel 2` 下每槽更小）。
- `/root/p60-ab-harness.sh` 的启动行同样没有 `-ub`（R218 已记）。
⇒ **若 ub 真是主因，那么「把 ub 提上去」对用户真实场景（256K agent）就是一个零改码、只改一个启动参数的杠杆**（按 M1 外推到 248901 是 961 t/s，而实测 370）。
⚠️ **三条必须同时验的约束**（别只报速度）：① **显存** —— ub 越大峰值显存越高（16 GB 卡 + 256K KV）；② **门** —— 换 ub 会改数值路径，**必须先确立该配置自己的门值**（同臂两次一致），不得沿用 `f3edac19…`；③ **解码不回退**（harness 臂 MEDIAN_TG 不低于约 100）。

**判定**：不涉及采用/否证（数字待测），**不改动采用链**（基准仍是 **B2**）。
⇒ 这条如果成立，**它的量级（可能 1.4-2.6× 的预填充）大于我们此前追的任何一项**，且代价只是一个启动参数。
### R218 ★★ **R214 的「32K→256K 机制变化」很可能是 ub 口径差 —— 而这意味着 256K 预填充可能有现货**

**触发**：核 256K 那个点的原始口径（我上一轮把「机制变化」立成 CURVE 假设，全压在它身上）。

**查到的事实（三条，都可核）**：
1. `1CAT-PORT-BACKLOG.md:186`：**「我们那个 370 t/s」= 248901 token / 672.2 s** ⇒ 它**不是** `llama-bench -p 262144`，而是**一个真实请求**（248,901 token 的 prompt 走 llama-server），而且 **n 也不是 262144**（我 R214 用的 262144 就不准）。
2. **harness 启动 server 时没有 `-ub`/`--ubatch-size`**（`/root/p60-ab-harness.sh` 的启动行只给 `--ctx-size 8192 -ngl 999 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1` 等）⇒ **那个 256K 点跑在 llama.cpp 默认 ub = 512 上**。
3. 而我们的官方预填充基线 `pp8192/pp32768` **显式写了 `-ub 2048`**。

**⇒ 两个数据集口径不同（512 vs 2048），这才是 R214 那个「模型低估 2.5 倍」的最简解释。**
（另有一个佐证：`/tmp/p0.log` 那批 Sep 20 的数据**也没有 `-ub`**，它在 32768→65536 两点反解出的二次系数是 **b = 6.58e-9**，正好是 `-ub 2048` 那组（**b = 2.68e-9**）的 **2.5 倍** —— 与「256K 点上模型低估 2.5 倍」是同一个倍数。）

**⇒ 结论的修正**：
- ❌ **不再主张**「32K→256K 之间有机制变化（注意力拐点）」。**首要解释改为「ub 口径不同」**；
- ⚠️ 但**不能反过来断言注意力没问题**：用 ub=512 那组参数外推 n=248901 给 **520.9 s（478 t/s）**，实测 672.2 s（370 t/s）⇒ **仍乐观 29%**，说明长上下文**还有**一个未解释项（量级远小于此前的「2.5 倍」）。
- ★★ **新出现的现货（待测）**：若 ub 真是主因，则按 `-ub 2048` 参数外推 n=248901 是 **259 s（961 t/s）** —— **对用户真实的 256K agent 场景，这可能比 32K 的 1.4× 更值钱**。⚠️ 但这是**8 倍外推、2 点拟合、无误差估计**，只能当假设，必须实测。

**实验（已派，零改码）**：
1. **长度扫描必须锁死 ub**：`-p 8192,32768,65536,131072 -ub 2048 -r 3`（否则又会把 ub 效应误读成长度效应）；
2. **长长度上的 ub 对照**（新增，优先级提到与①同级）：`-p 65536,131072` × `-ub 512 / 2048 / 8192` —— 这一条直接决定「256K 场景是不是白捡」。

**判定**：不涉及采用/否证（数字待测），**不改动采用链**（基准仍是 **B2**）。

**教训（本会话第五次前提修正，且又是「口径」类）**：**跨数据集比较前，先确认「哪些参数是显式写的、哪些是默认值」** —— 我是靠翻 harness 的启动行才发现 256K 那个点用的是默认 ub。这条与 R215 规矩①是同一族，但**更难查**：它不是「卡数不同」这种显眼的差别，而是「一边写了、一边没写」。
⇒ **新增规矩③：比较两个数字前，逐项列出两侧的显式参数，并标出「未显式给出 ⇒ 用默认值」的那些项。**
### R217 ★ **TP4 及以上会打破 greedy sha256 门（新事实）** + 解码 TP3 最优的独立复现

⚠️ **归属更正（R221 核实）**：这 5 个 harness 解码臂**不是子代理 37e83355 跑的**（它明确否认，且它自己的 ABBA 到 22:56:30 才跑完），而是**本会话更早时段**跑的 —— 文件 mtime 为 `tp4-ab.log` = **21:04:32**、`tp56.log` = **21:57:16**、`p60-tp3b-server.log` = 21:04:29。**它们也不是新数据**：本会话的阶段小结里早已记着「TP4/TP5/TP6 post-FGC 均值 ms/轮 51.86-51.90 / 53.7-53.9 / 56.5-60.5」，与下表逐项吻合。⇒ **R217 的增量只是「把 TP4/5/6 的门值列全」这一项**，其余是重述。

5 个 harness 解码臂（`CARDS` + `FULLGRAPH=1`）**同一次比对里**得到：

| 臂 | tg p1/p2/p3 | AL p1/p2/p3 | **ms/轮 p1/p2/p3（均值）** | greedy sha256 | 门 |
|---|---|---|---|---|---|
| **TP3**（tp3b） | 100.47 / 88.28 / 129.16 | 5.55 / 4.22 / 6.38 | **55.6 / 48.2 / 49.8（51.2）** | `f3edac19…` | **✓ = 门值** |
| **TP3**（t3b） | 99.76 / 87.55 / 128.04 | 5.55 / 4.22 / 6.38 | 55.6 / 48.2 / 49.8（51.2） | `f3edac19…` | **✓** |
| **TP4**（tp4a） | 75.16 / 80.01 / 121.53 | 4.19 / 3.93 / 6.08 | 55.7 / 49.1 / 50.0（51.6） | **`ccc284e4…`** | **✗ 门破** |
| **TP4**（tp4b） | 74.96 / 80.04 / 121.53 | 4.19 / 3.93 / 6.08 | 55.7 / 49.1 / 50.0（51.6） | **`ccc284e4…`** | **✗** |
| **TP5**（t5b） | 77.53 / 100.13 / 117.98 | 4.65 / 5.03 / 6.08 | 60.0 / 50.2 / 51.5（**53.9**） | **`69207026…`** | **✗** |
| **TP6**（t6b） | 73.81 / 69.88 / 110.72 | 5.01 / 3.98 / 6.29 | 67.9 / 57.0 / 56.8（**60.6**） | **`69207026…`** | **✗** |

**① ⚠️ 自查更正：**我最初把「TP4 门破」写成了新发现，**错了** —— `PLAN-GRAPH.md` 的 `TP4B` 节点（R198）**早就记着** *sha256 变为 ccc284e4...（换卡数必变，同配置内两臂完全一致）*。
**真正新增的只有 TP5/TP6 的门值**（此前未逐臂列出）：两者都是 **`69207026…`**。
而 `69207026…` **也不是新值**：它正是**原版 b11053 基线的门值**（`goal-phase1-findings.md:22`），并且 E11 的 device-AR 臂也落在它上面。
⇒ **三个值的谱系（这才是本轮真正厘清的东西）**：`f3edac19…` = **我们改造后的树（B2，TP3）**；`ccc284e4…` = **TP4 与 layer split**；`69207026…` = **原版基线 + device-AR + TP5/TP6**。
⇒ **规矩（仍然成立，但并非新增）**：**任何 TP4+ 的实验不得用 `f3edac19…` 作为其正确性门**；要么先在该 TP 上重新确立门值（同臂两次一致 = 确定性通过），要么显式标注「门是 TP3 专属」。
⇒ **教训（本会话第四次「把已记录的东西当成新发现」）**：**写「新事实」之前先 grep 那三个 hash 与相关节点** —— 本轮是靠 grep `69207026` 才发现的，而它一查就有 24 处命中。

**② 独立复现「TP3 最优」**：ms/轮 **TP3 51.2 / TP4 51.6 / TP5 53.9 / TP6 60.6** ⇒ 自 TP4 起**单调变差**，与既有「加卡不买时间」结论一致（这次是**新一次独立运行**，且 TP3 两次都 51.2）。

⚠️ **口径警告**：`/tmp/p0.log`（Sep 20 21:58 的 KV dtype A/B）里 pp32768 = 1492.27、pp65536 = **1128.98**，但那一批**没有 `-ub`（默认 512）**，与我们的 `-ub 2048`（pp32768 = 2164.72）**口径不同，不得混比**；它只能说明「ub 对预填充影响很大」。

**判定**：两条都不是采用/否证（TP4+ 本就不采用），**不改动采用链**（基准仍是 **B2**：tg 100.95/88.30/129.12、AL 5.55/4.22/6.38、ms/轮均值 50.73）。
### R216 ★ **反量化的代价被直接量出来了：算子级 −19%、端到端只有 3.8%，且已在带宽墙上**

子代理用**同库同形状**（m=4096, n=512, k=14336）做 `test-backend-ops perf` 单算子对照（这是第一次把「现反量化」与「纯 cuBLAS」在同一形状上分开量）：

| type_a | 时间 | 速率 |
|---|---:|---:|
| **f16**（纯 cuBLAS，无反量化） | 880.84 µs | **68.26 TFLOPS** |
| **q8_0**（反量化 + cuBLAS = 现状） | 1090.11 µs | **55.16 TFLOPS** |
| 差 | **+209.3 µs** | **−19%** |

**验算（子代理自做，吻合到 5%）**：权重 4096x14336 = 58.72M 元素 ⇒ q8_0 读 62.4 MB + f16 写 117.4 MB = **179.8 MB**；按 900 GB/s 应为 200 µs，实测 209.3 µs。
⇒ 有效带宽 **约 860 GB/s ≈ 峰值的 95%** ⇒ **反量化内核已经贴在带宽墙上，本身没有可优化空间** ⇒ 只能「**少做几次**」，不能「**做快一点**」。

**规模换算（每卡，TP3）**：一次前向读 9.68 GB(q8) + 写 18.24 GB(f16) = **27.9 GB/ubatch**；`-ub 2048` 时 pp8192 = 4 个 ubatch ⇒ **112 GB ≈ 124 ms = 3.246 s 的 3.8%**。
⇒ **「少做几次」的全部头寸 = 约 2.9%**（`-ub 8192` 可拿回约 3/4）。**它远远不足以解释 1.4× 的缺口** ⇒ 「缺口在别处」成立。

**同时撤销一个跨形状对比**：子代理主动更正 —— 它上一封的「裸 cuBLAS 103 TFLOPS vs 现状 60.8」**是两个不同形状**（103 是 n=8192，R203 的 60.8 是 n=512）。**同形状的真实差距是 55.16 vs 68.26（−19%）**，且其中还含 `test-backend-ops test` 自身的开销。
⇒ **「103 vs 60.8」这个对比作废，不得再引用**；R213/R215 里「那 40% 」的措辞自此替换为：**算子级 −19%、端到端 3.8%**。
（这正好是 R215 规矩①的一次自我执行：**先把两侧形状对齐再相除** —— 而且是子代理自己抓出来的。）

**判定**：不涉及采用/否证（`-ub` 的收益等 ②的实测）；**不改动采用链**（基准仍是 **B2**）。
### R215 A' 的第一个形态被**否证**（子代理同源 A/B）+ 1cat 预填充对比的**按卡归一更正**

**① 实验：把「Q8_0 -> FP16 反量化 + cuBLAS」显式实现一遍，什么也换不来 —— 不采用。**

子代理把这条路实现成 env 门控 `GGML_CUDA_SM70_HMMA_Q8`（源码 `ggml-cuda` 一处），同源 A/B（同一 `libdir-instr`，唯一变量是 env，`-p 8192,32768 -n 0 -r 3 -ctk q8_0 -ctv q8_0 -fa 1 -ub 2048`，`CUDA_VISIBLE_DEVICES=0,1,2`）：

| 臂 | env | pp8192 | pp32768 |
|---|---|---:|---:|
| A1 | off | **2523.66 ± 2.79** | **2163.16 ± 3.66** |
| B1 | on | 2504.18 ± 5.23 | 2153.60 ± 2.26 |
| 差 | | **-0.8%** | **-0.4%** |

- **A/B 有效性已核**：`libggml-cuda.so.0.24.0` md5 由基线 `dfb838b3199e7cd1e9f4fab861e11557` 变为 **`89fe01bdeb14ebc999f1fbe1598c6f89`**（变了 = 新码在库内），且二进制标记串在：`ggml_cuda: GGML_CUDA_SM70_HMMA_Q8 enabled, using Q8_0 to FP16 dequant plus cuBLAS`；`libggml-base.so` 仍 `a8de7c4888531935795888a875c27e36`（未变）✓；无 build lock。⇒ **两臂确实是同一二进制 + 不同 env。**
- 对照组与官方基线吻合到 **0.04%**（2523.66 vs 2524.55，2163.16 vs 2164.72）⇒ 口径干净、可入账。

⇒ **判读**：这正好印证 R213 的读码 —— **默认路径本来就是「全量反量化成 F16 + cuBLAS」，把它显式重写一遍不会改变任何东西**。
⇒ **「那 40%（60.8 vs 103 TFLOPS）来自反量化」这个假设被否证**；40% 更可能是 **n=512 与 n=8192 的差别**（子代理的裸 cuBLAS 是 n=8192 测的，而 R203 的 60.8 是 n=512，且是**未切分的单设备形状**）。
⇒ **节点 DEQ 的 A4（自写融合内核 / F16 副本缓存）不要做**；A2（`-ub`）与 A3（放大 `MMQ_DP4A_MAX_BATCH_SIZE`）仍值得，因为它们问的是「固定开销」而不是「反量化值不值」。

**② 更正：1cat 预填充对比必须按卡归一 —— 缺口是约 1.4×，不是 1.65-1.88×。**

`SESSION-2026-09-20-measurements.md:773` **早就写明** *1cat 公开 32K/64K 3567-4069 t/s（4 卡）*，而我们所有预填充数字都是 **3 卡**；但 `PLAN-to-180ts.md:19` 与 `AGENTS.md` 直接拿原始 t/s 相比，得出「差 1.65-1.88×」并被反复引用。

| 口径 | t/s | **每卡 t/s** |
|---|---:|---:|
| 我们 pp32768（TP3） | 2163 | **721** |
| 我们 pp8192（TP3） | 2524 | 841 |
| 1cat 32K（4 卡） | 4039-4069 | **1010-1017** |
| 1cat 64K（4 卡） | 3567 | 892 |

⇒ **同长度（32K）按卡归一的缺口 = 1010-1017 / 721 ≈ 1.40×**，不是 1.65-1.88×。
⇒ 附带核过：**权重位数对预填充几乎无关**（每个 ubatch 每卡读 9.68 GB ≈ 13.8 ms，而每 ubatch 约 811 ms ⇒ 占 1.7%）⇒ **1cat 用 NVFP4、我们用 Q8_0 这一点不解释缺口**（`1CAT-PORT-BACKLOG.md:187` 已记其契约是 NVFP4+DFlash2）。

**判定**：A' 第一形态**不采用**（-0.4%~-0.8%，且机理上本就不该有收益）；**不改动采用链**（基准仍是 **B2**：tg 100.95/88.30/129.12、AL 5.55/4.22/6.38、ms/轮均值 50.73）。

**③ 本会话第三次「前提被推翻」，且三次都是同一个方向：文档把差距说得比实际更大。**
1. R211/213：把「60.8 ≈ dp4a 峰值 97%」当成「在跑 dp4a」（实为巧合，本就在 cuBLAS-F16 上）；
2. R213：把「换张量核能翻倍」当成主线（我们本就在张量核上）；
3. R215：拿 4 卡的 t/s 比 3 卡的 t/s（缺口 1.65-1.88× 实为约 1.40×）。
⇒ **规矩（追加）**：**任何跨系统的性能对比，先把两侧的「卡数 / 形状 / 量化 / 长度 / 是否切分」对齐再相除**；对不齐的只能当线索，不得进结论。
### R214 预填充曲线的**机制变化**：不存在一组 (a,b) 能同时拟合 8K / 32K / 256K

用三个已有官方点检验 R213 那个二次模型 `t = a*n + b*n^2`（`a = 3.742e-4 s/token`、`b = 2.678e-9 s/token^2`，由 8K+32K 两点定）：

| n | 实测 t | 模型 t | 差 |
|---|---:|---:|---|
| 8192 | 3.245 s（2524.55 t/s） | 3.245 s | 拟合点 |
| 32768 | 15.137 s（2164.72 t/s） | 15.137 s | 拟合点 |
| **262144** | **约 708 s（370 t/s，TTFT 672 s）** | **282 s** | **模型只有实测的 40%** |

⇒ **模型在 256K 上失效（低估 2.5 倍）**。而且**没有任何 (a,b) 能同时拟合三个点**：
若强行让 b 去拟合 256K（需 b ≈ 8.9e-9），则 32K 会预测成 21.8 s（1503 t/s），与实测 2165 t/s 差 30%。

**⇒ 结论（假设级，但方向明确）：32K 与 256K 之间存在一次机制变化，最可能是注意力。** 这与 R203 早先把 `FLASH_ATTN_EXT`（D=256/24头/n_q=512）标成**病态**（108,972 µs、11.7 GB/s）是同一件事。
⇒ 对**用户真实场景（256K agent）**而言，这条比 GEMM 更值钱；但对**当前标尺（32K）**而言注意力只占约 19%。

**待做判别（已加入子代理的预填充扫描，一次加载即可）**：`llama-bench -p 8192,32768,65536,131072`（+ 若能跑 `-p 262144`）⇒
- 若曲线是**平滑幂律**（log-log 上一条直线）⇒ 只是一个更大的 b，注意力常数大但无拐点；
- 若**出现拐点**（131072 或 262144 处掉得比幂律快）⇒ 有定性的机制变化（FA 的 split-KV / KV f16 转换 / 显存压力），需要单独查。

⚠️ 256K 那个点是**较早、不同口径**测的（未必同库同参），所以按「线索」对待，不作为定量基线；但它与 32K 点的矛盾**无法用测量噪声解释**（差 2.5 倍）。

**判定**：不涉及采用/否证，**不改动采用链**（基准仍是 **B2**）。
### R213 ★★ **定论：预填充早就在 FP16 张量核（cuBLAS）上 —— A 的原前提作废，真缺口是「每次 matmul 全量反量化」**

纯读码 + 构建配置确认，**不需要实测即可定**（本轮没有上机、没有性能数字）。

**证据链（全部可核）**：
1. `build-instr/CMakeCache.txt`：**`GGML_CUDA_FORCE_CUBLAS=OFF` 且 `GGML_CUDA_FORCE_MMQ=OFF`** ⇒ 两个 `#ifdef` 分支都不生效。
2. Q8_0 在 Volta 上过四道关全部不过：`mmvf`（要非量化）-> `mmf`（**`mmf.cu:135` 对量化类型直接 `return false`**）-> `mmvq`（`MMVQ_MAX_BATCH_SIZE 8`）-> **`mmq`（`mmq.cu:334` `ne11 < MMQ_DP4A_MAX_BATCH_SIZE = 64`）**。
   ⇒ **`ne11 >= 64` 时落到 `ggml-cuda.cu:2014` 的 `ggml_cuda_mul_mat_cublas`。**
3. `ggml-cuda.cu:1762-1764`：量化 src0 取 `compute_type = fast_fp16_hardware_available(cc) ? F16 : F32`，sm_70 为 **F16**。
4. `ggml-cuda.cu:1585-1608`：**每次调用都** `src0_alloc.alloc(ggml_nelements(src0))` + `convert_func(...)` = **把整张权重全量反量化成 F16**，再交 cuBLAS。
5. `mmq.cuh:8` 注释自证设计意图：*Max. batch size to use for dp4a MMQ kernels **when FP16 tensor cores are available***。

**⇒ 三条被推翻的旧判断**：
- ❌ 「预填充 ffn_gate 1502 µs = 60.8 TFLOPS ≈ INT8 dp4a 峰值 62.8 的 97% ⇒ 在跑 dp4a」—— **n=512 根本不走 dp4a**，那个 97% 是**巧合**。
- ❌ 「1cat 预填充领先 1.65-1.88× 的来源是 FP16 张量核」—— **我们已经在张量核上**。
- ❌ R211 那条「Q8_0 逐层反量化成 FP16 再走 cuBLAS」的『新形态』—— **它本来就是现状**，不是待实现项。

**⇒ 真缺口（新主轴 A'）**：60.8 TFLOPS（llama.cpp 路径，n=512）vs **103 TFLOPS（裸 `cublasGemmEx`，子代理实测）** ⇒ **差的那约 40% 是「每个 matmul 一次 alloc + 全量反量化整张权重」**，不是内核能力。

**新计划（前三条零改码或一行改码，已派子代理）**：
| # | 内容 | 代价 | 判读 |
|---|---|---|---|
| **A1** | `test-backend-ops perf -o MUL_MAT` 扫 `ne11 = 8..2048`，预期**在 64 处跳变**；并用 `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`（只被 `mul_mat_cublas` 读）交叉验证 | 10 分钟零改码 | 坐实分派 |
| **A2** | **`-ub` 扫描**（512/2048/8192/16384）：反量化是**每次 matmul 的固定开销，与 n 无关** ⇒ ubatch 越大摊得越薄 | 10 分钟零改码 | **可能是白捡的提升** |
| **A3** | env 门控地把 `MMQ_DP4A_MAX_BATCH_SIZE` 在 Volta 上放大 => 预填充改走 **MMQ 融合 dp4a 内核（不建 F16 副本）**，与 cuBLAS+反量化对打 | 一行改码 | 三种结果都有信息量（含「反量化≈免费」） |
| **A4** | 仅在 A2/A3 显示真头寸时：融合反量化+HMMA 内核，或 F16 副本缓存（**16 GB 卡放不下 19.4 GB，必须逐层复用**） | 大 | 最后 |

**顺带得到一个可证伪的预测（顺便量注意力份额）**：用两个官方点做二次拟合 `t = a*n + b*n^2`（`t(8192)=3.2451 s`、`t(32768)=15.1369 s`）⇒ **a = 3.742e-4 s/token、b = 2.678e-9 s/token^2**
⇒ 8K 时二次项占 **5.5%**（0.180 s）；32K 时占 **19.0%**（2.876 s）；**二次项与线性项相等点在 n ≈ 140K**。
⇒ **预测 pp65536 ≈ 1820 t/s**（t≈36.0 s）。⚠️ 两点拟合两参数是恰好确定、**无误差估计**，只能当假设；用 `-p 65536` 打一枪即可判。

**判定**：不涉及采用/否证，**不改动采用链**（基准仍是 **B2**：tg 100.95/88.30/129.12、AL 5.55/4.22/6.38、ms/轮均值 50.73）。

**教训（比数字更值钱）**：**「性能数字恰好接近某个 roofline 的 97%」被当成了「它在跑那条路」的证据。** 这是拿相关性当因果，代价是**一整条主轴的方向错误**（子代理已经在写这条路的内核了）。
⇒ **规矩：凡「它在用 X 而不是 Y」的判断，必须落到「读分派链到确定」或「一个只有 X/Y 之一能通过的实验」**，不得用性能数字与峰值的接近程度代替。
### R212 分派链读码：**「预填充到底在跑 MMQ(dp4a) 还是 cuBLAS-F16」是一个尚未定论的开放问题**

（本轮纯读码 + 已发出判别实验；**尚无答案**，写下以防下一轮又照着未验证的前提投入工作。）

**触发**：R211 的路线转向建立在「预填充 ffn_gate 60.8 TFLOPS ≈ INT8 dp4a 峰值 62.8 的 97% ⇒ 在跑 dp4a」这个判断上。读分派链后发现**源码不支持这个前提**。

**源码证据链（本地树，行号可核）**：
1. `ggml-cuda.cu:1961-2014` `ggml_cuda_mul_mat` 分派顺序：bad_padding/类型 -> mmvf -> mmf -> mmvq -> **mmq** -> **最后才 `ggml_cuda_mul_mat_cublas`（:2014）**。
2. `mmq.cu:266-335` `ggml_cuda_should_use_mmq` 的 N 卡终判：`return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;`
3. `common.cuh:326-330`：`fp16_mma_hardware_available(cc) = IS_NVIDIA && cc >= GGML_CUDA_CC_VOLTA` ⇒ **sm_70 为 TRUE**（前面 `if (turing_mma_available(cc)) return true;` 对 Volta 为假，dp4a 架构检查也不截）。
   ⇒ **按源码，Volta + Q8_0 的判定应是 `should_use_mmq = (ne11 < 64)`，即 `ne11 >= 64` 会落到 cuBLAS。**
4. `ggml-cuda.cu:1762-1764`：**`mul_mat_cublas` 自己就处理量化 src0** —— `if (ggml_is_quantized(compute_type)) compute_type = fast_fp16_hardware_available(cc) ? F16 : F32;`，而 `common.cuh:320-323` 对 sm_70 返回 **TRUE** ⇒ **它会把 Q8_0 反量化成 F16 再走 cuBLAS**。
5. 上游本就有 `GGML_CUDA_FORCE_CUBLAS`（*always use cuBLAS instead of mmq kernels*）与 `GGML_CUDA_FORCE_MMQ` 两个构建开关；`ggml-cuda.cu:1771` 还专门给 Volta 特判过（`cc >= VOLTA ? 8 : 128`）。

**⇒ 两种可能，必须实测分辨（已派给子代理，10 分钟）**：
- **甲：预填充早就在跑 cuBLAS-F16** ⇒ R203 的「60.8 ≈ dp4a 97%」是**误读**；真正的缺口是「**每次 matmul 都现反量化整张权重 + 布局转换**」（裸 `cublasGemmEx` 在同形状是 103 TFLOPS）。目标改为「让反量化只做一次」。
- **乙：仍在 MMQ** ⇒ 说明还有别的分支抢先（最可能是该 build 的 CMake 开了 `GGML_CUDA_FORCE_MMQ`）；查清后**改动只需约 3 行**（`should_use_mmq` 对 `Q8_0 + Volta + ne11>=64` 返 false + env 门控），**因为 `mul_mat_cublas` 已自带 F16 反量化**，不需要自写反量化管线/scratch/capture 兼容。

**判别实验**：(a) `test-backend-ops perf -o MUL_MAT` 在已装库上扫 `ne11 = 8/16/32/64/128/256/512/2048`，**看 64 处是否跳变**；(b) `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`（只被 `mul_mat_cublas` 读）作交叉验证；(c) `grep -e FORCE_CUBLAS -e FORCE_MMQ build-instr/CMakeCache.txt`。

**已同步更正**：R203 表里「MUL_MAT（预填 n=512）12.8 TFLOPS ≈ FP32 峰值 82%」这一行是**异质形状的误导性平均值**（跨 25× 形状范围），**不得再引用**；单形状数字以 ffn_gate 的 60.8 TFLOPS 为准。

**判定**：不涉及采用/否证，**不改动采用链**（基准仍是 **B2**）。
### R211 A 的路线**转向**：手写 m8n8k4 否证，改走「Q8_0 逐层反量化 -> 已有 FP16 cuBLAS 路径」

子代理 37e83355 的**决定性微基准**（`/tmp/sm70probe2.cu`，sm_70，V100，80SM@1.53GHz）三条硬数字：

| # | 事实 | 证据 |
|---|---|---|
| 1 | **sm70 只有 `m8n8k4`**：ptxas 实测 `m16n8k16.f16` 在 **sm_70 与 sm_75 都报错**（*requires .target sm_80 or higher*）⇒ Turing 的 q8_0 MMA 路径（int8 `m16n8k16`）**在 Volta 根本不存在**，不能「补一个 Volta 变体」 | ptxas |
| 2 | **手写 m8n8k4 当前形态只有 26.7 TFLOPS**（纯寄存器、ptxas 报 0 spill、4x4 mma 循环 26.7；smem 喂数 27.1；32x16 tile 26.7） | 微基准 |
| 3 | **cuBLAS 在我们的真实形状上就是 103 TFLOPS**：W[17408,5120]xX[8192,5120] **C16F 103.0 / C32F 101.3**；W[6144,5120]xX[8192,5120] 101.5；W[4096,14336]xX[512,14336] 96.2；只有 ffn_down W[5120,17408]xX[8192,17408] 偏低 **67.8** | 直接 `cublasGemmEx` |

**⇒ 关键推论：换 FP16 张量核的收益主要来自「用 cuBLAS」，不是来自「手写 mma」。** 且 llama.cpp 的 F16 路径在 Volta 上**本来就强制 f32 输出**（`ggml-cuda.cu:1649-1650`，`CUBLAS_COMPUTE_32F`）⇒ 精度不是问题。

**⚠️ 我上一轮的错误（R209 ⑰）已撤销**：那条「否决 cuBLAS + 全局反量化」的算术是 **decode 形状**的 —— 我比的是「反量化流量 450 MB vs 权重读取流量 94 MB = 4.8x」。预填充的正确比法是「**反量化流量 vs GEMM 时间**」：GEMM 时间随 n 线性涨，**反量化每 ubatch 只付一次**。每卡 38.8 GB 额外搬运 ≈ 54 ms，对 pp8192 的 3.25 s 是 **1.7%**，对 pp32768 再低 4 倍 ⇒ **子代理的账是对的，我的是错的**。

**新形态（子代理实施中）**：Q8_0 权重**一次性反量化成 FP16**（复用已有 `dequantize_block_q8_0_f16_cuda`）-> 走**已有 fp16 cuBLAS 路径**；env 门控 `GGML_CUDA_SM70_HMMA_Q8`，只挂 `ne11 >= 64` 的预填充分支，**只动 `ggml-cuda.cu` 一个文件**。

**两条已提醒的实现硬约束**：
1. **16 GB 卡装不下 19.4 GB 的整模型 FP16 副本** ⇒ 必须是**逐层复用的同一块 scratch**（dequant 一层 -> GEMM -> 覆盖）。单层最大权重 ffn 17408x5120：Q8_0 94.7 MB -> FP16 178 MB，TP3 每卡约 60 MB，放得下。
2. **每个 ubatch 都要重反量化一次**：`-ub 2048` 时 pp8192 = 4 个 ubatch ⇒ 54 ms x 4 ≈ **216 ms（6.7%）**（不是 1.7%）。⇒ A/B 里**同时测 `-ub 8192`**（pp32768 用 ub 8192 只付 4 次而不是 16 次）。

**预登记期望值（先写下，避免事后迁就）**：GEMM 60.8 -> ~95 TFLOPS（加权，ffn_down 拖后腿）时，**pp8192 约 2524 -> 3300-3400**，**pp32768 约 2165 -> 2900-3100**（ub 越大越接近上限）。这能把 1cat 的 3567-4069 追到七八成。

**第 2 条记为「未定论」而非「指令极限」**：sm70 上除 `m8n8k4` 没有别的张量核指令，而 cuBLAS 用**同一套指令**跑到 103 TFLOPS ⇒ 26.7 **不是指令天花板**。最可能是**占用率** —— V100 每 SM 4 个张量核，125 TFLOPS 需要 4 个 warp 同时喂，而 **1 warp/SM 的上限正好 ≈ 125/4 ≈ 31 TFLOPS**，非常接近实测 26.7。⇒ 正确表述是「**当前形态受占用率/寄存器带宽限制**」，将来若要重写内核，关键是**每 SM >= 4 warp + 每 warp >= 4 个独立累加器**，而不是改 tile 形状。

**判定**：尚无性能数字，**不改动采用链**（基准仍是 **B2**：tg 100.95/88.30/129.12、AL 5.55/4.22/6.38、ms/轮 50.73 均值）。
### R209 A 的参考弹药：**ninfer 里有 Volta 原生、且与 Q8_0 同构的张量核 GEMM**（纯读码，未上机）

这一轮只做侦察，**没有产生 tg/ms 数字**（子代理 37e83355 占着机器在跑 A 的实现）。产出是两份可执行的弹药与一条改写的旧判断。

**① 否决「cuBLAS + 全局反量化」这条岔路（算流量就判死）**
1cat 的 `benchmarks/csrc/sm70_awq_m5_batched_gemv.cu` 是 **decode 侧**的量化 batched GEMV（`kM=5` = 5 tokens），不是预填充 GEMM。以我们的预填充形状（m=17408, k=5120, n=512，TP3，每层约 89M 参数）估算每卡每层流量：

| 路线 | 每卡每层流量 | 说明 |
|---|---|---|
| 现状：dp4a 直接读 Q8_0 | **约 94 MB** | 权重只按存储密度读一遍 |
| cuBLAS + 全局反量化 FP16 | **约 450 MB（4.8x）** | 读 94 + 写 178(FP16) + GEMM 再读 178 |

⇒ **反量化流量本身会取代 GEMM 成为瓶颈** ⇒ 反量化必须留在 smem 里逐 tile 做。已发给子代理避免走弯路。

**② 正面答案：`v100-refs/ninfer-v100/src/ops/linear/w8/w8_volta_mma_gemm.cuh`**
- **sm_70 门控**（`__CUDA_ARCH__ == 700`）的 `mma.sync.m8n8k4` **融合反量化 GEMM**；注释原话：*dequantise into **shared** memory and feed the tensor cores directly, so the weight is read once per K pass at its stored density*。
- **它的权重格式 W8G32 = 每元素 1 个 signed int8 + 每 32 个 K 一个 FP16 scale ⇒ 与 ggml Q8_0 同构**（只差 codes/scales 分开存 vs 交错）。**这就是 A 的现成蓝图。**
- **魔数解码**（`:120-134`，每 2 元素仅 2 条指令、全程无 int->float）：`0x6400|u` 作为 fp16 位型恰等于 1024+u；码 byte XOR 0x80 变成 u=b+128 ⇒ `0x6400|u == 1152+b`；一条 `__hsub2` 减 1152.0 还原有符号码 + 一条 `__hmul2` 上 scale。
  ⇒ **推翻「smem 内反量化的指令开销会吞掉张量核收益」这个担心。**
- **fragment 取数**（`ops/common/volta_mma.cuh`；注释：*Volta has no ldmatrix, sm_75+ only*）：`volta_load_qp`=A、`volta_load_k`=B（I-major-mirrored，32 lane 只有 8 个不同行）、`volta_mma_qk`=`mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`、`volta_d_get_i|j`。
- **调度**：`kWarps=4`（每 warp 8 输出行）/`kKStep=32`/`kTTile=32`/**`kXPad=8`（注释：hardware floor）**/128 线程/双缓冲；**K 循环全程只有 1 个 `__syncthreads()`**（`:142-144` 自己解释了为什么够）—— 与已知的「v100-skinny 赢法 = 全程 1 barrier」一致。
- **两条避坑**：① Large-T 版 `w8_rowsplit_gemm_mma.cuh` 是 `m16n8k16`+`ldmatrix` = **sm_80+**，且 Volta 48KB 静态 smem 会让 **nvlink 以链接期失败**拒绝 ⇒ **Volta 只能用 `m8n8k4` + 普通 smem 取数**；② 它的 SIMT W8 回退 *re-reads the entire weight once per 8 output columns ⇒ cost exactly linear in T*（T=24 搬 3.9 GB 而只需 1.31 GB）—— **我们的 dp4a 没这个缺陷，别拿它当基线**。

**③ 改写一条旧判断**：§3.2.9 的「HMMA 需要 f16 操作数（q8_0 要新写 34 字节块的解包分支）」**对注意力仍然成立**（sm70-attn fork 的 FA 内核确实只吃 f16），但**对矩阵乘已作废** —— 解包分支在 ninfer 的 `ops/linear/w8` 里已经存在。已同步进 `PLAN-GRAPH.md` 的 §3.2.3/NF10、HMMA 节点与 Round 209 日志。

**④ 检索教训**：上游确实**没有** sm70 MMQ/GEMM 的 HMMA 补丁（只有 FA 的 Volta MMA，PR #17505 已合并且我们树里已有），但**本地只读参考 `v100-refs/` 里有一份完整实现** ⇒ **「联网找不到」不等于「没有答案」，先翻本地参考再下结论。**

**判定**：不涉及采用/否证（无性能数字），**不改动采用链**（当前基准仍是 **B2**：tg 100.95/88.30/129.12、AL 5.55/4.22/6.38、ms/轮 50.73 均值）。价值在于把 A 从「从零写」变成「照蓝图搬」。
### R208 A 的读码/参考侦察（非侵入，未碰服务器）

- **上游无捷径**：GitHub 搜索确认只有 FA 的 Volta MMA（PR #17505 已合并，我们树里已有），**没有人给 MMQ/矩阵乘加过 sm70 HMMA 路径** ⇒ A 是新地。
- **本地只读参考里有整套 sm70 HMMA 实验台**（`1cat-vllm/`）：`flashinfer-sm70/include/flashinfer/attention/sm70/volta_mma.cuh`、`benchmarks/csrc/sm70_awq_m5_batched_gemv.cu`（量化权重 batched GEMV，decode 侧 M=5）、`sm70_hmma884_schedule_micro.cu`、`sm70_raw_hmma_probe.cu`、`sm70_hmma_c2a_mincomm_micro.cu`、`sm70_awq_splitk_reducer_micro.cu`、`benchmark_sm70_cublas_gemm_algorithms.py`。
- **否决「cuBLAS + 全局反量化」这条岔路**：以预填充形状（m=17408,k=5120,n=512，TP3，每层 89M 参数）估算，每卡·层流量从现状约 94 MB（读 Q8_0）涨到约 450 MB（读 94 + 写 178 FP16 + GEMM 再读 178）≈4.8× ⇒ 反量化流量本身成为瓶颈。**反量化必须在 smem 里逐 tile 做**。
- **代码侧证据链**（确认 V100 预填充在跑 dp4a）：`mmq.cuh:8` `MMQ_DP4A_MAX_BATCH_SIZE 64`（注释：FP16 张量核可用时 dp4a 最多用到 64）；`mmq.cuh:275-285` 设备侧 `__CUDA_ARCH__ >= VOLTA` 但非 Turing-MMA ⇒ 返回 `pascal_dp4a` 配置；`mmq.cuh:197-201` 整段 MMA tile 挂在 `TURING_MMA_AVAILABLE`；`common.cuh:360` 已有 `volta_mma_available(cc)`；`fattn-mma-f16.cuh` 是我们树里 sm70 跑 `mma.sync` 的现成范例。
- 1cat 参考里可抄的点：**一次性权重重排**（`prepare_weight_kernel`，为 int4→half2 快速转换器排布，主循环零额外指令）与 tile 参数（`kThreads=128, kThreadK=32, kTileK=64`，小 M 形状）。
### R206 草稿版本 A/B（Q4_K_M 控制 vs Q8_0）：**不采用 Q8_0**

四臂 ABBA、同库同 env、只换草稿文件、四臂 sha256 门全一致 `f3edac19...`（输出完全一致）：

| 草稿 | AL p1/p2/p3 | ms/轮 p1/p2/p3（均值） | tg p1/p2/p3 | 中位 tg |
|---|---|---|---|---:|
| **Q4_K_M（控制）** | 5.58 / 4.25 / 6.38 | 55.37 / 48.19 / 49.32（**50.96 / 50.99**） | 100.8 / 88.2 / 129.3 | **100.83** |
| **Q8_0** | 3.98 / 4.78 / 6.59 | 51.81 / 48.53 / 49.12（**49.82 / 49.88**） | 76.9 / 98.5 / 134.1 | 98.46 |

⇒ **不采用**。关键教训：**AL 不同时「均值 ms/轮」不是公平指标** —— Q8_0 的 p1 AL 从 5.58 掉到 3.98，每轮接受更少 ⇒ 固定开销被摊薄 ⇒ 轮时更短（51.81 vs 55.37）但需要更多轮。
按 tg 判：Q4_K_M 赢 p1（+31%）、Q8_0 赢 p2/p3（+12%/+3.7%），**中位 tg 仍是 Q4_K_M 高**（100.83 vs 98.46）⇒ 维持 Q4_K_M。
⇒ 反直觉事实（已复现，两臂完全一致）：**草稿精度只改变「接受模式」，且强烈依赖 prompt**（p1 −29%、p2/p3 反升）；与官方评测「Q4_K_M 的 AL 最高（5.39 vs 5.13）」方向一致。
⇒ Q8_0 草稿文件保留在本地盘（2,056,414,816 B，sha256 c18e800d...与 HF etag 逐字符相同），随时可再试。
### R205 预填充 FA 开刀结果：**否证**（附真实预填充基线与缺口定位）

**基线（llama-bench，r=3，本地盘加载，TP3 tensor + q8_0 KV + fa=1 + ub2048）**
| 口径 | t/s |
|---|---:|
| pp8192（空 KV） | **2524.55 ± 4.63** |
| pp32768 | **2164.72 ± 3.19**（与账本 2162.64 吻合）|
| pp1024 @ d=100000 | **910.34 ± 22.19** |
| pp8192 @ d=100000 | **1031.49 ± 0.51** |

**① 上游 PR #27997**（唯一对形状的：sm70 FA config for DKQ=256/DV=256/ncols=64，宣称 PP +27.8%，+2/-0 行）
按其把 Volta 表里 `(256,256,64, ..., 128, 128, 128, ...)` 的第 7 参 128 改成 64，两臂同库同命令（脚本内换文件 + 重编）：
| 口径 | PR 配置 | 我们原有 | Δ |
|---|---:|---:|---:|
| pp8192 | 2522.82 ± 4.17 | 2524.55 ± 4.63 | -0.07% |
| pp32768 | 2139.69 ± 2.72 | 2164.72 ± 3.19 | **-1.2%** |
| pp1024 @ d100k | 872.00 ± 18.56 | 910.34 ± 22.19 | **-4.2%** |
| pp8192 @ d100k | 991.74 ± 1.19 | 1031.49 ± 0.51 | **-3.9%** |
=> **两个口径都更慢 ⇒ 不采用**。原因：该 PR 修的是「256x256 回落到 Ampere 配置」的路径，而**我们树里已有自己的 D=256 Volta 配置**（commit 2168d6a28，Q 走 shared memory），比上游回落版更好。源码已还原，`git status` 干净。

**② KV 类型（f16 vs q8_0）**：pp32768 2138.03 vs 2133.42（另一次 2123.19）；pp8192 2509.56 vs 2510.52 ⇒ **无差别** ⇒ [FAK] 的 `need_f16_K/V=1` 内核内转换**不是**预填充瓶颈。

**③ 缺口真正的位置（逐算子表反推）**：预填充 `ffn_gate` 形状 m=17408,k=5120,n=512 实测 1502 µs = **60.8 TFLOPS ≈ INT8 dp4a 峰值 62.8 的 97%** ⇒ **我们的预填充 GEMM 已经贴在 dp4a roofline 上**；
而 V100 的 **FP16 张量核是 125 TFLOPS（≈2× dp4a）** ⇒ 1cat 的 1.65-1.88× 优势**来自张量核 GEMM**，不在注意力。
=> 预填充的真正解法 = 给量化权重做 **FP16 张量核 GEMM（smem 内反量化 -> mma.sync）**，参考：flashinfer PR #3526（D=256 预填充 64 KiB smem）、`jackinthebox52/qwen38-v100-serve` 的 `0001-t2-001-gqa-packing-sm70.patch`、v100-skinny（QP 切 N / A-stationary / 56 寄存器 / 全程 1 barrier）。
### R204 FGC 复核 + 捕获路径拆解 + 两个「陷阱」

**① FGC 的收益再次被独立复现**（本轮 `f2` 臂是真对照：没有设 `GGML_META_FULLGRAPH`）：
- 非 FGC：`[META] sub/call=47.6 total=9.660 ms/call`，均值 ms/轮 **54.33**（p1/p2/p3 = 55.15 / 52.36 / 55.46）
- FGC：`sub/call=3.2 total=3.88-3.92`，均值 ms/轮 **50.91-51.20**
⇒ **-6.3%**，与 E15 四臂 A/B 的 -6.5% 一致；B2 站得住。

⚠️ **陷阱 1（我自己脚本踩的，必须记）**：`GGML_META_FULLGRAPH` 用 `getenv(...) != nullptr` 判断，**只看变量是否存在、不看值** ⇒
`GGML_META_FULLGRAPH=0` **照样开启 FGC**。要关就是**完全不设**。（本轮 `n1` 臂因此不是对照，浪费了一臂。）

**② 捕获路径终于拆开**（新增 `sig=/fgrun=/fgcap=` 计数器，832 calls，稳态）：
| 项 | ms/call | 含义 |
|---|---:|---|
| `sig` | 0.103 | 我加的签名哈希（120k 次 mix）|
| `fgrun` | 0.272 | 录制时跑一遍子图循环（只录不执行）|
| `fgcap` | 0.442 | `cudaStreamEndCapture` + `cudaGraphInstantiate` |
| `loop` | 2.798 | 全部「走过循环」的调用均值（含上面两项 + **plain loop**）|
| `prologue` | 1.118 | 入口 + rebuild（24 次 × 约 26 ms 建 14850 张图像）+ 签名 + 槽查找 |
⇒ 我原以为「首次见到签名白跑一整轮」占 2 ms/call，**实测不成立**（见下）。

**③ 捕获优先（`GGML_META_CAPTURE_FIRST`）：不采用**
当 `!needs_rebuild`（图必然复现）时第一次见到签名就录制，省掉那次 plain loop。四臂同源 ABBA：
- 控制 c0a/c0b 均值 ms/轮 **51.20 / 51.16**；特性 c1a/c1b **51.00 / 50.93** ⇒ **-0.4%**，落在同日离散带（0.7%）内。
⇒ **在噪声内，不作为阶段成果**；代码保留但 env 默认不设。（又一次：计数器变化 ≠ 墙钟收益。）
### R203 逐算子表（用我们模型真实形状，CUDA0，`test-export-graph-ops` + `test-backend-ops perf --test-file`）

| 算子 | 单项 | 效率 | 判读 |
|---|---:|---|---|
| **FLASH_ATTN_EXT**（D=256/24头/n_q=512） | **108,972 µs** | **11.7 GB/s** | **病态**；这是**预填充**走的路（[FAK]：n_q>1 -> MMA_F16 + KV 转 f16） |
| MUL_MAT（解码 n=1 各形状） | 70–1536 µs | **678–820 GB/s** | **已接近 V100 峰值 900** ⇒ 解码的矩阵乘没有内核空间 |
| MUL_MAT（预填 n=512） | 0.8–19.9 ms | ~~12.8 TFLOPS~~ | ⚠️ **R212 更正：这是个误导性的平均值** —— 该行跨了 25× 的形状范围（0.8-19.9 ms），12.8 TFLOPS 是**异质形状的聚合**，**不能代表 ffn_gate**；单个 ffn_gate(m=17408,k=5120,n=512) = 1502 µs = **60.8 TFLOPS**。**该行不得再被引用**，见 R212 |
| GATED_DELTA_NET | 913 µs | 49.4 GB/s | 48 层都是它，值得单独查 |
| SSM_CONV / CONCAT / SILU / RMS_NORM | 55–110 µs | 386–705 GB/s | 正常 |

⚠️ 口径与警告：
- 该表是**单设备（未切分）形状**、且 `test-backend-ops` 用默认 f16 KV/无 mask 构造 FA，**绝对值不可直接乘进每轮账**；
  但**同一张表内的相对关系**（FA 比同批其它算子差 1-2 个数量级、解码矩阵乘已在 678-820 GB/s）是可信的。
- 与 E16 自洽：8-token target GPU 28.44 ms ≈ 17.3（矩阵乘，按 n=8 的 559 GB/s）+ 9.4（非矩阵乘）✅
- 由此得到的战略结论：**解码侧（tg）的矩阵乘已到极限，150 t/s 不能再指望它**；能动的只剩 ①一轮里的非矩阵乘 9.4 ms ②串行链约 20 ms；
  而**预填充**侧存在一个数量级的病态项（FA n_q>1），是第二大缺口 2162 vs 3567-4069 的主要嫌疑。
- 复现：`test-export-graph-ops -m <target.gguf> -o /tmp/ops.txt`（只读元数据、不加载权重，2.5 s）+ `test-backend-ops perf --test-file /tmp/ops.txt -b CUDA0`。
### 上游查证（R200-R201，用户要求「多用互联网查 PR」）

- **PR #23432**（**closed，未合并**，2026-05-20）*ggml: replace fixed 1GB context pool with growable buffer in meta backend*（修 issue #22404）：
  上游记录了我们撞到的**同一类**崩溃 —— meta 后端给每个 simple backend 一个**固定 1 GB** 的 ggml_context 存 tensor 元数据，
  大图（长上下文 / **投机解码** / checkpoint 恢复）时会 `ggml_new_object: not enough space in the context's memory pool`；
  修法是**可增长 buffer**（4 MB 起按需增长）。关联 **#22616**：先试「固定尺寸精算」，**仍然会崩**。
  ⇒ 我们的情形是同一 bug 的**小号版本**：compute 容器只有 `16 x 模型 ctx`（约 754 KB），加长掩码块时**只差 1 个 tensor（368 B）**。
  已用「固定尺寸 + 可调 headroom」（`GGML_META_COMPUTE_HEADROOM`，默认 32）解决，**但继承了同样的脆弱性**；
  要根治应移植 #23432 的可增长 buffer（结构性改动，用户已授权内核级/结构性改刀）。
- **PR #26575**（**open，未合并**，2026-08-04）*spec: respect safe draft caps before block decode*（关联 #26478）：
  DFlash/DSpark 的**每序列草稿上限**与 block decode 之间有布局安全约束，且提到 16k 上下文边界附近的 KV 位置缺口。
  ⇒ 与我们「加长掩码块」的实验直接相关（我们绕过了 n_max 上限，这在上游是被显式约束的）。
- **PR #25173**（DSpark 投机解码，open）：`is_dspark` / `sample_from_anchor` 那条分支的来源，我们的树里已有该分支。
- 上游坐标补充：#23432(closed 未并) / #22404(issue) / #22616(closed 未并) / #26575(open) / #26478(issue)。

| 阶段 | ms/轮 | 来源 |
|---|---:|---|
| target 阶段（提交到同步完成） | **36.86** | target ctx `sync_us/rounds = 10836496/294` |
| ↳ 其中主机 enqueue | 8.40 | `enqueue_us/rounds = 2470651/294` |
| ↳ 其中 **GPU 执行** | **28.44** | 两者之差 |
| draft 阶段 | 6.13 | draft ctx `sync_us/rounds = 214601/35` |
| selector | 5.16 | spec timing |
| sampler + 其余 | ~2.5 | 残差 |
| **合计** | **50.65** | = 实测轮时 50.7（三臂 50.73/51.16/50.65） |

=> **AL 5.55 下的地板 = 28.44 + 6.13 + 5.16 + 2.5 = 42.2 ms => tg 上界 132**（即使主机时间全部消失）。
=> 150 t/s 只有两条路：**砍 target 的 28.44**（每卡字节数或有效带宽）或 **抬 AL**。
=> 加 sync 只让轮时贵 0.4 ms => 目标 GPU 本来就没有与别的工作重叠；「靠重叠隐藏」不是选项。
=> 该结论同时**解释了 E14**：Q4_K_M 把每卡字节砍 41% 却更慢 => k-quant 在 Volta 上每字节 GPU 时间更多，字节数不是约束。
=> 结论对下一阶段排序的影响：TP4（每卡 9.68 -> 7.25 GB）与「一轮两块草稿抬 AL」是仅有的两条通向 150 的路。

## 已否证、不得再当基准的阶段成果（灰名单）

| 候选 | 结果 | 为何不采用 |
|---|---|---|
| E7 形状敏感 CUDA 图 key | tg 97.40 -> 97.31（噪声内） | 机制成立但性能为零；计数器增减不是墙钟代理 |
| E10 延迟 AllReduce（`GGML_META_DELAY_AR`） | 未上机（结构不可能） | 129/129 边界 `del=0`，被归约值总被下一个节点消费 |
| E11 设备端/内部 AllReduce | tg 98.12 -> 94.45 / 89.80 | 更慢 **且 sha256 门破**（`69207026...`） |
| Q4_K_M 目标权重（E14） | ms/轮 58.6 vs 55.8（反而更慢），AL 5.55 -> 4.55 | k-quant 在 Volta 上每字节 GPU 时间更多；权重字节不是约束 |
| TP4 / TP6（E-sweep） | 59.98 / 48.76 vs 97.10 | 4 个 KV 头无法 6 分；跨岛 NCCL 单价高 |
| **MMVQ 批量上限提高（R202，内核级）** | `c15` 两臂 tg=0 / `PARSE_FAIL`（15-token 验证在 MMVQ 下不出结果）；控制臂 100.07 正常 | 内核不支持该批量 ⇒ 与 hr14（MMQ 太慢 +2.59 ms/token）合起来，**AL 抬升路线彻底关闭** |
| **草稿块扩展（R201，n_max=14，headroom 已修好，四臂同源）** | AL **6.47/6.06/8.24**（↑）但 ms/轮 **73.13/72.94 vs 51.29/51.24**；tg 中位 86.3 vs 100.1 | target 阶段 37.0→**55.1 ms** ⇒ 每 verify token **1.24 → 2.59 ms**（8→15）⇒ 盈亏平衡 p>26%，只有 p3 勉强够 ⇒ **AL 路线关闭** |
| **草稿块扩展（R199，n_max=14/15，当时被 arena 卡住）** | 臂 LAUNCH_FAILED：target 首次 decode 即 `ggml.c:1805 GGML_ASSERT(obj_new)` | 固定 arena 耗尽（与 E6 五次 abort 同源）=> 抬 AL 的这条路目前被 arena 卡住，需先解决 arena 尺寸 |
| **RBSKIP（R199，指针指纹跳过 meta rebuild）** | 控制 50.79/50.78 vs 特性 **51.94/51.81（慢 2.3%）**；scheduler alloc 反而 +10-11%；门未破 | 无收益（且走的是从未真正跑过的 skip 路径）=> 两个 env 必须保持不设 |
| **TP4（R198，FGC 之后重测）** | 均值 ms/轮 **51.86/51.90 vs TP3 50.86/50.91**（慢 2%）；AL 5.58/4.25/6.38 -> 4.23/3.97/6.08 | 权重/卡 9.68 -> 7.25 GB **没买到任何 target 时间**；与 E14 同源 => 28.44 ms 不是权重带宽瓶颈 |
| `--spec-draft-device`；`VLLM_SM70_USE_BREAKABLE_CUDAGRAPH` | 见 HANDOFF §5 | 更慢 |
