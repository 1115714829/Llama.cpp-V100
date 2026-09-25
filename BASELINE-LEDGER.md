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

### R303 ★★★ **P-P3-T GQA2 设计定案（融合 × 打包 = 两轮反证后的缺象限）：grid.z=24 头各一 CTA = K/V smem 6x 流量实锤；P=2 打包（预算内、位级等价）**（2026-09-24）

- **代码级实证（:30-35、:947-949）**：`grid(q_pad/64, nb*hkv, **hkv*gqa=24**)`、`head_q = j*gqa + c` 每 Q 头独立 CTA、各自 smem 载入共享 kv-head 的 K/V（:850 `k + head_kv*k_head_stride`）⇒ **6x K/V 全局流量**（236K 深度时 K/V 每头 121 MB >> L2 40 MB = 真 miss）。
- **缺象限声明**：T2（分解 × 打包）被 S 物化流量反证；M1（79T=分解+打包 @m=2048）= Path A 持平；Path A（融合 × 不打包）= 45.9 TFLOPS；**唯独"融合 × 打包"象限从未被测 = R303 主张**（1cat 梯子思想入融合内核，非分解）。
- **P=2 预算数学**：O 累加器 64 regs/头 ×2 = 128 + Q/K 碎片 + 寻址 ≈ 200 < 256（`__launch_bounds__(256,1)` 无占用率损失）；P=3 超预算 ✗。q_smem ×2 = +32 KB（96 KB/SM 预算内待核 kSmemBytes）。
- **★ 位级等价潜力**：每行 kv 块累加序列完全不变 ⇒ **输出可逐位同 ⇒ 门值免重立**（比 T2 分解形态的根本优势）。
- **预登记（[S4] 沿用 + 本条）**：FA 算子 ≥+20% 或 256K TTFT ≥+7%（175.8 → ≤164 s）；**门值必须原样**（`c4e11b2d`/`f3edac19` 双谱系）；纯步不回退 >2%；显存不升。及格 = 全过；失败入灰。
### R305 ★★ **P-P3-T 三臂 A/B 定稿（fa-gemm-pp3 T4 预登记执行）：Wide 平、Wide+RegP 略负 ⇒ 存量形态全线无增益，整线入灰（含显式 mma 地基——下游客户已尽失）**（2026-09-24）

- **三臂（同栈、双 rep、离散 ≤0.5%、纯步零回归）**：Path A **175.8 s** ｜ Wide（`LLAMA_SM70_FA_GEMM=1`）**176.9 s**（+0.6% 平）｜ Wide+RegP（`+REGP=1`，预登记"被否项组合重测"）**179.2 s**（+1.9% 略负）⇒ [S4] 口径（FA +0~10%）**三臂未达** → 全灰。RegP 单开历史"无增益"与 R272-V2"+1-2ms/轮"（decode 小账）在 256K 预填充口径归零 = 互相印证。
- **整线收边理由**：T2"通用 PV 显式 mma 地基"的下游客户已尽失（GQA 摊销 R303-R 证伪、P-P3 分解 R300 证伪）——地基孤立无 ROI，**P-P3 全家桶（P-P3/P-P3-T/Wide/RegP）就此闭环入灰**。
- **R302c-2 实测（SLOTS=2、双 rep，史存——"改灰不删"补录）**：rep1 183.5 s（预热行军如预期）｜ rep2 177.2 s = 仅 -3.5%——距破线预期差 27 s ⇒ 当时判"残余 miss 源未明"。**R302c-3 定罪**：即 key 的 `op_params[0]` 逐调用变（诊断 n_calls=0 铁证）；嫌疑清单（小图流/尾批/纯步段）全部排除。

### R307 ★★★ **R302c-3 采用（key 剔除 op_params = 收养全命中）：TTFT 171.9 s（-2.9%）、alloc -97%、hit=65/miss=0；★ 洞察 = 气泡搬家（隐藏 GPU 等待在 alloc→compute 间腾挪），下一刀 = n_copies≥2 双缓冲**（2026-09-24）

- **机制账（双探齐）**：`[GALLOC_SLOT] hit=65 miss=0`（收养两形态全命中）｜`alloc 75.3 → 2.06 s（-97%）`｜**但 enqueue 261→315.8 s（+54 s）**——砍掉的调度税里 ~69 s 是**隐藏 GPU 等待（管线气泡）**，墙钟只兑现 9 s/rep。TTFT **171.6/172.1**（离散 0.3%）、纯步 34.3 ms 持平。
- **收养杀手定罪**（诊断跑三段剥）：结构键的 `op_params[0]` 逐调用变 ⇒ key 每调用新 ⇒ activate 恒 -1（`n_calls=0` = [GALLOC_FAST] 零输出即铁证）；剔除后全命中。R302c 首测"槽不够"归因同步修正。
- **战报**：距 BL1 **1.11x → 1.08x**（159.0 vs 171.9）。
- **R308 双缓冲 = 无效（灰，组合条件）**：`GGML_SCHED_COPIES=2` TTFT 171.4/171.9（≈R307 逐位级持平）、enqueue 315.5 s 原封——**气泡等待不存在于 split-copy 面**（n_copies=2 的 ping-pong 无等待可消）。显存 13.1 GB 无恙。
- **★ 重估（第三次刀序翻转，史存 + R309 修正注）**：compute 段 315 s/2rep = 墙钟 ~45%；当时判"host 实打实耗时"——**R309 修正：76% 是 host 被 GPU 同步阻塞（copy 段兜底全同步），非纯 host 计算**（见下条）。
- **R310 meta events 补建 = 无效（灰，史存 + R311 补注）**：委托式事件三件套落库后 copy=489.0 ms 原封——当时判"真拷贝非等待"，**R311 修正：是逐拷贝同步等待，只是 events 只换了 fallback 分支的一半（:1827 未动）**。events 实现已无害留树（caps.events=true 修正同步入）。
- **R318 T2 判决（q8_0 KV 直读 = 显存使能件非性能刀；流量模型臆想被历史数据重锤）**：源码注记（8/23-24 实测）：q4-direct 预填充 -3.6%（窄 u16 块读 > dequant 转换成本）、镜像转换全成本仅 ~0.1s/770s（0.013%）⇒ "KV 流量减半 ⇒ FA +30-60%" = 流量模型错误。q8-direct 模式已实现（`LLAMA_SM70_Q8_DIRECT=1`）⟹ 降级为 T1 显存使能件。

### R324 ★★★ **T1-C 第二硬里程碑：256K 预填充 TTFT 148.57 s ≤ BL1 152.5 s（时序口径达成）——真凶 = 每调用末尾的无条件 `cudaStreamSynchronize` 把四卡同步成串行**（2026-09-24）

- **判决（同源双臂，256K 标准 prompt 236313 tok，ub2048，4 卡，gen4，同源码）**：**k1 引擎 ON = TTFT 148.57 s / pp 1590.6 t/s**；k0 融合核 OFF = **168.59 s / 1402 t/s**（上一轮同源实测；本轮源码差异仅"删同步"，对 k0 惰性）。**BL1 = 152.5 s / 1549 t/s ⇒ 预填充硬指标本轮达标（152.5/148.57 = 1.027x）**。显存 12.96 GB/卡（k0 = 11.96）⇒ 引擎工作区 1.00 GB/卡，仍在 ≤4×V16GB 包络内。
- **★ 严格口径复核（删同步最终版，双臂各 2 rep，同源同 prompt）**：**k1 = 149.243 / 149.256 s（均值 149.25，离散 0.009%，pp 1583）**；**k0 = 167.648 / 168.905 s（均值 168.28，离散 0.75%，pp 1409）**；引擎自身 2 rep 累计 51.902 s ≈ 25.95 s/rep。⇒ **对 BL1（152.26/152.82，均值 152.5）= 1.022x 反超；对自家 k0 = 1.128x**。
- **真凶与机制（今日最贵一课）**：入口内每调用末尾的**无条件流同步**（s15/s2b/stage3/store 四处）令 4 个 head-group 调用（分居 4 卡）**完全串行** ⇒ ① 引擎自身累计 **276.8 s → 25.9 s（10.7x）** ② 每调用白等 ~13-14 ms（7296 调用 ≈ 100 s 空转）+ 四卡算力无法重叠。修复 = **热路径零同步**（只留 `cudaGetLastError` 探针；同步一律 `if (t1c_verbose)`）。
- **配套加固**：异步设备符号拷贝（`g_rows/g_row_max/g_row_sum_out/g_pv_task_base`）的宿主源从**栈局部**改为**每设备 pinned host 槽位**（`cudaMallocHost(&slot.hsym, 24)`）——删同步后栈帧生命周期会悬垂，属必修项而非优化。
- **成本结构实测（删同步前，按 prefix 扫描拟合）**：`t ≈ 14.4 ms + 0.2064 µs × prefix` ⇒ **边际 = 61 TFLOPS/卡**（cuBLAS-RAW QK + CUTLASS FP32 PV ≈ fp16 峰值效率）⇒ **1cat 79T 形态红利成立**，先前 1.81x 落后**纯属我方同步 bug**（不是形态问题、不是算力问题）。
- **数值门取证（R324 附，方法学可复用）**：**宿主参考对拍**（`T1C_REF=1`，纯 C++ 复算 q8_0 反量化 + 因果 softmax + PV，不需要 numpy）+ 分段开关 `T1C_REF_PREFIX_ONLY` / `T1C_SKIP_TAIL` = **单调用级四层二分**（score / num / psum / out）。已定罪并修 **6 项**：① `g_row_sum_out`/`g_row_max` 未接（PV epilogue 解引用空设备指针 = 非法访存 + 后续 memcpy 全 0）② `kt` 尺寸按 kRows 申请（6.3 MB vs 256K 需 134 MB）③ 前缀 QK 的 K 偏移（K 是 dim-major，应为 `+begin` 而非 `+begin*256`）④ V 三段 guard 必须在 PV 之前（否则 PV 读未缩放 V）⑤ 每调用把 rows(12288) 个 float 的 state 回写调用方的 **24-float 栈数组** = 爆栈（上轮"服务假死"真凶）⑥ **PV 融合行和的行映射错位**：`psum` 放大 **12.9x**，反解 eff_shift 命中**别行**的 pmax（row5→row2620、row6→row4）⇒ 自建行和核（`t1c_row_sum_kernel`，Σ_j exp(s−m) 与 PV 同 shift）覆盖后，**首 128 行 maxdiff 0.356 → 6e-4（= half 精度，正确）**。诊断踩坑两条：宿主解引用设备指针（`ws.psum[row]`）= 段错误；扫描计数读回必须 `cudaMemcpyAsync`+流同步，否则得到 `bad > n` 的假计数（4 组并发共用计数器）。
- **残留真 bug（当前前线）**：PV（CUTLASS）**A 操作数行映射被 pitch-linear 线程图置换**。REFPERM 探针（±160 行窗口内找引擎 num 真正属于哪一行）：row100→242、row127→86、row200→242、row6000→6025、row12282→12124；row0/16/60/128/129 = 恒等（对）。**非单射（100 与 200 同指 242）⇒ 不是纯置换，是行间混合**。候选修法：① 切 `PREFIX_PV_WARP_SPECIALIZED` 路径（该核行索引显式线性，且自带正确的行和写回）② 补 A/C 行映射 ③ 换 cuBLAS PV（`P̃` 物化 1.2 GB/块流量 = 不可行，已否）。
- **R325 ★★★ T1-C 数值门通过（2026-09-25）：PV 行映射 = 一行宏缺失；端到端 greedy sha 与融合核逐位一致**

- **一行根因**：`PREFIX_PV_M128_W64_ROW_SUM` 的自动开启挂在 `PREFIX_TORCH_EXTENSION`（prefill.cu:485）下，我方 OBJECT 构建**未定义**它 ⇒ PV 的 A-transform 里 `row_max` 只按 `[element]`（8 槽）填写与使用，而 M128/W64 线程图每线程有 **2 个 contiguous 访问**（tile 内行 0..63 / 64..127）⇒ **每 tile 后 64 行套用了前 64 行的 max**。行级指纹全表吻合：row0/16/60（local<64）恒等正确、row100/127（local≥64）错、row128/129（下一 tile local<64）又正确。
- **修复** = 把该宏补进 36 宏圣经（附注释：为何不能依赖 `PREFIX_TORCH_EXTENSION` 自动开启）。
- **判决（宿主参考对拍，全因果，11 个 (t,h) 组）**：修前 maxdiff = 2.87 / **33.04** / 3.55 / 0.426 / 0.678 / 0.220（row100/127/200/1000/6000/12282）⇒ 修后 **0.00069 / 0.00166 / 0.00164 / 0.00049 / 0.00023 / 0.00062**，全部落在 half 精度档（与本就正确的 row0/16/60/128/129 同级）。
- **端到端门值（新件 `t1c-gate.sh` + `t1c-gate-client.py`）**：固定 100 KB prompt + greedy（temp0/seed1234/n_predict=32）：**引擎 ON sha = `bcda0092bfbaaa9d30db21d4a30ccc1bc1e6e68c3ff3101f86402069a7450bb8`（131 chars）**；**引擎 OFF sha 完全相同** ⇒ 预填充数值端到端忠实（token 级逐位一致）。
- **方法学教训（三条，可复用）**：① 移植 1cat 内核时**宏的组合语义就是代码的一部分**——独立 OBJECT 构建缺 `PREFIX_TORCH_EXTENSION` 会静默关掉一批"自动开启"的正确性补丁；② 数值二分最优序 = **score → num → psum → out** 逐层宿主复算对齐（本轮 7 处 bug 全由此序列定位）；③ **行级指纹**（哪些行对/错）比整体误差更能指认线程图类 bug。
- **踩坑补记（工具层）**：pwsh `Set-Content` 会把脚本写成 CRLF ⇒ 上传后 `bash` 首行重定向带 `\r` 秒退（表现为"tmux 会话立刻消失、日志不生成"）；所有上传脚本一律先 `sed -i 's/\r$//'`。

- **R326 ★★★ T1-C 预填充收官（数值正确 + 时序打平 BL1）：256K TTFT 151.93 s（151.507/152.356）vs BL1 152.5 s；撤多余行和核收回 16.1 s**（2026-09-25）

- **关键发现（省一刀）**：R325 的宏补丁之后，**PV 自带的融合行和已经正确** —— 关掉我自建的行和核（`T1C_OWSUM` 默认关）后，端到端门值依旧逐位一致（`bcda0092…`）⇒ 那份"每调用多读 ~2.9 GB 分数 + 全量 exp"的核可**永久撤下**；撤下后 256K 预填充 **168.06 → 151.93 s（-16.1 s）**，正好等于其代价。
- **最终 A/B（同源双臂各 2 rep；256K 标准 prompt 236313 tok；ub2048；4 卡；自建启动器 `/root/libdir-gb/llama-server`）**：**k1 引擎 ON = 151.507 / 152.356 s（均值 151.93，pp 1555）**；k0 融合核 = 167.160 / 167.723 s（均值 167.44，pp 1411）；**BL1 = 152.26 / 152.82（均值 152.5）** ⇒ 对融合核 **1.102x**、对 BL1 **1.004x**；引擎内部累计 **26.2 s/rep**（与修前 25.9 一致 ⇒ 宏修本身零成本）。
- **口径诚实标注**：对 BL1 仅领先 0.57 s，而两臂各自离散 0.6-0.9 s ⇒ **臂间区间重叠，"打平（略优）"而非稳健反超**；要稳健需再挖 3-5 s（候选：反量化开销、kBlockN 调档、PV 已修路径的额外归约）。
- **环境隔离修正（用户质询后落实）**：本轮起 `t1c-run.sh` 改用**自建启动器**（不再借正式树二进制）；并已实证 `/root/llm/llama.cpp`、`/root/llm/systemd` 自 9/20 起**零文件改动**、当前无 llama/vllm 服务在跑、构建产物只落 `/root/libdir-gb`。**遗留风险提示**：正式单元用 `CUDA_VISIBLE_DEVICES=2,5`，测试用 `0,1,2,3` ⇒ **GPU 2 卡位重叠**（本轮正式服务未运行，未冲突）。
- **预填充线剩余（判定为"够用但不宽裕"）**：数值 ✅ 门值 ✅ 时序 ✅（打平），余量不足 4 s ⇒ 采纳前的可选加固 = 再挖 3-5 s；**吐字线（tg 12-13 vs BL1 124.6 t/s）完全未动 = 剩下最大且最难的战线**。

- **R327 ★★★ 吐字线根账（减法分离）：解码每轮 90.6 ms = host ~52-58 + GPU ~33-39；与 BL1 的 4.17x 分解闭合为 host 1.91x + 字节 1.44x × AL 1.25x**（2026-09-25）

- **方法**：`LLAMA_ROUND_TIMING_SYNC=1`（树内既有开关，`llama-context.cpp:1361`）令 compute/sync 计数器给出真 GPU 等待；**减法分离** = 同形状 256K 跑 gen=128 与 gen=1 各一次相减（118 vs 158 rounds），剔除预填充段。
- **每轮分账（256K，40 个解码轮）**：target `graph_compute` 内 **host 提交墙 20.8 ms** + **GPU 等待 33.0 ms**；**target ctx 之外 36.8 ms**（= draft 2 次调用 + CPU selector + CPU sampler + 槽位/HTTP）；build/alloc/setin 仅 0.1/1.6/≈0 ms（图缓存复用正常）。
- **判定**：解码为 **host-bound**（~52-58 ms host vs ~33-39 ms GPU）。与 BL1（27.2 ms/轮 = 8.0 ms/token × AL 3.4）对账：**host 段 = BL1 整轮的 1.91x**；**GPU 段差 = 每轮字节 43 GB（Q8_0 27 + Q4_K_M 16）vs ~30 GB（NVFP4 15 + NVFP4 draft 15）= 1.44x**（1cat README 自述生产形态 = **Qwen3.8-27B-NVFP4 + DFlash2**；BL1 单元实读 = FP8 目标 + DFlash2 draft，卡 0,1,3,4）。
- **追平路线（两刀）**：① **砍 host 段**（P1 图/管线 + P3 GPU selector + P4 GPU sampler + P-D1 meta 容器 + P-D3 两 draft 合一）⇒ 轮 90.6 → ~40 ms、tg 29.9 → **~85 t/s**；② **字节/格式**（NVFP4 同构 GEMV 或抬 AL）⇒ GPU 39 → 27 ⇒ tpot 8.0 达标。**红线允许**：「参数可以等价齐平」= BL1 的 FP8/NVFP4 形态可对齐。
- **注意（E14 判例）**：llama.cpp 的 MMVQ 走 Q4_K_M **曾更慢**（58.6 vs 55.8 ms/轮）⇒ 字节刀必须配**融合/M=1 专用 GEMV**（1cat marlin FP4 形态），不是裸换量化。
- **AL 波动口径**：同构建两次跑 AL 2.76 / 3.12（内容相关）⇒ tg 必报 AL；每轮墙钟（90.5/90.6 ms）为稳定量。

- **R328 ★★ 吐字线第一刀实验（P-D1 重测）：CUDA 图未被 MoE 禁用、基线 replay 已 63%；`GGML_META_REBUILD_CACHE` 收益 -2~12%（噪声内）⇒ P-D1 降级为小头寸，主因仍在 CPU 段 + draft 双调用**（2026-09-25）

- **判据修正**：上条(R327 附注)「图 warmup 永远完不成、92% 直发」系 **5K 探针误读**（该探针的计数含预填充子图，直发占比被稀释）；25K 口径实测基线 replay=63%。**已修正，不删**。
- **三臂实测（25K prompt / gen96；判据用每轮墙钟，tg 受 AL 内容波动不可比）**：A `GGML_META_REBUILD_CACHE=1` = **76.0 ms/轮**（replay 67%）；B `+GGML_META_REBUILD_PTR=1` = **68.6 ms/轮**（replay 71%）；C 基线 = **77.5 ms/轮**（replay 63%）。
- **机制实证**：`mul_mat_id needs sync = 0`（Q8_0 在 Volta 的 mmid 上限 4，但 `should_use_mmq` 兜底 → 图不被禁）；`props changed` 11-12 次（打印条件 n<6 或 n%1000）⇒ 仍有高频属性抖动；图统计 `[GRAPH] calls/capture/replay/direct` = **子图粒度**（meta 后端每 ubatch 跑多个子图，`ggml-cuda.cu:5075` 注释可证）。
- **刀序修正**：**P1（图/容器）降为 -2~5 ms/轮 头寸**（与 1-计划文档原账一致）；主刀改判 = **P3 GPU selector（+5 ms/轮 + 一次 host 往返）+ P4 GPU 采样 + P-D3 两 draft 合一**；其次 P5/P6（融合 GDN / fused W13）压 GPU 段。
- **工装**：`t1c-run.sh` 支持任意 env 透传（未白名单变量直接进服务）；图诊断 `GGML_CUDA_GRAPH_DEBUG=1` 可得 `[GRAPH] props changed / capture/replay/direct` ✓ 已纳入测量法。

- **R329 ★★★ 吐字线真凶二号：meta 后端每 ubatch ~126 个子图 × ~165 µs host = 20.8 ms/轮提交墙；图 key = nodes[0] 指针 ⇒ 内容键化缺失**（2026-09-25）

- **每轮分账（LLAMA_SPEC_TIMING 实测，25K/gen96）**：`spec timing: draft_decode=17.73 + selector=2.29 (fetch 0.10/topk 0.71/gate 0.63) + walk=0.08 + spec overhead 0.60` = **draft 侧 20.7 ms/轮**；target 侧 compute wall ≈55（host 20.8 + GPU ~34）⇒ 合计 ≈77.7 = 实测 77.5 ✓✓ **账闭合**。
- **`GGML_CUDA_DIRECT_DEBUG` 归因**：`old_ne=[0,0] old_data=(nil)` ⇒ 每次与**空白图**比较 = 图对象每次 ubatch 新建；`ggml_cuda_graph_get_key = cgraph->nodes[0]`（ggml-cuda.cu:2739）= **指针作 key**，容器重建即换 key ⇒ warmup 永远差一次稳定 ⇒ 63% replay 全来自 llama 侧 `reuse` 的那部分。
- **算术闭合**：meta 后端每 ubatch **~126 个子图**（calls=16384/130 ubatch；`ggml-cuda.cu:5075` 注释"the meta backend runs many graphs per ubatch"）× **~165 µs host/子图** ⇒ **20.8 ms/轮** ✓ = target 提交墙真身 ⇒ **P1 真刀 = 子图粒度/数量（合并子图）+ 内容键化**，而非图复用率。
- **P3（GPU selector）现状**：`GGML_SPEC_SELECTOR_INGRAPH=1` **在 load_model 即崩**（TP 下 lm_head 被切分，图内 selector 无法构建——代码注释已预告）⇒ P3 须照 1cat `TP4_M1_FAST_SELECTOR` 实做 TP4 GPU selector；且实测 selector 仅 2.29 ms/轮 ⇒ **P3 头寸下调**，主刀回到 **P-D3（draft 2 调用合一，draft_decode 17.73 ms/轮 大头）+ P1 子图合并**。
- **工装**：`LLAMA_SPEC_TIMING=1`（speculative.cpp:1190/1601/3261）= 每轮四段分账；`GGML_CUDA_DIRECT_DEBUG=1` = 图属性逐字段归因 ✓ 双双纳入测量法。

- **R330 ★★★ 吐字线真凶三号：draft KV 注入的特征 gather = **10.47 ms/call**（copy 0.48 / submit 2.60 ⇒ 同步税而非带宽）；P-D3 真身 = 特征路径去 gather**（2026-09-25）

- **实测（`inject timing`，LLAMA_SPEC_TIMING，25K/gen96）**：`gather=10.47 + copy=0.48 + submit=2.60 + wait=0.00 ms/call（layers=5 tok=60.27）`。
- **机制**：每轮 DFlash2 的 draft KV 注入 = 从 target 抽 5 个 extract 层、~60 token（= draft 上下文窗口 64，启动日志 `capping draft context ubatch ... 64 (block draft)`）逐层 `llama_get_embeddings_layer_inp` + memcpy；6.1 MB 纯拷贝应 ~0.5 ms，实测 10.47 ms ⇒ **20x 差 = 逐层提取的同步/搬运**（不是 PCIe 带宽：copy 段 0.48 ms 可证）。
- **P-D3 真身修正**：不是"两 decode 合一"（token/embd 互斥 E4 阻断），而是**特征路径去 gather**：target 侧 extract 层输出留 device、注入走 GPU（或并入 block decode 图）。1cat 对应 = `DFLASH2_FUSED_GDN_METADATA/VERIFY` + 图内 selector（特征全程在 GPU）。
- **头寸**：gather 10.47 ms/call × ~1 次/轮 = **~10 ms/轮**；叠加 P1 子图合并（20.8）+ P-D3 图合一（draft host ~12）= 三件合计 **~33-43 ms/轮**（当前 77.5→~40）。
- **下一步实施序**：① 特征 gather 去同步（把 `llama_get_embeddings_layer_inp` 的逐层提取改批量/异步，或 target 侧直接出 GPU 指针）② meta 子图合并 + 内容键化 ③ draft 两调用图合一。

- **R331 ★★ 吐字线：gather 10.44 ms = 一次全上下文同步（等 target GPU 尾巴）；同步上提为诊断零收益，真刀 = 设备侧特征路径**（2026-09-25）

- **实测（同步上提 + 计时分离）**：`inject timing: sync=10.44 + gather=0.07 + copy=0.49 + submit=2.59 ms/call` ⇒ 5 层 getter 内部同步在上提后变 no-op（gather 10.47→0.07 ✓），**10.44 ms 全在 `llama_synchronize(ctx_tgt)`**（`llama-context.cpp:4052` 的 `ctx->synchronize()` = 全上下文等 GPU）。
- **机制判定**：`extract_layer_inputs`（`llama-context.cpp:2288`）用 `ggml_backend_tensor_get_async` 把 5 层特征 D2H，但排在 ubatch 计算**之后** ⇒ 等拷贝 = 等整条流 ⇒ **同步在当前结构下不可省**（上提本身零收益 = 已实证）。
- **P-D3' 真刀定案 = 设备侧特征路径**：特征随 verify 图产出并留在 device，注入侧（draft）直接消费 device 指针（1cat `DFLASH2_FUSED_GDN_METADATA/VERIFY` 同构）；或增量注入（每轮只注新增 8 token 而非整窗 60）。
- **头寸**：sync 10.44 ms/轮；叠加 P1 子图合并 20.8 + draft 两调用 host ~12 ⇒ 合计 ~43 ms/轮（77.5 → ~35 理想值）。
- **改动记录**：`common/speculative.cpp` 加同步上提 + `sync=` 计时（诊断基线保留）；`p3-build.sh` 新增 `/tmp/p3-speculative.cpp` payload 位（可复现构建）。

- **R332 ★★★ 吐字线第一刀落地（零代码 env）：`GGML_META_SUBGRAPH_CAPTURE=1` ⇒ 25K 每轮 77.9→68.1（-12.6%）、256K 每轮 90.5→86.8（-4.1%）、tg 29.9→33.9（+13%）**（2026-09-25）

- **机制（树内现成开关，`ggml-backend-meta.cpp:2254/:2826` "F3: capture each (subgraph, device) compute as its own graph and replay it"）**：meta 后端逐子图 CUDA graph 捕获/回放 ⇒ 每 ubatch ~126-360 个子图的重复发射被回放替代。
- **四臂扫描（25K/gen96，判据每轮墙钟）**：基线 75.6 / `FULLGRAPH=1` **79.7（+4.1 有害，弃）** / `SUBGRAPH_CAPTURE=1` **62.0** / 两者同开 71.0 ⇒ 采纳 **SUBGRAPH_CAPTURE**、FULLGRAPH 关闭。
- **同源 2 臂 ×2 rep 复测（25K）**：J（SC）70.6/65.6 = **68.1 ms/轮** vs K（基线）74.0/81.9 = **77.9 ms/轮** ⇒ **-9.8 ms/轮（-12.6%）**。
- **256K 同口径确认（gen128，2 rep）**：每轮 **87.7/86.0 = 86.8**（基线 90.5）⇒ **-4.1%**；tpot 33.36→29.5（-11%）；tg 29.9→**33.9（+13%）**；AL 2.76→2.95。**收益随 GPU 占比稀释**（25K -12.6% vs 256K -4.1%），如实标注。
- **剩余缺口判定（防误判）**：256K 每轮 86.8 vs BL1 27.2 = **3.2x**；host 侧再清零也只到 ~50 ms（1.8x）⇒ **GPU 刀不可免**：字节 43 GB（Q8_0 27 + Q4_K_M 16）vs BL1 ~30 GB = 1.44x，但**验收必须 Q8_0 全量**（目标硬纪律）⇒ 字节刀受限，须靠**效率**（23%→40%+：fused W13/W2、fused GDN、M=1 GEMV）+ **AL↑**（2.95→3.4+）补齐 1.8x。
- **改动**：`t1c-run.sh` 默认开 `GGML_META_SUBGRAPH_CAPTURE=1`；`FULLGRAPH` 明确关闭（附判据）。

- **R333 ★★★ 吐字线 GPU 侧判定翻转：我方带宽效率与 BL1 同级（1049 vs 1100 GB/s 聚合）⇒ 差距 = host 36 ms + 非矩阵乘项；矩阵乘实测 678-820 GB/s/卡（非瓶颈）**（2026-09-25）

- **差分口径（可靠）**：25K 每轮 77.5（host 36 + GPU 41）vs 256K 90.5 ⇒ **KV 10x 只值 +13 ms/轮** ⇒ 轮成本以 KV 无关项（权重/GDN/host）为主。
- **带宽核算**：我方 43 GB/41 ms = **1049 GB/s 聚合（262/卡）**；BL1 30 GB/27.2 ms = **1100 GB/s（275/卡）** ⇒ **同级** ✓。而 decode matmul 实测 **678-820 GB/s/卡**（R212 profile，2.7-3.3 TB/s 聚合潜力）⇒ **矩阵乘效率不是瓶颈**，轮内 matmul 占比不足。
- **分解定案**：每轮 86.8（256K+SC）≈ host ~36 + matmul ~13-20 + **非矩阵乘（GDN 48 层 + 注意力 + 小算子 + 发射间隙）~20-30** ⇒ **达标路径 = P5 融合 GDN + P6 fused W13/W2（抬 matmul 占比）+ host 收尾**；AL（2.95→3.4）为放大器。字节刀受 Q8_0 验收纪律封顶。
- **工装警示**：`GGML_CUDA_OP_TIMING=1` 的 `total/avg` **不可靠**（事件复用致负值/万亿 ms）；仅 `gpu_total=14.765 ms/ubatch` 与 call 计数可信。后续逐算子分账改用 **test-backend-ops**（R212 配方）或差分口径。
- **下一步**：P5（GDN 融合，1cat 四件套同构）+ P6（fused W13/W2 + M=1 GEMV）；两者都以门值 sha + 每轮墙钟双验。

- **R334 ★★★ 吐字线 GPU 刀施工序被逐算子真值重排：P7 decode 注意力 = 头号（194 GB/s，~12 ms/轮）；GDN 状态算子仅 6.9 µs（P5 状态融合头寸可忽略，真身 = GDN 小投影 37 GB/s）；权重 GEMV 已 680-765 GB/s（P6 头寸小）**（2026-09-25）

- **工具**：`test-export-graph-ops -m <Q8_0> -o /tmp/ops.txt`（91 独立算子：50 prefill + **41 decode**）+ `test-backend-ops perf --test-file /tmp/ops.txt -b CUDA0`（R212 配方）= 逐算子真值 ✓（`GGML_CUDA_OP_TIMING` 已判不可用）。
- **decode（M=1）真值**：`FLASH_ATTN_EXT`（256K KV f16）**5147 µs/1.05 GB/194 GB/s**（峰值 900 的 22%）；`result_output`（LM head）**1540 µs/1.32 GB/817 GB/s**；`ffn_gate/ffn_out/Qcur_full/node_13/z-0/linear_attn_out` = **681-765 GB/s**（高效 ✓）；`node_36`（GDN 投影 48 列）**37.5 GB/s**；`GATED_DELTA_NET` **6.92 µs**、`SSM_CONV` 2.43 µs（状态算子极小 ✓）。
- **256K 每轮 86.8 ms 闭合分解**：host ~36 + 权重 GEMV ~25（含 GDN 小投影低效）+ **注意力 ~12**（q8_0 KV 下 0.53 GB/call ≈ 2.7 ms × 16 全注意力层，与 R333 差分 +13 ms 吻合 ✓）+ **LM head ~6.4**（TP4 分片后 0.4 ms/位置 × 16 位置）+ 小项 ~7。
- **施工序重排（数据驱动）**：① **P7 decode 注意力**（194→600 GB/s ⇒ 12→2.6 ms/轮，**-9**；1cat = `sm70_grouped_long/scalar-attention.cu` + `E4M3_LONG_ATTENTION_MANIFEST`）② **host 收尾**（-20~25）③ **P5 GDN 小投影融合**（`node_36` 37 GB/s ⇒ 融合进大 GEMM = `FUSED_GDN_INPUT_FP16` 同构）④ **P4 LM head**（top-1 融合/少位置 ⇒ 6.4→2）。
- **判据修正（不删史）**：P5"状态融合"（GDN 状态算子）头寸可忽略（6.9 µs/层级）；P6"权重 GEMV 形态"头寸小（已 680-765 GB/s）；两者改为"小投影/融合"与"形状调优"口径。
- **红线提醒**：Q8_0 全量 = 验收硬纪律 ⇒ 字节刀（NVFP4 30 GB）不可用；上述四刀全在"效率/结构"面。

- **R335 ★★ P7 首攻（decode 注意力入引擎）判负并安全回退：cuBLAS QK 对小 rows 报 status 13（EXECUTION_FAILED）且波及 load_model warmup ⇒ 门值 g1 臂崩；回退后 g0=g1=bcda0092 ✓ 零回归**（2026-09-25）

- **动刀**：① `fattn-sm70-d256.cu::supported()` 放宽 `min_q`（decode 形状准入）② `prefill.cu` 入口放宽 `query_len % 8`（q 1..8 准入）③ cuBLAS 算法选择改 rows-aware（`rows%8==0 ? ALGO9_TENSOR_OP : DEFAULT`，**保留**（prefill 无影响））。
- **判据（门值）**：g1（引擎臂）`launch cuBLAS raw QK: cuBLAS status 13`，随后 `ggml_abort`（`ggml_cuda_error`）于 `server_context::load_model`（warmup 前向）⇒ **门破 = 不采用**（staged-baseline 铁律 3）✓ 当场撤刀。
- **二分结论**：放松本身即根因（含 warmup 形状被误准入）；`rows%8` 算法选择非根因（prefill rows=12288 走 TC 不变）。**回退后 g0=g1=bcda0092**（131 字符）✓ **已知好态恢复、生产路径零回归**。
- **P7 下一步（改图不弃图）**：decode 形状需要**专用小-M QK 路径**（M=6..48 的 GEMV 形态，1cat `sm70_grouped_long/scalar-attention.cu` 同构），或 cuBLAS 小-M 配置 + 入口加 `rows >= kMin`/`kv >= kMin` 下界（防 warmup 误入）；另注意 `kv_len % kBlockN != 0` 的尾块覆盖。
- **工装新增**：门值 = P7 类改动的**第一判据**（引擎臂崩 = 当场撤），`t1c-gate.sh` 一次 ~2 min ✓ 已是标准流程。

- **R336 ★★ P7 二攻：q=1（rows=6）正确性过门（带 env 门值 g1=g0=bcda0092 ✓），但 q=2（rows=12）cuBLAS QK 仍 status 13 ⇒ 算法阈值非真因（DEFAULT 亦败）⇒ 真因在 GEMM 尺寸/指针设置；env 门控保生产安全**（2026-09-25）

- **正确性（关键里程碑）**：`LLAMA_SM70_79T_DECODE=1` 下门值 **g1 = g0 = bcda0092…**（131 字符）⇒ **decode 注意力经 T1-C 引擎与融合核逐位一致（q=1 口径）** ✓✓。
- **性能 A/B（25K/gen96，2 臂 ×2 rep）**：arm P（decode 入引擎）**崩**（spec verify 的 q=2 形状 `launch cuBLAS raw QK: cuBLAS status 13`）；arm Q（基线）1901.5/2310.6 ms 正常 ⇒ **P7 未达性能判定条件**（门值已过但形状未全绿）。
- **根因修正（不删史）**：R335 的"`rows%8`/`rows>=128` 阈值"**不是真因**——`CUBLAS_GEMM_DEFAULT` 对 rows=12 同样 EXECUTION_FAILED；真因在 GEMM 的尺寸/指针相关设置（q=1 rows=6 ✓ 过、q=2 rows=12 ✗ 败 = 尺寸相关）。下一步 = 在 `check()` 失败点打印 m/n/k/lda/ldb/ldc + A/B/C 指针与 `ws.qt/ws.kt/ws.scores` 尺寸，做 q=1 vs q=2 的最小对照。
- **安全线**：decode 准入 = `LLAMA_SM70_79T_DECODE=1` env 门控（默认关）⇒ 生产路径零影响 ✓（stress arm Q 基线正常可证）。
- **工装**：门值可带 env 跑（`LLAMA_SM70_79T_DECODE=1 bash t1c-gate.sh`）= P7 类改动的标准正确性检验 ✓。

- **R337 ★★★ P7 真因定案：cuBLAS fp16 小 stride 对齐约束（lda=12 非 8 倍数）——TC status 13、SIMT ALGO0 status 15 双拒；prefill lda=12288 是 8 倍数故从不触发；修法 = 行填充到 8 倍数或专用 decode GEMV 核**（2026-09-25）

- **诊断（QKFAIL 打印，`prefill.cu` 的 `check()` 失败点）**：
  - 第一击：`status=13 rows=12 width=2 k=256 lda=12 ldb=256 ldc=12 algo=-1`（cuBLAS 默认启发式 = TC 路径，拒小 m/n）。
  - 第二击：`status=15 rows=12 width=254 k=256 lda=12 ldb=256 ldc=12 algo=0`（**SIMT ALGO0 亦拒** = NOT_SUPPORTED）。
  - ⇒ **共性 = lda=ldc=12（rows=q*heads_q=12，非 8 倍数）**；prefill rows=12288 是 8 倍数 ⇒ 从不触发 ✓。与 width 无关（2 与 254 都败）。
- **修法定案（二选一）**：① **行填充到 8 倍数**（rows → (rows+7)&~7；workspace kRows=12288 免费；softmax/store 只认有效行；掩码按 row/heads_q 映射天然兼容）② 专用 decode GEMV 核（1cat `sm70_grouped_long/scalar-attention.cu` 同构，无 cuBLAS 约束）。
- **正确性地基（保持）**：q=1（rows=6）门值 g1=g0 ✓（decode 注意力经引擎逐位一致）；decode 准入 = `LLAMA_SM70_79T_DECODE=1` env 门控（默认关 ⇒ 生产零影响 ✓）。
- **下一步**：实施修法①（行填充）→ 门值（含 spec 形状）→ 性能 A/B（预期 **-9 ms/轮**）→ 采用则入采用链。

- **R338 ★★★ P7 真因第二层：cuBLAS 唯一可用组合 = Tensor Core + m/n 皆 8 倍数；decode tail 块 width=254（非 8 倍数）是最终拦路石（m 填充已解决 rows 侧；prefill width=24576 从不触发）**（2026-09-25）

- **实测链**：rows=12/width=2/ALGO9(TC) → status 13；rows=12/width=254/ALGO0(SIMT) → status 15；**rows=16(已 8 倍数)/width=254/ALGO0 → 仍 status 15** ⇒ ALGO0+compute_16F 组合不支持 ⇒ 可用组合只剩 **TC 且 m/n 皆 8 倍数**。
- **已落的半程修复（保留）**：`rows = (query_len*heads_q + 7) & ~7` 行填充 + `g_rows = rows_valid`（store 只写有效行 ✓）；spec 形状门值 **g1=g0=bcda0092 ✓**（q=1 与 spec verify 双覆盖）——**正确性地基全绿**。
- **最后一块拼图（实施方案已定）**：tail 块 `width` 填充到 8 倍数 + **把填充列掩码到 -inf**（`t1c_tail_tri_mask` 同处注入 ✓）；prefix 块 width=24576 天然合规。或改用引擎自带 QKDirect/QKRaw 内核（绕开 cuBLAS 约束，兼得 q8_0 直读）。
- **风险控制**：decode 准入仍 `LLAMA_SM70_79T_DECODE=1` env 门控（默认关 ⇒ 生产零影响 ✓）；每步门值（spec 形状）先行，性能 A/B 后置。
- **工装沉淀**：`QKFAIL` 诊断打印（status/m/n/k/lda/ldb/ldc/algo）= cuBLAS 类问题的标准定位法 ✓。

- **R339 ★★★ P7 判负（性能）：decode 注意力走 T1-C 引擎 = 每轮 96.7 vs 基线 70.6 ms（**+37% 更慢**）⇒ 不采用（env 保持关）；根因 = 引擎为 prefill 大 rows 设计，decode 小 rows 下每调用固定开销（11 个 prefix 块 × cuBLAS+PV+合并）压过带宽收益；预期 -9 ms/轮**不成立**（已修正）**（2026-09-25）

- **正确性（已证，保留为资产）**：行填充 + width 填充 + TC 算法后，**spec 形状门值 g1=g0=bcda0092 ✓**（q=1 与 spec verify q=2..8 双覆盖）⇒ 引擎的 decode 路径**数值正确** ✓（`LLAMA_SM70_79T_DECODE=1` env 门控、默认关）。
- **性能（25K/gen96，2 臂 ×2 rep，判据每轮墙钟）**：V（引擎）98.0/95.3 = **96.7 ms/轮** vs W（融合核）71.6/69.7 = **70.6 ms/轮** ⇒ **+37% 更慢，不采用**（staged-baseline 铁律 2/判读纪律）。
- **根因**：引擎的每调用结构（prefix 块循环 cuBLAS QK + FP32 PV + stable 合并 ×11）为大 rows 优化；decode rows=6..48 时固定开销主导 ⇒ 带宽收益被淹没。**P7 的正确形态 = 专用小-M GEMV 核**（1cat `sm70_grouped_long/scalar-attention.cu` 同构），不是复用 prefill 引擎。
- **判据修正（不删史）**：R334 的"P7 头号刀 -9 ms/轮"**预估作废**（被本条实测推翻）；decode 注意力的头寸改为"专用核后重估"。
- **资产沉淀**：cuBLAS 约束链（TC 需 m/n 皆 8 倍数；SIMT 拒 compute_16F）+ `QKFAIL` 诊断打印 + 行/宽填充手法 ✓ 均可复用。

- **R340 ★★★ 吐字线 host 段定案：SC 的收益全在 draft 侧（draft_decode 19.58→9.55）；target host 提交墙 23.1 ms/轮（裸提交 ~21.5）纹丝不动 = 当前 #1 刀**（2026-09-25）

- **减法分离（SC 采纳后，25K，gen128−gen1 共 40 解码轮）**：build 0.1 + alloc 1.4 + setin 0.05 + **enqueue 23.1**（其中裸提交 ~21.5）+ GPU 等待 13.8 = target 段 36.9；ctx 外 ~31（注入 sync 10.4 + submit 2.6 + draft block 15.1 + selector 2.3）⇒ 合计 ~68 ✓ 与实测吻合。
- **判读**：`GGML_META_SUBGRAPH_CAPTURE` 的 -12.6% 来自 **draft 侧图回放**（draft_decode 19.58→9.55）✓；**target host 提交 23.1 ms/轮未被触及** ⇒ ~2000 节点 × ~11 µs = meta 容器逐节点/逐 split 开销（`GGML_META_FULLGRAPH` 合并已测无效/有害，R332）。
- **#1 刀（下一实施）**：**meta 容器 target 段的逐节点/逐 split 开销**（23.1 ms/轮）——方向：① split 合并/批量化 ② 逐节点镜像构建跳过（内容键化 ✓ P-D1 原案）③ 拷贝段批量化（R311 族手法移植到 decode）。
- **次刀**：注入特征 sync 10.4 ms/轮（设备侧特征路径，P-D3'）；draft block 15.1（含 GPU ~5.9 = 16 GB 权重）。
- **GPU 侧现状**：25K GPU 等待 13.8 ms/轮（256K 约 33 ✓ KV 差）⇒ GPU 刀（P5/P4/专用 decode 核）在 host 刀之后。

- **R341 ★★ SC+REBUILD_CACHE 组合无叠加收益（68.2 vs SC 单开 68.1，噪声内，不采用）；meta 逐节点开销机制定位 = split-state 缓存查表/比对 + 拷贝分发（2000 节点 × ~11 µs ≈ 23 ms/轮）**（2026-09-25）

- **组合探针（25K/gen96，2 臂 ×2 rep，判据每轮）**：X（SC+REBUILD_CACHE）68.9/67.4 = **68.2** vs Y（SC）79.2/64.9 = **72.0**；对照 R332 的 SC 单开 68.1 ⇒ **REBUILD_CACHE 在 SC 之上零叠加**（R332 判读复证 ✓）⇒ 不采用。
- **机制定位（代码实读）**：meta 的逐节点工作 = `ggml_backend_meta_get_split_state`（`split_state_cache` std::map 查表 + memcmp + miss 时 `calculate_split_state()`）+ 拷贝分发 + 子图循环（`run_subgraphs`）；`GGML_META_DEBUG` 只打 split-state 缓存事件（`ggml-backend-meta.cpp:1256`）**无逐段计时**。
- **#1 刀实施方向（更新）**：① **逐节点工作减少** = split-state 缓存改扁平哈希/免 memcmp ② **融合减少节点数**（P5 GDN 投影 / P6 fused W13 = 同时减 host 与 GPU ✓✓ 最干净的杠杆）③ 拷贝段批量化（R311 族手法）。
- **判据警示**：两臂离散大（Y 79.2/64.9 = 14.3 ms）⇒ 25K/gen96 的 AL 内容波动让 ms/轮 带 ±7 ms 噪声 ⇒ 后续 A/B 用 **≥2 rep + 对照臂差 <1%** 的 ABBA 或加 rep 数。
- **注入 sync 观测（顺带）**：6.2-14.9 ms/波动（随 GPU 负载）⇒ 真实依赖确证（P-D3' 设备侧路径仍是次刀）。

- **R342 ★★★ 归因再修正：`run_subgraphs` 19.58 ms/call 大部分是提交时的 GPU 队列阻塞（非纯 host）；FULLGRAPH 合并令调用数骤降但轮墙钟不变（72.3 vs 68-72）⇒ R340 的"host 23.1 = #1 刀"归因过强，实为"串行 GPU 链 + 阻塞式提交"**（2026-09-25）

- **新仪表（保留）**：`GGML_META_TIMING=1` ⇒ `[MT] run_subgraphs <us>/call`（两处 `run_subgraphs()` 调用点累计）✓。
- **实测**：基线 `[MT] run_subgraphs 19580 us/call`（与 [RT] enqueue 23.1 ms/轮 同量级 ✓ 一致）；`FULLGRAPH=1` 后调用数骤降（打印未触发 = <256 次）但**每轮 72.3 vs 68-72 无改善** ⇒ 提交墙 = **GPU 队列阻塞为主**（异步提交在 GPU 忙时阻塞 ✓）。
- **判读修正（不删史）**：R340 的"target host 提交墙 23.1 = #1 刀"**降级**：解码轮 = 3 个串行 compute（target verify + 注入 + draft block）各自的"提交 + GPU 等待"，GPU 工作 ~24 ms（25K）+ 提交阻塞 ⇒ **真刀 = ① 压 GPU 工作（融合 P5/P6、格式）② 管线重叠（去串行）③ AL↑**；纯 host 侧（子图合并/键化）收益有限（FULLGRAPH/REBUILD_CACHE/SC 三者实测已证）。
- **⚠️ 方法论教训**：**提交墙 ≠ host 税**（异步提交的阻塞会伪装成 host）；判 host 必须用"提交 - GPU 等待"或合并实验的墙钟 ✓（本条即为该判据的反例修正）。

- **R343 ★★ P6 勘察定论：融合 W13 已生效（`build_moe_ffn` 走 `gate_up_exps`，R334 逐算子 `node_13`=[10240,1]=2×5120 实证）⇒ P6 头寸小；解码剩余真刀 = 管线重叠（去串行）+ draft 侧（28 ms/轮最大单段）+ AL**（2026-09-25）

- **P6 scoping（代码实读 + 逐算子实证）**：`llama-graph.cpp:2183` `if (gate_up_exps) { build_lora_mm_id(gate_up_exps, ...) }` ✓ 融合 W13 一次 `mul_mat_id` ✓；R334 的 `node_13`=[10240,1]（=2×5120）✓ `z-0`=[6144,1]（down）✓ = **融合已跑**。⇒ P6"融合减节点/GPU"头寸**基本已拿**，剩余只有形状调优级。
- **解码剩余刀序（R342 修正后）**：① **draft 侧 28 ms/轮**（注入 sync 10.4 + 注入 submit 2.6 + draft block 15.1）= 最大单段 ⇒ **管线重叠/设备侧特征路径（P-D3'）**② AL↑（2.95→3.4）③ 专用 decode 注意力核（P7 正确形态）④ P5 GDN 小投影（`node_36` 37 GB/s，头寸小）。
- **方法论沉淀**：**动手前先 scoping**（P6 若直接开工会白做 ✓ 本轮用代码实读 + 逐算子名实证避免）✓。
- **当前吐字线账（25K，68 ms/轮）**：target compute 36.9（提交阻塞 ~23 + GPU 13.8）+ 注入 13 + draft block 15.1 + selector 2.3 ✓；256K 86.8（GPU +13 KV）。

- **R345 ★★ P-D3' 半程实施：设备特征 getter 链落地（`embd_layer_inp_dev` + `llama_get_embeddings_layer_inp_dev`，env `LLAMA_SPEC_DEVFEAT`）；工装缺口发现并修复（p3-build.sh 原不装 llama-context.*/llama-ext.h ⇒ 本轮改动一度是假绿）**（2026-09-25）

- **已落（build 真绿 ✓）**：`llama-context.h` 成员 `embd_layer_inp_dev` + getter 声明；`llama-context.cpp` 的 resize + `extract_layer_inputs` 的 devfeat 分支（记录 `t->data` 设备指针、跳过 D2H）+ `get_embeddings_layer_inp_dev`；`llama-ext.h` C API 声明；`llama-context.cpp` C API 实现（**无 `synchronize()`** ✓）。
- **剩余（下一轮）**：① 注入的 **5 层拼接设备缓冲**（batch 期望 [n_chunk, n_embd_enc] 连续 ⇒ 设备侧 concat 或 5×D2D）② `speculative.cpp` 注入段的指针直通（去 memcpy + 去 `llama_synchronize`）③ 门值 + `LLAMA_SPEC_TIMING`（sync→~0）+ 每轮 A/B（预期 **-8~11 ms/轮**）。
- **⚠️ 工装缺口（已修）**：`p3-build.sh` 的 payload 位只覆盖 ggml-cuda + speculative ⇒ `src/llama-context.*`、`src/llama-ext.h` 的改动**不进 build**（"BUILD_RC=0" 假绿）⇒ 已补 3 个 payload 位 ✓。**教训：新改文件必须确认在 payload 名单内，否则 build 绿 = 无效判据**。
- **门值纪律**：P-D3' 完成后必须过 `SPEC=1 bash t1c-gate.sh`（greedy sha）✓ 第一判据。

- **R346 ★★★ 战略账闭合：追平路径 = host 清零 + AL↑（字节刀被 Q8_0 验收封顶）；AL 的具体落点 = BL1 的 `draft_sample_method: probabilistic`（我方代码缺该机制）**（2026-09-25）

- **战略推演（带宽核算，R333 延伸）**：BL1 轮 27.2 ms / 30 GB（NVFP4）= 1100 GB/s；我方轮地板 43 GB（Q8_0 27 + Q4_K_M 16）/ 1100 = **39 ms** ⇒ **字节刀被 Q8_0 验收纪律封顶 ⇒ 纯靠效率到不了 8.0 ms/token**。但 **AL 2.76→4.0（1.45x）正好抵消字节差 1.44x** ⇒ **追平 = host 清零（P-D3' 等，-10~15）+ AL↑（-1.2x）** ✓✓ 路径闭合。
- **AL 落点（P8 具体化）**：BL1 的 `--speculative-config '{"method":"dflash",...,"draft_sample_method":"probabilistic"}'`（`vllm-1cat.service` 实读）；我方 `common/speculative.cpp` **无 sample_method 机制**（只有 `sample_from_anchor`/`block_size`/selector/walk）⇒ **probabilistic draft 采样 = AL 2.76→3.4+ 的候选刀** ✓（1cat 同构）。
- **刀序（终版）**：① P-D3' 完成（host -8~11）② **P8 probabilistic draft 采样**（AL↑）③ 专用 decode 注意力核（重估）④ P5 小投影。
- **P-D3' 剩余**：设备侧 5 层交错缓冲（batch 布局 [token][layer][dim] ⇒ 需 interleave 核或 backend 内 concat）+ speculative 指针直通 + 门值。

- **R347 ★★ P8 首攻（LLAMA_SPEC_PROB=1 块游走 softmax 采样）判负：AL 2.95 vs 基线 3.43、每轮 78.2 vs 74.0 ⇒ 不采用；BL1 的 `probabilistic` 是别的机制（draft 自身采样/温度/top-k 方案），非游走处裸采样**（2026-09-25）

- **实测（25K/gen96，2 臂 ×2 rep）**：A2（PROB=1）AL 3.56/2.34 = **2.95**、每轮 85.6/70.8 = **78.2**；B2（argmax 基线）AL 3.43/3.43 = **3.43**、每轮 80.5/67.6 = **74.0** ⇒ **采样让块内部一致性下降、接受率跌**。
- **机制判读**：DFlash 的块是**联合生成**（in-place denoising + lattice walk），逐位独立采样破坏块的相干性 ⇒ AL 反降。BL1 的 `draft_sample_method=probabilistic` 须另找其真身（1cat `vllm/spec_decode` 的 draft 采样路径）再对齐。
- **保留资产**：`LLAMA_SPEC_PROB=1` env（默认关 ✓ 生产零影响）+ `<random>` include ✓ 可复用作后续 draft 采样实验的基座。
- **AL 刀序更新**：P8 的朴素形态入灰；改查 1cat 的 draft 采样真身（temperature/top-k/块级采样）后再试；P-D3'（host）仍为第一刀。

- **R348 ★★★ P8 二攻方向定案（1cat 文档实读）：`probabilistic` 真身 = "exact probabilistic rejection sampling"（draft 概率 + min(1,p_t/p_d) 接受准则），非游走裸采样**（2026-09-25）

- **实读证据**：`RELEASE.md:30` "exact probabilistic rejection sampling"；`vllm/v1/spec_decode/llm_base_proposer.py:2476` "probabilistic rejection sampling"；`docs/design/sm70_deepseek_v4_dspark_acceptance.md:35` "old 1Cat proposer rejected probabilistic draft sampling and always returned the greedy token ... not comparable to the paper's probabilistic results"；`draft_probs` 全链（`sm70_dflash2_lookup.py:307` `probabilistic: bool`、`speculative.py:283` `rejection_sample_method: standard/synthetic`）。
- **机制**：我方 = **贪心 token 匹配**（draft token == target token 才收）；1cat = **拒绝采样**（收 min(1, p_target/p_draft)）⇒ **接受率系统性更高** ⇒ AL 2.76 → 3.4+ 的真落点 ✓✓。
- **P8 二攻实施面**：verify 段改拒绝采样（需 target 的 logits 概率 + draft 的 `draft_probs`）；贪心（temp=0）路径不受影响 ⇒ **门值（greedy sha）天然安全**；非贪心路径的分布保持由"精确拒绝采样"契约保证。
- **风险**：分布漂移须以 PPL/质量门复核（1cat 文档有 "probabilistic quality pair" 门）；TP 同步（`llm_base_proposer.py:192` "same token on all TP ranks"）。
- **刀序（终版）**：① P-D3'（host -8~11）② **P8 二攻 = 拒绝采样**（AL↑）③ 专用 decode 核。

- **R349 ★★★ 起跑线对齐（用户指出）：draft 从 Q4_K_M 换 BF16 同级 F16 GGUF（HF `incoai/Qwen3.8-27B-DFlash2` @ `dedf8df…`，魔搭 aria2c 16 线程 98 MiB/s 下载）⇒ AL 2.96→3.74（+26%）、合成 tg 44.4→52.5（+18%）⇒ 采用；字节刀"封顶"推论作废**（2026-09-25）

- **起跑线纠正（用户 2026-09-25 指出，认账）**：我方 draft 原为 `…-DFlash2-Q4_K_M.gguf`（Q4_K_M，2 GB）vs BL1 的 `Qwen3.8-27B-DFlash2`（**BF16，3.6 GB，1.92B 参数/5 层**）✗ **不等价**；`DFlash2` 是 5 层小 draft（非 27B），我此前"16 GB draft"的字节账**有误**。
- **对齐动作**：HF 官方仓无 GGUF ⇒ 下载官方 BF16（revision `dedf8df…` = BL1 同 revision；魔搭镜像 + aria2c -x16，98 MiB/s）→ `convert_hf_to_gguf.py --outtype f16`（**容器封装非量化**）→ `Qwen3.8-27B-DFlash2-F16.gguf`（4 GB）；`t1c-run.sh` 默认切换 + 新增 `DRAFT=` 旋钮。
- **同源 A/B（25K/gen96，2 臂 ×2 rep）**：F16 AL 4.57/2.91 = **3.74**、每轮 81.3/61.1 = **71.2**、合成 tg **52.5**；Q4_K_M AL 3.10/2.82 = 2.96、每轮 65.7/67.6 = 66.6、tg 44.4 ⇒ **AL +26% 抵过每轮 +7% ⇒ 合成 +18% ⇒ 采用 F16** ✓。
- **字节账重算**：BL1 = FP8 27 + BF16 3.6 = 30.6 GB/轮；我方 = Q8_0 27 + F16 4 = 31 GB/轮 ⇒ **同级** ✓ ⇒ **R346"字节刀被封顶 ⇒ 纯效率到不了 8.0"**推论**作废**（不删史）；剩余差距**全部是代码问题**（用户红线："1cat 能实现 我们就有理由能实现，达不到就是代码问题"）。
- **显存包络**：27+4 = 31 GB + KV/workspace ✓ 仍在 4×V16GB 内 ✓。

- **R350 ★★ P-D3' 设备路径实现到门值全绿（g1=g0=bcda0092 ✓）但 stress 崩：R344 的"`ggml_backend_tensor_set` 自动推断 D2D"**洞察被证伪**（它是 H2D 语义，传设备指针 = 段错误）⇒ 修法改为"set_inputs 的 embd 走 D2D 分支"**（2026-09-25）

- **已落（build 绿 + 门值绿 ✓）**：`ggml-cuda-d2d.cu`（`ggml_cuda_copy2d` = 2D D2D 交错、`ggml_cuda_alloc_bytes`/`free_bytes`）+ `speculative.cpp` 设备交错路径（`LLAMA_SPEC_DEVFEAT=1`：5×2D D2D、去 memcpy、去 `llama_synchronize`）+ 设备指针直通 `batch_inject.embd`。
- **崩因（定位）**：`llm_graph_result::set_inputs` 对 embd 做 `ggml_backend_tensor_set` = **H2D** 拷贝 ⇒ 设备指针被当 host 地址 ⇒ 段错误（stress 于 prefill 后首轮注入崩；门值因 spec-off 未触发 ✗ 门值绿 ≠ 全形状绿，教训）。
- **修法（下一轮）**：`set_inputs` 的 embd 分支加 **D2D 分支**（源指针属设备时 `cudaMemcpyAsync(DeviceToDevice)`）或 `llama_batch` 加 `embd_dev` 标志位；**或**回退 host 路径并把 sync 移到 copy 后（净收益 -0，仅诊断价值）。
- **⚠️ 判据教训**：门值是 spec-off 口径 ⇒ **spec 形状的正确性必须用 `SPEC=1 bash t1c-gate.sh`**（本轮已用但 g1 的注入路径未被触发 ✗ 需 stress 级验证）；"build 绿 + 门绿"≠"全形状绿"。

- **R351 ★★★ 用户授权明确（2026-09-25）："直接把有用的 copy 过来，不需要完全重新造轮子" ⇒ 1cat 代码可直接拷贝（开源 BSD-3-Clause）；已 vendored `scalar-attention.cu`（1249 行 decode 长注意力核）+ `fp8_kv_utils.cuh`/`fused_mma.h`/`flash_v100_traits.cuh`/`paged_kv_utils.cuh` → `ggml/src/ggml-cuda/sm70-long/`（附 LICENSE 溯源）**（2026-09-25）

- **拷贝方针（用户定调，长期有效）**：1cat-vLLM（BSD-3-Clause）的有用实现**直接 copy**，不做"干净室重写"；只做接口适配（剥 ATen 薄壳接 ggml）。已拷清单见 `ggml/src/ggml-cuda/sm70-long/`。
- **P7 正确形态就位**：`scalar-attention.cu` = decode 形状（q 1..8 × 长 KV）的专用核 ✓ 正是 R339 判负时认定的"专用小-M GEMV 核" ✓ ⇒ 下一步 = 接入 ggml 的 FA 分发（decode 形状走它），剥 `torch/library.h`/ATen 依赖。
- **并行拷贝候选**（按 ROI）：`csrc/attention/sm70_grouped_long/kernel/grouped-attention.cu`（228 KB prefill 核）、`csrc/moe/marlin_moe_wna16/sm70_marlin_*`（MoE GEMM）、`csrc/libtorch_stable/sampler.cu`（GPU 采样）、`fused_sigmoid_gating.py` 对应的 GDN 融合核。
- **既有 P-D3' 修复在途**：`cudaMemcpyDefault` 一行修（H2D→自动推断）build 绿 ✓ 待 stress 验证。

- **R353 ★★★ 认账改判（用户指正）：llama.cpp 的 TP 支持 6 卡（PR #24554 "TP: allows the usage of 4-10 gpus"，issue #24486 的非整除修复，已合并）⇒ 我此前"4 KV 头无法 6 分"是 vLLM 记述误用 ✗ 作废**（2026-09-25）

- **R354 ★★ P-D3' 三攻：batch gather 的 memcpy 已跳过（devfeat）但新崩因 = 设备不匹配（`ggml_backend_cuda_host_buffer_free_buffer` @ device 3）⇒ `ggml_cuda_alloc_bytes` 的 cudaMalloc 未固定设备（多卡 alloc/free 落不同 device）；修法 = 帮手函数带 device 参数（cudaSetDevice）**（2026-09-25）

- **进度**：① `set_inputs` 的 `cudaMemcpyDefault`（R350 ✓）② batch gather memcpy 跳过 + `ubatch.embd` 直通（R354 ✓ build 绿）③ **崩因转到设备管理**（多卡下 cudaMalloc/cudaFree 的 device 一致性）。
- **修法（下一轮）**：`ggml_cuda_alloc_bytes(bytes, device)`/`free_bytes(p, device)` 加 `cudaSetDevice`；或用 backend buffer API（`ggml_backend_buft_alloc_buffer`）避开裸 cudaMalloc；或回退 host 路径（净收益 0）。
- **⚠️ 多卡教训**：任何裸 CUDA API（cudaMalloc/cudaFree/cudaMemcpy2DAsync）在 TP 多卡下**必须显式 `cudaSetDevice`**，否则 alloc/free/copy 落到不同 device ⇒ 崩（本次 device 3 vs 其他）。

- **R355 ★★★ P-D3' 判定：架构性阻塞（TP4 下 extract 张量分散在多卡）⇒ 设备特征路径 env 默认关、入灰；转投用户授权的 `scalar-attention.cu` 集成**（2026-09-25）

- **阻塞根因（架构级，非 bug）**：`extract_layer_inputs` 的 `t = res->get_layer_inp(il)` 逐层可能落在**不同 device**（`ggml_backend_sched_get_tensor_backend(sched, t)` 逐层不同）；而 batch 的 embd 在 llama.cpp 里是 **host 缓冲、各 device 各自取自己的分片** ⇒ 单个设备缓冲不足以承载交错后的全行数据，正确实现需要**每卡 embd 分片**（复刻 split-input 逻辑）⇒ 工程量远大于 ~10 ms/轮收益。
- **处置**：设备路径（`LLAMA_SPEC_DEVFEAT`）**默认关** ✓（生产零影响）；相关代码保留为实验资产（`ggml-cuda-d2d.cu` + llama-batch/llama-context 的 env 分支），本轮已把 build 修绿（`llama_get_device` 不存在 ⇒ 改传 0 并注释架构限制）。
- **教训**：TP 下"每卡本地"假设必须逐层核实 backend/device，不能假设单一设备。
- **转向**：接用户授权的 `scalar-attention.cu`（1cat decode 长注意力核，P7 正确形态）到 ggml FA 分发 —— 预期收益大于 P-D3'，且是 1cat 已验证的形态。

- **R356 ★★★ 1cat 注意力核接入方案定案 + 256K 新基线（F16 draft）：E4M3 无对应 ⇒ 用核的原生 FP16 路径（我方 f16 K/V 已现成）；256K spec-on 口径 tg 35.18、TTFT 175.2 s、85.7 ms/轮、AL 3.05**（2026-09-25）

- **E4M3 问题（用户问，已查证）**：我方 llama.cpp **无 E4M3 类型**（`ggml.h` 仅 NVFP4 提及 E4M3 scale；KV 支持 f16/q8_0 等）⇒ 按用户指示"找等价替换"：**1cat 核原生支持 `KV_CACHE_DTYPE_FP16`（模板参数 0）**，而我方 `fattn-sm70-d256` launcher **本来就产出 f16 K/V**（native f16 或 dequant 镜像 `f16_extra.K/V`，见 `fattn-sm70-d256.cu:801-803`）⇒ **等价替换 = 直接喂 f16，零解码改写**（比把 q8_0 移植进模板便宜得多）✓。
- **接入面**：核入口 `private_grouped_e4m3_fp32_paged(q,k,v,out,block_table,row_lengths,partial,lse,scale,k_scale,v_scale)` 需 **(a)** 剥 ATen 壳（裸指针）**(b)** f16 KV 的**分页表**（我方镜像是连续 scratch，需 identity 页表或改用连续路径）**(c)** 接 ggml FA 的 decode 分发。
- **256K 新基线（F16 draft 对齐后，spec-on，1 rep 有效）**：TTFT **175.2 s** / pp 1348.7 t/s；tg **35.18 t/s**；**85.7 ms/轮**（3599/42）；**AL 3.05**；rep2 = `ERROR no tokens`（服务级偶发，待复测确认）。
- **⚠️ 新发现（口径纠正）**：**预填充必须按 spec-on 口径比**（BL1 = FP8+DFlash2 spec-on）⇒ 我方 spec-on TTFT **175.2 s vs BL1 152.5 s = 1.15x 劣** ✗（此前 R326 的 151.9 s 是 **spec-off** 口径，不能直接对 BL1）⇒ 预填充线需补 spec-on 口径的收口工作（draft 注入的 prefill 附加成本）。
- **对比基线更新**：采纳 R349 的 F16 draft 后，256K 采用链 = tg 35.18 / 85.7 ms/轮 / AL 3.05（spec-on）。

- **R357 ★★★ 用户关切核实：T1-C 的两处 decode 实验改动已限制为 decode-only，预填充恢复原始路径；复测 150.867/152.535 s（均值 151.70）vs 采用值 151.93 ⇒ 未变，仍 ≤ BL1 152.5 s**（2026-09-25）

- **用户关切**（"预填充都调好了别动那边"）：自查 `git diff 37286c5fb..HEAD` 确认 T1-C 收口后有 **5 个 commit** 触及 `fattn-79t/prefill.cu`，其中**两处对预填充有真实行为差异**：① `qk_algorithm`（常量 ALGO9 → `(rows%8&&width%8)?ALGO9:ALGO0`，尾块会走 ALGO0）② `width_pad` 作为 GEMM 的 n（尾块多算 ≤7 列）。
- **处置（R357）**：新增全局 `t1c_decode_exec`（仅 `LLAMA_SM70_79T_DECODE=1` 时为真），两处均改为 **decode-only** ⇒ 预填充回到一字不差的旧路径 ✓；行填充（rows 12288 本就是 8 倍数）与 QKFAIL 诊断打印保持（无行为影响）✓。
- **复测证据（spec-off，k1 臂 ×2 rep，256K）**：TTFT **150.867 / 152.535 s**（均值 **151.70**，pp 1566/1549），`total=52.300s calls=14592`；对照采用值 151.93 s（R326）⇒ **未变** ✓ 仍 **≤ BL1 152.5 s** ✓；门值 g0=g1=bcda0092 ✓。
- **启动参数未动**（已核对 `t1c-run.sh`）：`--ctx-size 262144 --ubatch-size 2048 --tensor-split 1,1,1,1 --cache-type-k/v q8_0 --flash-attn on` ✓；唯一变更 = draft 换 F16（用户指示）✓。
- **vendored 隔离**：`sm70-long/` 为**子目录**，CMake `file(GLOB GGML_SOURCES_CUDA "*.cu")` 只收根目录 ⇒ **不进构建、零影响** ✓。

- **R358 ★★★ 1cat `grouped-attention.cu` 剥壳成功并编译通过（影子头方案，零 torch）**：写 `sm70-long-atenshim.h`（假 `at::Tensor` 带 size/stride/data_ptr/clone + `IntArrayRef`/`ScalarType`/`CUDAGuard`/`getCurrentCUDAStream`/`getCurrentDeviceProperties` + `TORCH_CHECK`/`TORCH_WARN`/`C10_CUDA_CHECK`/`C10_CUDA_KERNEL_LAUNCH_CHECK` 宏），文件头用 `#if defined(SM70_LONG_RAW)` 条件化 ATen 头、文件尾条件化 TORCH_LIBRARY 注册 ⇒ **`nvcc -std=c++17 -arch=sm_70 -DSM70_LONG_RAW` 单文件编译通过（1.1 MB .o，零 error）** ✓（2026-09-25）

- **编译迭代记录（缺口清单）**：① `IntArrayRef` 缺 `std::vector` 构造 ② 缺 `TORCH_WARN` ③ 需 `-std=c++17`（`std::is_same_v`）④ 缺 `C10_CUDA_CHECK`/`C10_CUDA_KERNEL_LAUNCH_CHECK` ⑤ 未用入口里的 `Tensor::clone()` 用别名顶替。
- **下一步（集成）**：写我方裸指针入口调用 `launch_flash_attention_decode_paged(...)`（模板于 D/PARTITION_SIZE/KV_DTYPE/SEQ_LEN_ROUTE）——喂 **f16 KV 镜像 + 恒等页表**（页=256 token，stride(0)=65536/stride(1)=256/stride(2)=kv_len ⇒ 零数据搬运），再接 ggml FA 的 decode 分发；随后门值 + 每轮 A/B。
- **资产**：`sm70-long/`（`grouped-attention.cu` 5011 行 + `scalar-attention.cu` + 依赖 + 影子头 + LICENSE）✓ 均已入库。

- **R359 ★★★ 集成入口锁定：`launch_flash_attention_decode_paged_xqa_tc_256_staged`（张量核 XQA、GQA 6:1、D=256、fp16 KV）与我方每卡形态完美同构（1 KV 头 : 6 Q 头、d=256）；KV 呈现方案确认**（2026-09-25）

- **入口签名**：`launch_flash_attention_decode_paged_xqa_tc_256_staged(q, k_cache, v_cache, out, block_table, seq_lens, tmp_out, max_logits, exp_sums, online_rescales, active_num_partitions, softmax_scale, launch_num_partitions, use_split_reduce, split_reduce_dim_tile, stream)`；内部 `kGroupSize=6`、`D=256`、走 `flash_attention_decode_xqa_tc_{qk,pv}_..._256_wide` 核 + `launch_flash_attention_decode_xqa_split_reduce<256>`；grid = (batch, num_heads_kv, partitions) ⇒ **按 KV 头调用** ✓。
- **KV 呈现（零搬运）**：核按 `k_index = phys_block*k_block_stride + block_offset*k_token_stride + kv_head*k_head_stride` 寻址且 head 维连续 ⇒ 我方 f16 镜像（`[256, kv_len]`、dims 连续、token 步长 256）可表示为**页=256 token 的分页缓存**：`stride(0)=65536`、`stride(1)=256`、`stride(2)=kv_len` + **恒等页表** `[1, ceil(kv/256)]` ✓。
- **下一步（写裸入口）**：`extern "C" sm70_long_decode_f16(...)` 构造影子 Tensor → 调该启动器；scratch（tmp_out [parts, n_q, 6, 256] f16 / max_logits+exp_sums [parts, n_q, 6] f32 / online_rescales）用 `ggml` 侧缓冲；然后接 ggml FA 的 decode 分发 + 门值 + A/B。

- **R360 ★★★ 裸入口落地并编译通过：`extern "C" sm70_long_decode_f16(...)`（影子 Tensor → xqa_tc_256_staged 启动器），`.o` 导出该符号** ✓（2026-09-25）

- **scratch 规格（权威来源：`flash-attn-v100/flash_attn_v100/flash_attn_interface.py:1315` + `tests/kernels/attention/test_sm70_grouped_e4m3_fp32.py`）**：`partial_out (parts, n_q, 6, 256)` **fp16**、`partial_lse (parts, n_q, 6)` f32、`online_rescales [n_q, 6, parts]` f32（`flash_decode_paged.cu:6128-6130` 的检查）、`active_num_partitions [1]` int32、`seq_lens [1]` int32、`block_table [1, n_pages]` int32。
- **入口 ABI（已实现）**：`sm70_long_decode_f16(q, k_cache, v_cache, out, block_table, seq_lens, partial, max_logits, exp_sums, online_rescales, active_num_partitions, n_q, n_kv_heads, page_size, n_pages, n_parts, n_q_heads_per_kv, softmax_scale, stream)`；内部按页=page_size、恒等页表、`stride(0)=page*256 / stride(1)=256 / stride(2)=page*n_pages` 组织影子 Tensor；q<2 时上取 2（staged 核要求 q≥2）；`use_split_reduce=true, split_reduce_dim_tile=256`。
- **编译证据**：`nvcc -std=c++17 -arch=sm_70 -DSM70_LONG_RAW` ⇒ 零 error、1.14 MB `.o`、`nm` 可见 `sm70_long_decode_f16` ✓。
- **下一步**：① 把该文件加入构建（仿 79T 的隔离 OBJECT 目标 + `SM70_LONG_RAW` + `-std=c++17` + sm_70）② 在 FA 分发里对 decode 形状取 f16 镜像指针 + 恒等页表 + scratch 并调用 ③ 门值 + 每轮 A/B。

- **R361 ★★★ 1cat decode 核已接入我方构建并进入产物库**：新增可选隔离 OBJECT 目标 `ggml-cuda-sm70long`（CMake `-DGGML_CUDA_SM70_LONG=ON` 开启，`SM70_LONG_RAW` + `--std=c++17 -O3` + `ARCH 70`），payload 位补齐 5 个 vendored 文件；**验证：`nm libggml-cuda.so.0.24.0` 可见 `sm70_long_decode_f16`（1 处）、`grouped-attention.cu.o` 1.14 MB 在构建树** ✓；门值 g0=g1=`bcda0092` ✓ 无回归（2026-09-25）

- **构建事实**：`BUILD_RC=0 / ERROR_LINES=0`；重配置 `cmake -DGGML_CUDA_SM70_LONG=ON .` rc=0 ✓。
- **隔离性**：该目标只在显式开启时编译（默认关 ⇒ 常规构建不受 5000 行重模板影响）✓；文件仍位于子目录、与既有 FA 路径零交互 ✓。
- **下一步**：FA 分发接线（decode 形状 → 取 f16 K/V 镜像指针 + 构造恒等页表与 scratch → 调 `sm70_long_decode_f16`）→ 门值 → 每轮 A/B（这是本线目前最大的一把刀）。

- **R362 ★★★ FA 分发已接线并编译通过（env 门控默认关，门值无回归）：新增 `BEST_FATTN_KERNEL_SM70_LONG` + `fattn-sm70-long.cu` 胶水（Q/K/V→f16、恒等页表、scratch、按 KV 头调用）**（2026-09-25）

- **接线内容**：`fattn.cu` 新增枚举 `BEST_FATTN_KERNEL_SM70_LONG=501` + 声明 + 分发 case + alloc/name 分支；`fattn-sm70-long.cu`（新文件，根目录 ⇒ 被 CMake glob 收）实现 `ggml_cuda_sm70_long_decode_supported/alloc_size/decode`，条件：Volta、d=256、q∈[1,8]、kv ≥ 512、GQA 恰为 6、K/V 同类型、Q 连续；env `LLAMA_SM70_LONG_DECODE=1` 开启（默认关）。
- **验证**：`BUILD_RC=0 / ERROR_LINES=0` ✓；默认门值 **g0=g1=bcda0092** ✓（既有 FA 路径零影响）。
- **⚠️ 布局阻塞（下一步）**：我方 KV 为 **`[256d][kv][hkv]` 交织布局**（`:603` 实测 `K=(256,262144,4,1) Knb1=1088 Knb2=272 rowK=272` ⇒ head 维在里层、token 步长 1088），而 1cat 核按 **token-major `[t][256]`** 连续读 head 维 ⇒ **必须先做"转置式反量化"**（T1-C 的 `t1c_k_q8_0_to_half_t` 正是该形态，可复用/移植到胶水）；`convert.cuh` 的 `to_fp16` 只有**扁平版**（无多维重载，本轮据此修掉 7 个编译错）⇒ **在转置反量化落地前不得开启该 env**（否则镜像错误 ⇒ 数值必错）。
- **判据纪律**：开启后第一判据 = `SPEC=1 LLAMA_SM70_LONG_DECODE=1 bash t1c-gate.sh` 的 greedy sha；通过后才跑每轮 A/B。

- **R364 ★★★ 接线完成编译但**未真正生效**（诚实修正）：`LLAMA_SM70_LONG_DECODE=1` 下 `[FAK]` 显示仍选 MMA_F16/TILE；诊断定位 = **decode 形状的 FA 调用收到 `kv=256`**（不是长上下文）⇒ 被我的 kv≥512 门槛拦下；此前"门值全绿"是**走老路**的结果，不算数**（2026-09-25）

- **证据链**：① `GGML_CUDA_FA_KERNEL_DEBUG=1` ⇒ `[FAK] kernel=TILE D=256 n_q=2 n_kv=256 kv_type=8` 与 `kernel=MMA_F16 D=256 n_q=42 n_kv=256`（后者是预填充分块，n_q>8 正确拒绝）② 新增形状诊断（最多 8 条）⇒ **8/8 全是 `q=2 kv=256 hq=24 hkv=4 Ktype=8 Qtype=0`、原因均为 "kv shorter than 512"**（prompt 为 5000 token 却只见 256 ✗）。
- **待查（下一轮主线）**：为什么 decode 的 FA 只看到 `kv=256`？候选：① 该模型全注意力层是 **滑动窗口(SWA)/分块**（256 = 窗口）② KV cache 的 cell 视图语义 ③ 该 FA 调用属于别的路径。**在弄清前不得把该核接到 decode**（否则语义错）。
- **诚实修正**：R363 的"q/kv 形状实测 24 头/4 头"仍成立 ✓，但"长上下文"前提在 decode 下**不成立** ⇒ 该核的用武之地（长 KV 解码）需要先确认 llama.cpp 侧是否真的以长 KV 调用解码注意力。
- **代码状态**：胶水（staging/scatter/页表/scratch）已实现且编译通过 ✓、env 默认关 ✓、默认门值 g0=g1=bcda0092 ✓（生产零影响）。

- **查证（官方 + 互联网）**：PR #24554（stepfun/laguna 的 4-10 卡支持 ✓ 已合并）；PR #19378（backend-agnostic TP 基建）；PR #23912（TP 下 KV 量化）。⇒ llama.cpp `--split-mode tensor` **支持非整除卡数**（KV 头分布处理），6 卡 Qwen3.8 可行 ✓。
- **误判根因**：把 vLLM 时代的 TP 整除约束（E 系列"4 个 KV 头无法 6 分"）误套到 llama.cpp ✗；E 系列该条目**限定为 vLLM 语境**（TP3/TP6 sweep），不再外推到 llama.cpp。
- **红线不变**：验收包络 ≤ 4×V100-16GB（>4 卡 = 不合格、更少卡 = 优）⇒ 6 卡机制可行但不在合格区；目标 = **3-4 卡达标**。
- **kernel 拷贝与 TP 的关系（复核）**：vendored 核为每卡本地 ✓ 不影响任何卡数的 TP ✓（heads 参数随配置走）。

- **R352 ★★ P-D3' stress 仍崩 ⇒ 第二处 H2D 假设定位 = `llama_batch_allocr::init` 的 embd 拷贝（不止 `set_inputs`）；TP 咨询答复（用户问）= 拷贝 kernel 为每卡本地、不影响张量并行**（2026-09-25）

- **崩溃链（两处 H2D 假设）**：① `llm_graph_result::set_inputs` 的 `ggml_backend_tensor_set`（已改 `cudaMemcpyDefault` ✓ R350）② **`llama_batch_allocr::init`（`llama-batch.cpp:25`）对 `batch_inp.embd` 的 host 假设拷贝** ⇒ 首轮注入即崩（stress 于 `n_tokens=42` 后死，门值不触发 ✗）。
- **修法**：batch 分配器加 `embd_dev` 通道（或 embd 指针的设备判定 + D2D）；或回退 host 路径（净收益 0，仅诊断）。
- **TP 咨询答复（用户 2026-09-25 问）**：llama.cpp `--split-mode tensor` 与 vLLM TP 同类，整除约束同源（4 KV 头 / 4 卡 = 每卡 1 头，与 BL1 同构）；**拷贝的 kernel 是每卡本地**（处理本卡分片，不跨卡通信、不参与切分）⇒ **不影响张量并行** ✓；唯一检查项 = 核内形状参数（heads_q/heads_kv）与我方一致（TP4 + 24 Q/4 KV ✓ 同构）。

- **R344 ★★★ P-D3' 实施面定案：设备侧特征路径三触点 + "D2D 免改图机器"洞察（`batch_inject.embd` 指设备内存 ⇒ `ggml_backend_tensor_set` 自动 D2D ⇒ host 同步 10.4 ms/轮 整条消失）**（2026-09-25）

- **三触点（代码实读）**：① `llama-context.cpp:2312` `extract_layer_inputs` 用 `ggml_backend_tensor_get_async` 落 **host** `embd_layer_inp` ⇒ 改设备缓冲 ② `common/speculative.cpp` 注入段 host memcpy 填 `batch_inject.embd` ⇒ 指针直通 ③ `set_inputs`（`llama-graph.cpp:1370`）对 embd 做 `ggml_backend_tensor_set` ⇒ **源指针为设备内存时 backend 按 kind 推断走 D2D**（CUDA memcpy 异构）⇒ **图机器免改** ✓。
- **收益机制**：host `llama_synchronize(ctx_tgt)`（10.4 ms/轮，R331 定位 = 等 target GPU 尾巴）整条消失；且 D2D 天然带流序（免 event）⇒ 注入可与 target 的 GPU 尾巴**重叠**（串行链"verify→sync→注入→draft block"破链）。
- **实施序**：① extract 的 `embd_layer_inp` 改设备缓冲（含 `output_reorder` 的设备版或 reorder 后再 D2D）② speculative 注入段改指针直通 ③ 门值（greedy sha）+ `LLAMA_SPEC_TIMING` 分账（sync 应 → ~0）+ 每轮 A/B（预期 **-8~11 ms/轮**）。
- **风险**：`output_reorder`（`:2343`）在 host 上交换行 ⇒ 设备版需另写或 reorder 后一次 D2D；M-RoPE/多 seq 语义保持。门值 = 第一判据。

- **⚠️ 口径与未决（NODROP）**：本条 = **数值/时序双口径**，FLOPs/形状与基线同构故 TTFT 可比；但**数值尚未过门**：① tail（对角块）本轮才实现、未做 PPL/greedy 校验 ② 已观测中后段**输入 Q 变非有限（99.9%）** ⇒ 模型发散 = 引擎数值错误的下游后果（`out` 非有限扫描 = 0，即我方从不写非有限值，是"值错"不是"越界写坏"）。**采纳条件 = 数值门（融合核对拍 / PPL）+ k0/k1 各 ≥2 rep 离散**。**下一刀 = 单调用参考值对拍**（融合核 dump 同 dst ↔ 引擎 dump，几何/KV 反量化/行 max/PV/merge 逐段定位）。
### R366 ★★★ 吐字线诊断大收口：1cat decode 核判死（A/B 相同）、FULLGRAPH 判负（host -10 ms/轮 但端到端更慢 ⇒ GPU 执行受限）、256K 新最佳 tg 53.7（NMAX=7、AL 5.73 > BL1 3.4）；差距收敛到轮成本 106 vs 27.2 = 3.9x（2026-09-25）

- **章程固化（用户 2026-09-25）**：吐字线与预填充同章程，**BL1 = 及格线**（tg ≥124.6 / tpot ≤8.0 / ms/轮 ≤27.2）；**预填充已落袋代码冻结**（公共组件例外但默认行为不变 + 门值重过）；显存 ≤ 1cat 包络（4×V100-16GB），超出 = 项目无意义；1cat 能实现 ⇒ 硬件没问题 ⇒ 必是代码问题。详见 AGENTS.md §1.0.1。
- **1cat decode 核（R351-R364 集成线）判死**：运行期执行探针 `[SM70EXEC]` = **0 次**（从未执行）；同源 A/B（5K/NO_SPEC/UB2048）开 **27.32** vs 关 **27.06 ms/token** ⇒ **完全相同**。根因：`supported()` 只在建图 reserve 被问过（q=1/kv=262144 通过并打印 SELECTED），运行期 decode 的 FA 走 VEC(q=1)/TILE 路径未进该分支（R364 的"kv=256 之谜"已解 = 诊断打印 cap 被预填充/预热调用耗尽的假象）；且即使选中，我方集成每调用转全 KV 到 f16（O(kv) 流量）vs 1cat 分页 fp8 直读。**5K 下解码注意力总共 ~0.2-1 ms/步** ⇒ 无收益空间。
- **FULLGRAPH（整步单捕获含集合通信）判负但机制学到**：实测 `sub/call 129→7.4`、`ar/call 128→7.3`、meta host 26.1→**15.6 ms/call（-10 ms/轮）** ⇒ 结构改进真实生效；但端到端 **24.72 vs 基线 22.38 ms/token（更慢）** ⇒ **解码步 = GPU 执行受限**（非 host 提交墙，R342 判读加强）。
- **GPU 利用率直接证据**：`nvidia-smi -lms 100` 生成期四卡 **85-87% 稳态**（非空闲窗口均值 84%）⇒ 每步 ~22 ms 中仅 ~3.5 ms 气泡。
- **权重流已到下限（独立复证 R203/R333）**：`test-backend-ops perf` 真实形状 `MUL_MAT(q8_0,m=4096,k=14336,n=1)` = **83.09 µs = 751 GB/s**（峰值 ~900）⇒ 27 GB/4 卡 = **9 ms/步地板**；每步 22 ms ⇒ **~13 ms 非权重开销**（每层 ~30 kernel × 3-6 µs；LM head 1.54 ms/步单卡独占，248320×5120 q8_0 = 1.27 GB/步）。
- **对照实验（稠密无 GDN）**：DeepSeek-R1-Distill-Qwen-32B-Q4_K_M（64 层稠密）8K ctx **18.14 ms/token** vs 自身地板 ~6.6 ms = **2.6x**，与 27B（2.5x）同比例 ⇒ **不是 GDN 特例，是 llama.cpp 解码系统性开销**。
- **轮分解（25K，SPEC=1 NMAX=7）**：target decode+sync **31.4** + draft_decode **13.1**（8 tok = 1.64 ms/tok，地板 1.28 ⇒ draft 不浪费）+ selector 2.3 + **injection 12.7**（sync 9.6 + submit 2.5，纯串行）+ sampling 1.6 ≈ **60-69 ms/轮（稳定）**；AL 随内容波动 3.4-5.7 ⇒ tg 波动是 AL 驱动，**轮成本才是物理量**。
- **n_max 扫描（25K 同源）**：2/3/5/7/11/15 ⇒ 21.1/20.6/19.1/**13.1-19.2**/21.3/23.1 ms/token ⇒ **NMAX=7 最优**（11/15 更差：草稿成本线性而超出段接受率低）⇒ 保持 7。
- **256K 新基线（NMAX=7 spec-on）**：tg **53.72 t/s**、**18.53 ms/token**、**AL 5.73（> BL1 3.4）**、轮 ≈106 ms；旧值（R356：tg 35.18、85.7 ms/轮、AL 3.05）⇒ tg +53%（AL 差异 + 代码演进）。**AL 我方更优 ⇒ 草稿质量/策略不是问题，唯一刀 = 轮成本**（106 vs BL1 27.2 = 3.9x）。
- **刀序（数据支撑）**：① **injection 12.7 ms/轮**（sync 9.6 纯串行；P-D3' 设备特征路径蓝图见 R344/R355）② 每步 ~13 ms 非权重开销（融合工程：每层 30→15 kernel）③ lm_head 拆 4 卡 ④ 256K 解码注意力已在带宽下限（+13 ms ≈ 16 层 × 570 MB ÷ 750 GB/s）⇒ 勿投 ⑤ n_max 保持 7。
- **本轮判死/判负清单**：1cat XQA decode 核（A/B 相同）；FULLGRAPH（端到端更慢）；TP2（KV 复制 4.35 GB/卡 ⇒ OOM，KV 非切分而是每卡全量）；draft 独立跑（`dflash requires ctx_other`）；`GGML_CUDA_OP_TIMING`（多卡 event 查询乱值/负时长）；nsys（ppc64le importer "Wrong event order" bug：node 级 trace 必崩、graph 级只出 runtime/memcpy 无 kernel 数据）。
- **工具落袋**：`rjob.sh`/`rwait.sh`（10 秒 sleep 轮询 + 终止符，AGENTS.md §1.4.1 强制项）；`t1c-run.sh` 新增 `MODEL/CTX/TS/PROF/SUB/FULLGRAPH/METAHOST/NSYS_*` 旋钮；R203 方法复用（`test-export-graph-ops` + `test-backend-ops perf --test-file` = 真实形状逐算子账：tg 图 41 唯一 op / 3751 节点，1362 视图 op）。
- **⚠️ NODROP（预填充观察，已判 = 零回归，口径差非回归）**：本轮 256K 跑出 TTFT 175.78 s（冻结值 151.70，+15%）⇒ 复测定性：**spec-on 两 rep 174.469/175.170**（与 R356 spec-on 175.2 吻合 ±0.5%）、**spec-off 两 rep 152.499/152.534**（vs 冻结 k1 150.867/152.535，均值 +0.5% 噪声内、rep2 逐位同）⇒ 我的 fattn.cu / fattn-sm70-long.cu 改动**零回归** ✓（门值逻辑不变）。**但体感 TTFT 口径（spec-on）预填充尚未达标**：175.2 vs BL1 152.5 = **1.15x 劣**（R356 已标注），差距 = 草稿链 prefill 附加成本 ~23 s ⇒ 归入吐字线公共组件刀序（与 injection 同源机制）。
- **路径核实（用户 2026-09-25 提示）**：9/24 夜确有「正式二进制 `/root/llm/llama.cpp/bin/llama-server` + 测试库 `/root/libdir-gb`」混装跑（dq 系列崩溃栈 3 处证实），但**冻结值 k1（9/25 08:24 `bin=/root/libdir-gb`）与本轮全部复测均用测试二进制** ⇒ 路径不是 TTFT 差异来源；正式环境 `/root/llm/llama.cpp`（mtime 9/10）、`/root/llm/systemd`（9/16）未被动过 ✓ 只读成立。

### R367 ★★★ 刀序重排（分相计数定案）：inject 刀判死（解码期 sync 仅 0.02 ms/call，"12.7 ms/轮"系预填充污染）；真实头号刀 = **target verify 59.2 ms/轮（64%）**；256K tg 新高 63.6（AL 5.89）（2026-09-25）

- **分相计数（`speculative.cpp` inject split，BUILD_RC=0）**：`pre n=3693 sync=17.429 ms/call | dec n=51 sync=0.027 ms/call`（256K）；25K：pre 11.099 / dec **0.021** ms/call ⇒ **解码期注入的 sync 是免费的**（0.02 ms），R330/R331 的"注入 sync 10.4 ms/轮"与 R366 的"12.7 ms/轮"**全部是预填充期调用的累计平均污染**（预填充每 64-token 草稿 ubatch 注入一次，388-3693 次 vs 解码每轮 1 次）。**injection 解码侧刀判死（只改灰不删）**。
- **反向坐实（公共组件刀）**：预填充期注入 = 11.1（25K）/17.4（256K）ms/call × 388-3693 calls = **spec-on TTFT 附加成本 ~23 s 的真身**（R356 口径差）⇒ 该刀仍在，但落点是**预填充期注入的分块次数**（草稿 ubatch 被 cap 到 64 ⇒ 2048-token 块 = 32 次注入调用/块），刀法 = 调大草稿注入批/合并调用，属公共组件（默认行为不变 + 门值重过）。
- **真实轮账（GEN=320，NMAX=7，LLAMA_SPEC_TIMING）**：256K **target decode+sync 59.169 ms/轮（64%）** + draft_decode 13.65 + selector 2.33 + injection ~3 + walk 0.08 ≈ 78；实测轮 92.6 ms（tg 63.60、AL 5.89）⇒ **~14 ms 未归因**（sampling 1.6 + accept/walk + 调度间隙）。25K：target **36.794** + draft 12.96 + selector 2.34，实测轮 63.4。
- **tg 新高**：256K **tg 63.60 t/s、15.69 ms/token、AL 5.89**（此前 53.72/18.53/5.73）；25K tg 45.55（AL 2.89，此 rep 接受率偏低）。距 BL1 124.6 = 1.96x（轮 92.6 vs 27.2 = 3.4x）。
- **target verify 59.2 ms 的成分账（256K）**：权重流 9（下限）+ KV 注意力 ~11.9 + M=1→8 GEMM 增量 ~5（n=1 83 µs → n=4 89.9 µs 外推 465 GEMM/步）+ 非权重算子 ~13 + lm_head 1.5 + 余量。
- **待定罪（决定 #1 刀形态）**：256K 注意力 11.9 ms/步的两种可能——(a) KV 每卡复制全量读 9.1 GB ÷ 750 GB/s；(b) KV 头切分但 decode 内核低效（2.3 GB ÷ ~200 GB/s，= v100-skinny §A-2 的"标量注意力"病）。**双 ctx（8192 vs 262144）显存差分在测**：差 ≈9 GB ⇒ (a) 刀 = TP 布局头切分；差 ≈2.3 GB ⇒ (b) 刀 = decode 注意力内核效率（抄 sm70-attn 的内核内 q4_0 直读改造 XQA）。
- **刀序（R367 重排后）**：① target verify 59.2（其中注意力 11.9 待定罪、非权重 13 融合、M=8 增量 5、lm_head 1.5 拆分）② draft+selector 16 ③ 轮内 ~14 ms 未归因（分账补测）④ 预填充期注入分块合并（spec-on TTFT 23 s）⑤ ~~解码期 injection~~（判死）。

### R368 ★★ 1cat decode 核：三处 ABI 修好后 131 调用干净，但**受阻于 F3 图捕获安全 + staging 流量**（判"暂缓"）；verify 59.2 ms 之谜立项（2026-09-25）

- **ABI 修复链（三处真 bug，均已修）**：① 头检查 `hq==24 && hkv==4` 改 GQA 6:1 任意对 ⇒ **[SM70EXEC] 首次真正执行**（`q=4 kv=236544 hq=6 hkv=1`，TP 头切分确认）② scratch 按 `n_parts` 分配 → 内核实际按编译期常量 `kSplits=80 × MAX_Q=8 × 6` 索引（`partial [80,8,6,256]`）⇒ 越界写；已改固定工作区 ③ `block_table`/`seq_lens` 内核按 **query token**（`batch_idx=q.size(0)`）逐行索引 ⇒ 需 n_q_pad 行，原 1 行 ⇒ 越界读到垃圾页号。修后 131 次调用（kv staged/kernel done/scatter done 全过）。
- **判"暂缓"（不是判死）**：第 132 次崩 `operation not permitted when stream is capturing` —— 我的胶水每调用 `cudaMalloc/cudaFree`+host 参数，撞上 F3 子图捕获/回放（回放时 host 代码不执行、kv_len 等参数被烘焙）。要继续需：持久工作区（alloc_size 期分配）+ 参数走设备内存 + 去 host 同步 + **免 staging**（当前每调用把全 KV 转 f16 = 9.8 GB 流量 ≈ 本身 12 ms，与 stock 相当）⇒ 报酬不确定，先转刀。
- **verify 之谜（新立项，#1 目标）**：256K 轮 92.6 ms 里 **target verify = 59.2 ms**，而 v100-skinny 同模型族同卡的 verify（M=8）仅 **~5 ms**（+0.383 ms/行 × 8）⇒ **12x**。我方 verify 成分：权重 9 + 256K 注意力 ~12 + 非权重算子 ~13 + M=8 GEMM 增量 ~5 + lm_head 1.5 ≈ 40，**~19 ms 未归因**。已在测：M=8 全形状 GEMM 账 + spec-on 生成期 GPU 利用率。
- **刀序（R368）**：① verify 59.2 之谜（归因 → 融合/M=8 GEMM/lm_head）② draft+selector 16（其注意力 ~6-7 ms = 草稿 KV 读全上下文，草稿 KV dtype/窗口待查）③ 轮内 ~14 ms 未归因 ④ 预填充期注入分块（spec-on TTFT 23 s）⑤ 1cat 核（暂缓）⑥ 解码 injection（判死）。

### R369 ★★ 三刀判负/封存 + verify/draft 成本模型定型（斜率实验）：草稿 KV dtype ✗、注入批调大 ✗（23 s TTFT = 草稿注入前向真工时）、1cat 核封存；**fusion 线成为主刀**（2026-09-25）

- **斜率实验（NMAX=2/5/7，256K spec-on，target decode+sync）**：43.90(3 tok) / 51.70(6) / 55.03(8) ⇒ **target verify = 37.2 ms 固定 + 2.23 ms/token**；draft_decode = 12.09/13.88/13.90 ⇒ **draft block = ~12 ms 固定 + 0.36 ms/token**（8 次迭代/块的形态，10x 于其 1.3 ms 权重地板）。
- **M=8 GEMM 无罪**：`MUL_MAT(q8_0,m=4096,k=14336)` n=1 = 82.83 µs vs n=8 = 112.11 µs = **1.35x** ⇒ 每 token GEMM 增量 ~0.4 ms，verify 斜率 2.23 的其余 ~1.8 ms/token ≈ PV 相位每 query 重读 V。
- **spec-on 生成期 GPU 利用率 = 85%**（与 spec-off 同）⇒ 轮是 GPU 工时受限，非 host。
- **判负 ① 草稿 KV dtype（DCTK/DCTV 旋钮）**：q8_0 比 f16 **慢 2 ms/轮**（draft_decode 15.12 vs 13.12，dequant 税）⇒ 草稿 KV 保持 f16。
- **判负 ② 注入批调大（LLAMA_SPEC_DRAFT_UBATCH）**：64→512 TTFT 175.3→176.9（无收益）、1024 OOM（草稿缓冲 2155 MiB）⇒ 预填充期 17.4 ms/call 的 sync 实为**每 2048-token 块一次的 GPU 尾巴**（550 ms ÷ 32 调用）；spec-on TTFT +23 s = **草稿注入前向的真实 GPU 工时**（236K token 过 5 层草稿 ≈ 0.1 ms/token），要收只能**与目标预填充重叠**（管线化），非分块问题。
- **封存 ③ 1cat decode 核**：已修 4 处真 ABI bug（头检查 24/4→GQA6:1、scratch 固定 [80,8,6,256]、block_table/seq_lens 按 query token 逐行、每 KV 头平面偏移），131 调用干净后仍有 IMA（vendored 内核内部契约与胶水不匹配）；env 默认关 = 生产零风险。**复活条件**：捕获安全工作区（已做）+ 内核 ABI 对拍（用 1cat 原始 caller 的 TORCH_CHECK 形状表）+ 免 staging（f16 直传已做）。
- **fusion 线升为主刀**（R368 刀序 ①②合并）：每步 ~13 ms 非权重算子（每层 ~30 kernel：RMS_NORM 257/步、MUL 225、SCALE 192、GET_ROWS 193、ADD 176、UNARY 160、CPY 144）× 目标+草稿两处 ⇒ 轮 ~27 ms 可收；对应 v100-skinny 的 GDN spec-state 快路（去 21 同步/70 拷贝/步）与 sm70-attn 的融合先例。

### R370 ★★★ P0+P1 落袋（GQA 打包 ncols2=3 + f16 KV）：256K 纯步 32.59→**27.12 ms/token（-17%）**、注意力差分 11.59→**5.65 ms（-51%）**；spec-on 侧待 N8T；f16 KV + spec-on @256K = OOM（2026-09-25）

- **轮子直接落地（用户授权 copy）**：`1cat-vllm-v100-study/patches/0001-t2-001-gqa-packing-sm70.patch`（jackinthebox52/qwen38-v100-serve T2-001，16 hunk 只改 `fattn-vec.cuh` +71/-21）`git apply --check` 过 → 已入树。机制：vec 核 `ncols2=1`（零打包）⇒ GQA=6 时 6x KV 重读；打包 `ncols2=3` ⇒ 2x。守卫 `packing_supported = K/V 都 F16/BF16 && gqa_ratio%3==0` ⇒ 对 q8_0 现役配置**惰性零风险**。
- **实测（256K spec-off 纯步，f16 KV）**：无打包 32.59 → **+打包 27.12 ms/token**；注意力差分（256K−5K）11.59 → **5.65 ms** ✓ 与"消掉 3x 重读"的预测同向（X1 先例 +44.9% 是 128K 单卡口径，我方 256K TP4 = -17% 纯步）。
- **spec-on 侧不受惠**：verify（q=8）走 TILE（`Q->ne[1]*gqa_ratio_eff=16>2`），补丁只覆盖 `Q->ne[1]==1` 的 VEC 分支 ⇒ 轮成本 90 ms 不变。**下一刀 = N8T**（TILE `ncols2=2→3`，配方在 PLAN-GRAPH §3.6：`launch_fattn_tile_switch_ncols2` 的 %2 判断前插 `gqa_ratio%3==0` → `switch_ncols1<256,256,3>`；约束 #28761：`ncols1*ncols2≤32` ⇒ 只能 8×3=24）。⚠ R165"收益≈0"的判词是 ctx 斜率口径（0.070-0.094 µs/ctx-token，与我方实测 0.09-0.11 同值）；但 256K 下 spec 轮的注意力差分 +21-25 ms/轮 仍在账上，N8T 在长上下文的真实收益**待实测复核**。
- **VRAM 包络卡死 f16 KV + spec-on**：f16 目标 KV 4.3 GB/卡 + 草稿 1 GB + KQ_mask 2.15 GB + 权重 10.5 GB ⇒ OOM（>16 GB/卡）⇒ **生产 KV 保持 q8_0**；打包的 f16 前提使其只能服务纯步/短 KV 场景，除非 ① 草稿 KV 降 q8_0（-2 GB，代价 draft_decode +2 ms）② 或 q8-direct（`LLAMA_SM70_Q8_DIRECT=1` 回收 470 MB f16 镜像，R319 遗产）。
- **刀序（R370）**：① N8T（TILE 打包，verify 侧 -10~15 ms/轮潜力）② verify 斜率 2.23 ms/token 之谜（与注意力差分分开归因）③ draft block 12 ms（10x 地板）④ fusion（目标+草稿 ~27 ms/轮）⑤ 1cat 核（封存）。

### R371 ★★ N8T 判负（TILE 打包零收益，R165 判词 256K 复证）+ verify 斜率归因定案（斜率与 ctx 无关 = 每 token 固定工；截距 ctx 部分 = 14.9 ms = q8_0→f16 暂存）（2026-09-25）

- **N8T 实现并判负**：`fattn-tile.cuh` 加 ncols2=3 专用阶梯（cols 24/12/6/3 + 4 配置项，`nthreads` 按 cpw=2^n 选 96/192/384）+ env `LLAMA_FA_TILE_GQA3`。**贪心 sha 逐位相同（4d6ae837…）= 数值正确 ✓**，但 256K spec-on **target verify 57.197（关）vs 57.177（开）ms/轮 = 零收益** ⇒ TILE 路的 GQA 冗余确实被 L2 吸收（R165 判词在 256K 亦成立）；打包刀的收益只在 VEC 路（q=1，R370 已落 -17%）。
- **踩坑两条（写进工装认知）**：① ncols2=3 的通用阶梯 `cols_per_block/ncols2` 非整除 ⇒ 必须专用阶梯（R155 补正复证）② 配置项被所有 DKQ/DV 实例化 ⇒ 仅加 256/256 会触发别处 static_assert；且 `cpw=ncols/nwarps` 必须 2 的幂（memcpy 尺寸）⇒ nthreads 96/192/384 而非 256。
- **verify 斜率归因（5K vs 256K 同源对照，spec-on）**：5K NMAX=2/7 ⇒ 27.238/35.440 ms（斜率 **1.64 ms/token**、截距 22.3）；256K ⇒ 55.0/57.2（斜率 2.23、截距 37.2）⇒ ① **截距随 ctx +14.9 ms = 256K 的 KV 读 + q8_0→f16 暂存**（f16 KV 可省 ~7-8 ms，但 spec-on VRAM OOM，见 R370）② **斜率 1.64 ms/token 与 ctx 无关 = 每 token 固定工**（GEMM M>1 增量 + GDN 状态 + 小算子数据量），非 KV 重读 ⇒ 需逐 token 算子级归因。
- **draft 块 = 固定 ~12 ms/轮**（NMAX=2→11.81、7→12.99；25K 12.96、256K 13.12）⇒ 与 token 数、ctx 都无关 = 5 层单次前向的**固定开销**（其权重地板 1.3 ms）⇒ 10x 地板，待插桩。
- **刀序（R371）**：① verify 截距 ctx 部分（f16 KV + VRAM 瘦身 或 核内 q8_0 直读）② verify 斜率 1.64 ms/token（逐 token 算子）③ draft 固定 12 ms ④ fusion（每层 ~30 kernel）⑤ N8T/1cat 核（判负/封存）。

### R372 ★★★ draft 块 12 ms 定性（LLAMA_SPEC_SYNC_SPLIT 探针）：**真实 GPU 工时非 host 开销**（wait=-12.65 = decode 调用内已等完）；轮 GPU 账闭合 ⇒ fusion 线 = 唯一能同时打 verify ops / draft ops / 每 token 工三处的刀（2026-09-25）

- **探针**：`LLAMA_SPEC_SYNC_SPLIT=1` 在 draft 的 `llama_decode` 后加同步并计 `wait = sync耗时 - decode耗时`。实测 256K：`draft_decode=13.56 ms/round, wait=**-12.65**` ⇒ 负值 = **decode 调用内部已等待 GPU 完成**（不是 host 提交开销）⇒ draft 的 13.5 ms 是**真实 GPU 工时**；注入同样 `sync=17.10, wait=-1.47`。整轮基本串行在 GPU 上（与 85% 利用率、R366 的 GPU 执行受限判定自洽）。
- **轮 GPU 账闭合（256K，~90 ms）**：verify 权重 12（Q8_0 地板 9 + M=8 增量 3）+ verify ctx 部分 14.9（q8_0→f16 暂存 + KV 读）+ verify 每 token 工 ~11（1.64/token，非 KV 重读）+ draft 13.5 + 注入/selector/间隙 ~15。**~35 ms = 算子数/内核效率**（verify ops ~16 + draft ops + 每 token 工）。
- **draft 13.5 ms 之谜收窄**：一次 `llama_decode`（3759 次调用中仅 31 次是块调用 ⇒ **每轮 1 次前向**）、ctx 无关（25K 12.96 / 256K 13.12）、token 数弱相关（3 tok 11.81 / 8 tok 12.99）⇒ **一次 5 层 8-token 前向 = 13.5 ms（10x 其 1.3 ms 权重地板）**，成分待逐算子归因（疑似 8 个块位置的算子链/权重重读）。
- **fusion 线 = 主刀定案**（与 PLAN-GRAPH R179 合流）：目标 + 草稿两处算子数（每层 ~30 kernel、~1550 kernel/步）是唯一能同时打三处的杠杆；对应 v100-skinny 教训 #3（GDN spec-state 快路去 21 同步/70 拷贝/步）与 sglang SK8（主机侧小算子 +19.4%）。
- **刀序（R372）**：① fusion（RS 状态链 529 op/步、RMS_NORM+ADD 433、MUL+UNARY 385）② verify ctx 部分（核内 q8_0 直读，去暂存 -7~9 ms）③ draft 13.5（逐算子归因后定刀）④ 1cat 核（封存）。

### R373 ★★ 逐算子真值修正预算（decode 形状）：小算子仅 **~5.3 ms/步**（2-5 µs/个）非 16；两谜团定位（verify 斜率 1.64 ms/token、draft 固定 12 ms）；attention 暂存 = 最大可归因项（2026-09-25）

- **逐算子真值（R203 方法复用，`test-export-graph-ops` + `test-backend-ops perf --test-file`，只取 tok=1 案例）**：RMS_NORM 4.0 µs、SCALE 5.1、CPY 4.7、GET_ROWS 3.4、MUL 2.4、ADD 2.3、SWIGLU 2.3、SILU 2.1、SIGMOID 2.0、SOFTPLUS 1.9、SSM_CONV 2.4、CONT 3.0、GATED_DELTA_NET（decode 案例未覆盖，prefill 案例 457 µs）。按 [OP] 计数加权 ⇒ **小算子合计 ~5.3 ms/步**（此前按 8-10 µs 估的 12-16 ms **作废**）。
- **纯步预算闭合（5K，f16 KV+打包 27.12 ms）**：GEMM 9（27 GB/4 卡 ÷ 751 GB/s）+ 小算子 5.3 + 注意力 5.65 + lm_head 1.5 + GDN ~2.5 + 间隙 ~3 ✓ 自洽。
- **两谜团（未归因）**：① **verify 斜率 1.64 ms/token（5K）/2.23（256K）** —— 与 lm_head M=1 的 1.54 ms 严丝合缝但草稿侧证伪（draft 3→8 tok 只 +0.24/token，若是每 token lm_head 应 +1.54）⇒ 待查 ② **draft 块固定 12 ms**（3 tok 11.81 / 8 tok 12.99 = 几乎不随 token 数变化）⇒ 疑似块扩散的固定迭代数（每次迭代重读权重 1.3 ms × N）。
- **最大可归因可修项 = attention 暂存 14.9 ms**（256K 截距 ctx 部分）：q8_0→f16 全量暂存 9.8 GB 流量；f16 KV/全镜像都被 spec-on VRAM 卡死 ⇒ **唯一 VRAM 中性解 = 核内 q8_0 直读**（sm70-attn 的 "in-kernel q4_0 direct loads" 同构先例，需移植到 q8_0）。
- **fusion 价值重估**：小算子仅 5.3 ms ⇒ 融合省 ~2-3 ms/步（非 27）；**主刀重排**：① attention 暂存（-7~10 ms/轮）② 两谜团归因 ③ fusion（-2~3）④ lm_head 拆卡（-1）。

### R374 ★★ n_max 256K 双 rep 判平（刀序收敛到代码线）+ draft 图结构勘察（dflash.cpp）；两谜团候选因定位（2026-09-25）

- **n_max 256K 双 rep（度量底线）**：NMAX=5 tg 32.17/34.13（AL 2.57/2.70）、NMAX=7 tg 33.33/31.76（AL 2.77/2.65）⇒ **统计上判平**（离散带内）；且 **AL 波动 2.6↔5.9 同配置跨跑** ⇒ tg 对比被内容噪声主导，**轮成本（~90 ms 稳定）才是物理量**。n_max 配置刀**判死**（保持 7）。
- **draft 图结构（`src/models/dflash.cpp`）**：5 层 ×（`build_dflash2_conv`（conv tap 链）+ `dflash_attn_conv_proj`/`dflash_ffn_conv_proj` GEMM + 注意力 + FFN）+ DFlash2 selector（`dflash_selector_hidden [5120,256]` + `next/prev [256,248320]` = 127 MB×2，get_rows + score_run）⇒ 一次前向 ~120-200 节点。**draft 12-13.5 ms 的成分仍未闭合**（GEMM ~1.3 + 算子 ~0.4 + selector ~0.3 ≈ 2 ms vs 实测 12-13.5）⇒ 下一手：conv tap 链的 `ggml_cont/concat` 实测、以及逐算子 dump（`test-export-graph-ops` 用 draft 模型）。
- **verify 斜率 1.64 ms/token 的候选因**：lm_head 逐位置假设**被 MTP 图代码否证**（`build_lora_mm(head_w, cur)` 是批量 M=8 ✓）⇒ 斜率另有其源（GEMM M>1 增量 0.24 + GDN 状态 0.25 + ?）；**下一手：以 draft/verify 的真实图 dump 逐算子对账**（R203 方法 + draft 模型导出）。
- **刀序（R374，收敛）**：① attention 暂存（verify ctx 14.9 → 核内 q8_0 直读，-7~10 ms/轮，最大可归因项）② draft 12 ms 归因（conv 链/逐算子 dump）③ verify 斜率归因 ④ fusion（小算子仅 5.3 ms，-2~3）⑤ 配置刀全部判死（n_max/注入批/草稿 KV dtype）。

### R375 ★★★ 证据链审计（用户批准"先修证据链"）+ 口径定案：全树哈希对账 2061/2066 一致、4 处漂移（`LLAMA_SPEC_REJ` 从未进过二进制）+ 吐字判据只锁 tpot/tg + KV 转换税立项（2026-09-25）

- **方法**：CR 归一化 md5 全树对账（本地工作树 2066 源文件 vs 构建树 `/root/llm/test/v100-opt/llama.cpp` 2082）+ 服务器独有代码回收 + 清单门控构建（fail-closed）+ manifest。脚本：`.tools/tree-md5.sh`、`.tools/p4-build.sh`（服务器 `/tmp/`）。
- **对账结果**：**2061 一致 / 4 差异 / 1 仅本地 / 16 仅服务器**（16 = `build-nccl/` 构建产物 15 + `common/build-info.h`，白名单）。
  - **`common/sampling.cpp`**：本地 `3905bcf3…`（含 `LLAMA_SPEC_REJ` 精确拒绝采样）vs 构建树 `21998222…`；**payload 名单里没有该文件**（`p3-build.sh` 只覆盖 24 个 `/tmp/p3-*`）⇒ R348/R349 定为 **P8 二攻方向**的机制**从未进过任何被测二进制**。"build 绿 + 门值绿 ≠ 改动进了二进制"的第三次实证。
  - **`ggml/src/ggml-cuda/fattn-common.cuh`**：**构建树多 6 行** —— `GGML_CUDA_FA_SPLIT_FLOOR`（强制抬高 FA 的 KV split 数，env 门控），实机加过、本地缺失 ⇒ **已回收到本地并逐字节对齐**（两侧 LF-md5 `68f700aa…`）。
  - **`src/llama-kv-cache.h`**：本地多 1 行注释（R296b 的 r=1.25 记录）⇒ 构建树落后，已同步。
  - **`ggml/src/ggml-cuda/fattn-79t/prefill.cu`**：子目录档案副本陈旧（7284 vs 7749 行）⇒ **不进构建**（`CMakeLists:106` 从 GLOB 移除、`:130-139` 单独 OBJECT 目标编译**扁平名** `fattn79t-prefill.cu`；该扁平名两侧一致）⇒ **预填充冻结值对应的引擎源码未被污染** ✓（判据：`build-instr/.../ggml-cuda-fattn79t.dir/fattn79t-prefill.cu.o` 存在）。
  - **`sm70-long/scalar-attention.cu`**：仅本地存在（`sm70-long/` 子目录默认不编译）⇒ 已补齐。
- **处置**：构建树已与本地工作树逐字节对齐；**`p4-build.sh` 取代 `p3-build.sh`**（全树对账 → 漂移即拒绝构建 → 构建 → 落 `$LIBS/BUILD_MANIFEST.txt`：时间/清单 sha256/文件数/两个二进制的 md5/BUILD_RC）；基准二进制留档 `/root/libdir-gb-bak-20260925-1521`（`libggml-cuda.so` md5 `7c4325fb2eda2f1e52daeab1bab7619a`）。纪律见 AGENTS.md §1.0.2。
- **口径定案（用户 2026-09-25 复核）**：**硬指标只有两条 = tpot ≤ 8.0 ms（⇔ tg ≥ 124.6 t/s）与 TTFT ≤ 152.5 s**；`ms/轮 ≤ 27.2` **降级为派生诊断值**（= 8.0 ms × AL 3.4，**vLLM 不报 AL**，该 AL 系反推）⇒ 不得单独作为不及格判据（AL 5.9 时 47 ms 即等价达标）。
- **预填充口径修正**：**必须按 spec-on 比**（BL1 全时段带 DFlash2）。我方今日 **22 个 256K spec-on rep 的 TTFT 全部落在 174.0–177.4 s**（pp 1332-1358；另有 4 个异常 rep = 181.7 / 182.3 / 200.3 / 340.1 s，未查因）vs BL1 **152.5 s** = **1.15x，未达标**；spec-off **151.7 / 152.5 s** 只是纯目标侧数字 ⇒ **R326「预填充达标（1.004x）」结论作废**（不删史）。缺口 ~23 s = 草稿链预填充前向（236K token × 5 层 × 1.92B ≈ 900 TFLOP，20-40 TFLOPS 有效算力 ⇒ 22-45 s，与实测吻合）。
- **吐字实测方差（今日 23 个 256K spec-on rep，原始 `-stress.log` 全量统计；**仅 3 个 tg > 50**）**：tg **26.1–63.6**、接受率 23.7–73.8%（同二进制 `/root/libdir-gb`、同 prompt `bl-prompt90.txt`、同采样参数）⇒ **tg 单点读数不可比**，轮成本（~83–106 ms）才稳定。**R366 的 tg 63.6 与 R374 的 tg 31.8 是同一配置的两次抽样，不是性能回归**（判读修正）。
- **KV 转换税立项（R373「最大可归因可修项」的形式化，代码实证）**：`fattn.cu:763-780` 对 `TILE`/`MMA_F16` **无条件** `need_f16_K = need_f16_V = true`；Volta 下 q=8 的 verify 必走 TILE（`:666`，`Q->ne[1]×gqa_ratio_eff = 8×2 = 16 ≤ 16`）⇒ `fattn-common.cuh:1026/1054` **每次调用**全量 `to_fp16(K_data, K_f16, ggml_nelements(K))`。按每卡 1 KV 头（R368 实测 `hq=6 hkv=1`）算：q8_0 KV ≈ 2.28 GB/卡、f16 镜像 4.56 GB/卡（与 R370「f16 目标 KV 4.3 GB/卡」吻合）⇒ **每步 16 层 ≈ 6.8 GB/卡 ≈ 9 ms @751 GB/s**。上游**已有**原生 q8_0 直读实例（`fattn.cu:413-459`：`FATTN_VEC_CASES_ALL_D(…, Q8_0)` 含 D=256），但 Volta 分支只在 `Q->ne[1]×gqa_ratio_eff ≤ 2` 才选 VEC ⇒ **decode verify 永远拿不到它**。
- **判死/改判（只改灰不删）**：① **1cat decode 核"判死"依据是 5K 口径 A/B**（5K 下解码注意力总耗 0.2-1 ms，天然无收益空间）⇒ 对 256K（差分 11.6-14.9 ms）**无效**，改判为"**形态错，非无收益**"；② `LLAMA_SPEC_REJ` 需重立实验（从未测过，与"判负"无关）；③ N8T 判负为**单 rep**（差 −0.02 ms）⇒ 复核时须 ≥2 rep。
- **构建与门值（本里程碑收口）**：清单门控构建 `p4e` **BUILD_RC=0 / ERROR_LINES=0 / MANIFEST_DIFFS=0**（`cmake --build build-instr -j176`，70 s 增量）；manifest sha256 `b97e84439d8c84b303ac5e56893b713dd2d4398e0538ba9d8ee1165482f41314`（2066 文件）、`libggml-cuda.so` md5 `7c4325fb2eda2f1e52daeab1bab7619a`（**与备份基准逐位相同 ⇒ 今日全部 CUDA 侧 256K 实测数据仍有效**；变的是 `llama-server` md5 `47467cf6f2234cf73c87cbc3fa239433`）；**门值 `SPEC=1 t1c-gate.sh`：g1 = g0 = `bcda0092bfbaaa9d…`（len=131, predicted=32）⇒ 默认路径零回归** ✓。
- **顺带修掉两处真缺陷（"REJ 从未编译过"的独立铁证）**：① `common/sampling.cpp` 缺 `<random>`（`std::mt19937` 未声明）；② `common/sampling.h:86` 的声明仍是 **5 参**而实现已改 **6 参** ⇒ `libllama-server-impl.so` 链接报 `undefined reference to common_sampler_sample_and_accept_n(..., bool)`。修法：头文件声明补默认参数 `const std::vector<float> & draft_p = {}`（调用方零改动），REJ 段注释改为"生成端尚未提供 draft_p ⇒ **当前 no-op**"。**若该机制曾进过任何构建，这两处必然早已暴露**。
- **刀序（R375）**：① **KV 转换税**（预期 −9 ms/轮，先做同源 A/B）② verify 未归因分相（逐节点 GPU 计时，替代坏掉的 nsys/OP_TIMING）③ 轮内重叠（draft/注入/selector 互等）④ 预填充草稿与 target 重叠（−23 s 缺口）。
- **⚠ 未决（下一件必做）**：**AL 方差来源未归因**（内容/采样种子/接受准则三选一）⇒ 决定后续所有 tg 型 A/B 是否可信。

### R376 ★★★ KV 转换税第一刀落地：TILE 内核 q8_0 直读（免 f16 全量镜像）——256K 轮成本 **83.95 → 78.75 ms（−5.2 ms，−6.2%）**、门值绿、预填充零回归；E1（Split-D 核 + q8-direct）判负复证（2026-09-25）

- **E1（先试零代码路线）判负**：`LLAMA_SM70_D256=1 LLAMA_SM70_79T_DECODE=1 LLAMA_SM70_Q8_DIRECT=1`（q8-direct 代码已在树内：`sm70_q8_dequant_one/group`、`k_direct` 免镜像路由）。256K spec-on ABBA：A（TILE+staging）tpot 28.45/27.04、轮 86.2/86.0 ms；**B（Split-D+q8-direct）tpot 125.04/144.12、轮 ~359 ms = 慢 4.2x**；spec-off 纯步 B tpot 320.7 ms vs A 33.3 ms = **慢 10x**。probe 实证路由：`ACCEPT: sm70 d256 + small-prefill (FISHLIKEXIE_BLACK_MAGIC)`（decode 形状被 small-prefill 分支接走，固定开销主导）⇒ **R339 判负在 256K + q8-direct 下复证**，Split-D 核不适合 decode。注：probe 形状 `K=(256,262144,4,1)` = **每卡 4 KV 头**（TP4 切分 16 KV 头），修正 R368 的 `hkv=1` 记述。
- **E2（TILE 加 q8_0）采纳**：`fattn-tile.cuh` 改造（模板加 `Kq8/Vq8`；`flash_attn_tile_load_tile` 加 q8 分支 + `flash_attn_tile_dequant_q8_0`（复用 `dequantize_V_q8_0<half,2>`，舍入与 staged 路径一致）；`iter_KQ/iter/kernel` 指针改 `const char *`，q8_0 时基址=行起始、`e0_base`=绝对元素索引；half2 粒度元素偏移恒偶 ⇒ 永不跨 q8_0 block）+ `fattn.cu`（TILE 的 `need_f16_K/V = !(K==Q8_0 && V==Q8_0)`）。
- **E2 数据（ABBA，唯一变量=TILE 改动；A=留档二进制 `libdir-gb-bak-20260925-1521`，B=`/root/libdir-gb`）**：
  | 臂 | libggml-cuda.so | tpot ms | AL | 轮成本 ms | TTFT s |
  |---|---|---|---|---|---|
  | e2a1 | 7c4325fb | 29.94 | 2.84 | 85.0 | 175.17 |
  | **e2b1** | **e497c968** | **27.31** | 2.87 | **78.4** | 175.28 |
  | **e2b2** | **e497c968** | **26.81** | 2.95 | **79.1** | 175.37 |
  | e2a2 | 7c4325fb | 32.50 | 2.55 | 82.9 | 175.31 |
  **A 均值 83.95 → B 均值 78.75 ms/轮（−5.2 ms，−6.2%）**；B 离散 0.7 vs A 2.1 ⇒ 收益稳定。TTFT 175.2–175.4 四臂一致 = **预填充零回归** ✓；门值 `SPEC=1 t1c-gate.sh` g1=g0=`bcda0092…`（131 chars）✓。
- **构建（§1.0.2 全流程）**：`p4-build.sh` 清单门 `MANIFEST_DIFFS=0`、`BUILD_RC=0`、`ERROR_LINES=0`、205 s（`-j176`）；manifest sha256 `873aecdd…`；`libggml-cuda.so` md5 `7c4325fb… → e497c968…`（**改动确证进二进制**）、`llama-server` md5 `47467cf6…` 不变。
- **收益归因（待查）**：预期 −12 ms（转换 ~9 + 读减半 ~3），实测 −5.2 ms（43%）。候选：转换的读被 L2 部分吸收、转换与其他 kernel 重叠、q8_0 dequant 的 ALU 抵消。留待逐节点计时（刀序②）。
- **新采用值（256K spec-on）**：轮成本 **78.75 ms**、tpot 26.8–27.3 ms、AL 2.87–2.95、tg 36.6–37.3；对 BL1（tpot 8.0）= **3.35x**（此前 3.4–3.9x）。tg 判据仍以轮成本+AL 并列（AL 方差未归因）。
- **遗留**：① q8_0 直读未加 env 门控（默认生效）——预填充走 79T 引擎不受影响、门值/TTFT 双绿，符合冻结条款；如需回退可用留档二进制。② MMA_F16 路径仍硬编码 f16（q>16 形状）。③ 预期收益缺口待归因。

### R377 ★★★ 刀序②交付：逐算子 GPU 账表打通（5 个 bug 修复），256K decode 每步 GPU ≈68 ms + host ~10 ms = 实测轮 78.75 ms ✓ 账目闭合；verify"未归因 15 ms"与 draft 12 ms 双双定案（2026-09-25）

- **工具**：`GGML_CUDA_OP_TIMING=1`（逐节点 CUDA event）经 5 轮修复后**数据干净**：
  ① 捕获态 `cudaDeviceSynchronize` 崩溃（meta 子图捕获中）→ `cudaStreamIsCapturing` 守卫
  ② fuse 跳过节点的 event 被查询（R333"越界"）→ `rec_flag` 标记实际 record 的槽
  ③ TP4 四线程共享累加器竞争 → per-device 累加
  ④ 多流分支并发（`concurrent_events`）污染时间戳 → `GGML_CUDA_NO_CONCURRENT=1` 诊断开关（默认关）
  ⑤ **真根因：`rec_flag` 跨图不重置**（event 池复用，旧标志让新图查到未重新 record 的 event ⇒ elapsed 为负、求和被污染）→ 查询后清零
  门值 `bcda0092…` 每轮绿（诊断默认关）；libggml-cuda.so `d598efd2→3c3697b4→…→最终版`。
- **测量条件**：256K spec-on、`GGML_CUDA_OP_TIMING=1 GGML_CUDA_DISABLE_GRAPHS=1 GGML_META_SUBGRAPH_CAPTURE=0 GGML_CUDA_NO_CONCURRENT=1`（禁捕获/多流以保证 event 语义；kernel GPU 时间不受影响）。
- **每步每设备 GPU 账（decode 段，256K spec-on）**：
  | 算子 | ms/步 | 占比 | 备注 |
  |---|---|---|---|
  | **FLASH_ATTN_EXT** | **25.4** | 37.6% | verify 16 层 + draft 5 层，avg 1.21 ms/call（21 calls/步） |
  | **MUL_MAT** | **25.2** | 37.4% | 权重 GEMM，665 calls/步，avg 38 µs |
  | CPY | 4.7 | 6.9% | |
  | ADD | 2.5 | 3.7% | |
  | RMS_NORM / GET_ROWS / SCALE / UNARY | 1.0–1.4 | ~7% | |
  | MUL / GLU / CONCAT / SET_ROWS / CONT / ROPE | 0.4–1.1 | ~7% | |
  | VIEW/RESHAPE/PERMUTE/TRANSPOSE | ~0.5 | <1% | R373"小算子 5.3 ms/步"复证 |
  | **GPU 合计** | **≈68** | | **+ host/gap ~10 ms = 实测轮 78.75 ms ✓ 闭合** |
- **三谜团定案**：
  ① **verify"未归因 15 ms" = MUL_MAT 超出静态地板**：R373 静态账"权重 9 ms"（27 GB ÷ 751 GB/s）vs **实测 25.2 ms** ⇒ **+16 ms** 即未归因真身（M=8 小 batch GEMM 效率、465 GEMM 的 launch/调度、GDN 小 GEMM）。
  ② **draft 12 ms = 它自己的 GPU 工时**（推翻 R372"等 target 尾巴"猜测）：MUL_MAT ~5 + FA ~6 + 小算子 ~2 = **13 ms ✓**。
  ③ **q8_0 收益缺口（预期 −12 vs 实测 −5.2）**：转换的读被 L2/重叠部分吸收 + dequant ALU 抵消；精确归因需 A/B 两臂 `[OP]` 差分（下一步）。
- **刀序影响（更新）**：三大件 = **MUL_MAT + FLASH_ATTN_EXT + CPY = 55 ms/步（82%）**。后续刀的预期收益可直接对账：MUL_MAT 线（融合/大批量/M=8 专用 GEMV）、FA 线（q8_0 直读已做、KV 重读、融合）、CPY 线（消除拷贝/视图）。
- **遗留**：`GGML_CUDA_OP_TIMING` 需 4 个 env 才干净（`DISABLE_GRAPHS` + `META_SUBGRAPH_CAPTURE=0` + `NO_CONCURRENT` + 本工具）⇒ 只用于诊断，不进生产。

### R378 ★★ FA split floor 刀判负（E6 扫描 + E8 GPU 时间复核）+ **重要方法论发现：墙钟 A/B 噪声 ±9 ms**（2026-09-25）

- **刀**：`GGML_CUDA_FA_SPLIT_FLOOR`（R375 回收的探针）——decode FA 并行度不足（q=8 + 长 KV 串行），E6 扫描 floor=0/1/2/4/8/16。
- **E6 墙钟**：floor=2 轮 64.9 ms vs floor=0 基线 70.3 ms ⇒ 看似 −5.4 ms。
- **E8 GPU 时间复核（`[OP]` 表，判据）**：`FLASH_ATTN_EXT avg` floor=2 **1.207 ms/call** vs floor=0（E3 对照）**1.21 ms/call** ⇒ **GPU 时间无差异** ⇒ **split floor 无效**（KV 重读被 L2 吸收，R371 判词复现）。
- **⚠️ 方法论发现（必须记住）**：**墙钟 A/B 的噪声是 ±9 ms/轮**——e8d（floor=0）61.7 ms vs e6f0（floor=0）70.3 ms，**同配置同时段内差 8.6 ms**。⇒ ① 单 rep 墙钟 A/B 的 ±5 ms 级结论不可信；② **判据必须用 `[OP]` 表的 GPU 时间**（或 ≥3 rep 取中位）；③ E2 的 q8_0（−5.2 ms）有 `[OP]` GPU 时间支撑（FA avg −0.2 ms/call）⇒ 可信 ✓。
- **处置**：fattn-common.cuh 的固化**已回退**（默认 off，探针保留）；门值 `bcda0092…` 绿、`BUILD_RC=0`。
- **②③④ 全景（截至本轮）**：
  | 刀 | 状态 | 收益 |
  |---|---|---|
  | TILE q8_0 直读（①） | ✅ 采纳 | **−5.2 ms/轮**（`[OP]` FA avg −0.2 ms/call） |
  | 2b' lm_head 切分粒度（②） | ✅ 已做 | −0~1 ms |
  | 2a GEMM/小算子融合（②） | 探明空间有限（mmid 形状不适用、CUDA fusion 已覆盖） | −1~2 ms |
  | 2c M=8 GEMV（②） | 探明 MMQ 已是 V100 最优（上游实测注释） | −0~1 ms |
  | ③ FA split floor | ❌ 判负（GPU 时间无差异） | 0 |
  | ④ GPU selector | 待做 | −2~3 ms |
- **结构性判断（待复核）**：距 BL1（tpot 8.0）的 3x 差距中，MUL_MAT 25 ms + FA 25 ms 是权重/KV 读的物理下限 + 内核质量差距；`②③④` 合计再收 ~5-8 ms 后仍差 ~2.5x ⇒ 需要 1cat 级内核（marlin FP4/FP8 GEMM、XQA FA）或结构性方案，而非调优。

### R379 ★★★ 1cat-vLLM 启动参数全量分析（用户指示"参数可照抄"）：**tpot 8.0 的结构性来源 = inductor 编译 + combo 内核 + 小 q CUDA graph**；`draft_sample_method: probabilistic` = AL 钥匙；`enable_prefix_caching` 我方未对齐（2026-09-25）

- **方法**：读 `/root/llm/systemd/vllm-1cat.service`（只读）+ vLLM 启动日志的 `compilation_config`；BL1 纯解码基线用 `/mnt/3.84t/Qwen3.8-27B-FP8`（红线 4：模型只从 /mnt/3.84t 加载）起测试实例（端口 8001，不碰正式环境/8000）。
- **参数对齐表**：TP4/tensor-split ✓、f16 激活 ✓、256K ctx ✓、max-num-seqs 1 ✓、chunked-prefill 2048（=ub 2048）✓、kv-cache fp8_e5m2 ≈ 我方 q8_0（8-bit 同级）✓、reasoning on ✓ ⇒ **配置基本对齐**。
- **★ 三个结构性发现（1cat 的 tpot 8.0 从哪来）**：
  | 1cat 机制 | 配置证据 | 我方现状 | 差距 |
  |---|---|---|---|
  | **combo_kernels + benchmark_combo_kernel**（inductor 组合内核，小算子合并 + 自动基准选优） | `inductor_compile_config: {combo_kernels: True, benchmark_combo_kernel: True}` | 无（CUDA fusion 只覆盖固定模式） | 小算子 17 ms 的解药 |
  | **fuse_norm_quant**（norm+quant 融合） | `pass_config: {fuse_norm_quant: True}` | 无 | RMS_NORM 8.6 ms + CPY 29 ms 部分 |
  | **小 q CUDA graph**（`cudagraph_capture_sizes: [1,2]`、`max_cudagraph_capture_size: 2`）| decode/draft 走 graph，verify(q=8) 走 eager | meta 后端 126 子图/ubatch | 调度结构差异（host/gap 10 ms） |
  | kernel_config: `rms_norm: ['vllm_c','native']`（自定义核优先） | `kernel_config` | 上游核 | 内核质量 |
- **AL 钥匙确认**：`--speculative-config '{"method":"dflash",...,"draft_sample_method":"probabilistic"}'` ⇒ 1cat 用 **exact probabilistic rejection sampling**（`min(1,p_t/p_d)`）——这正是 AL 2.7→3.4 的来源，阶段 A 已照抄（`sampling.cpp` REJ + `speculative.h` result_probs + walk 概率 + accept 传参，待构建验证）。
- **可照抄清单（按 ROI）**：① `probabilistic` 拒绝采样（阶段 A，在做）② `enable_prefix_caching`（**我方 t1c-run.sh 只有 slot-save，无前缀复用 ⇒ 可能是 TTFT 175 vs 152.5 的第二来源**，待验证）③ `fuse_norm_quant` + `combo_kernels`（阶段 C 的小算子融合，参考 inductor 的组合内核思路）④ `cudagraph_capture_sizes=[1,2]`（只捕小 q，verify 走 eager ⇒ 我方 FULLGRAPH 判负的旁证）。
- **⚠ 口径提醒**：1cat 是 FP8 权重（`quantization=fp8`）+ fp8_e5m2 KV；我方 Q8_0 权重 + q8_0 KV（同级 8-bit）⇒ 权重流地板同级，**差距在内核/编译层而非格式**（R375 已证）。
- **遗留**：BL1 纯解码基线（vLLM 无 DFlash2）跑到一半因路径红线重启，完成后填入"纯解码 tg"实测值（替代 36.7 的推算）。

### R380 ★★★ 差距分解修正（纯解码基线实测）：**纯解码只差 1.25x，投机轮开销才是病灶（2.9x）**；追平公式重写（2026-09-25）

- **三基线实测（256K，同 prompt/采样）**：
  | 指标 | BL1 纯解码（vLLM 无 DFlash2，`/mnt/3.84t` FP8） | BL1 投机（账本 152.5） | 我们纯解码（Q8_0 spec-off） | 我们投机（Q8_0 spec-on） |
  |---|---|---|---|---|
  | tg t/s | **37.14 / 40.88** | **124.6** | **29.76 / 30.27** | 32–46（E10 中位 37） |
  | tpot ms | 24.4–27.0 | **8.0** | **33.0–33.6** | 22–31（中位 27） |
  | AL | — | 3.4 | — | 2.84–2.99 |
  | 轮开销 ms | — | **27.2** | — | **65–87（中位 78.6）** |
  | TTFT s | 163.4–165.0 | 152.5 | 178.9 | 174.0–175.1 |
- **★ 三个决定性数字**：
  1. **纯解码差距只有 1.25x**（33.3 vs 24.4–27.0 ms/token）⇒ **内核质量差距远小于 R378 的"2.9x"估计**（那是把轮成本当纯解码 tpot 的推算错误，本次实测修正）。
  2. **投机轮开销差 2.9x**（78.6 vs 27.2 ms/轮）⇒ **真正的病灶**：verify 57 ms（BL1 ~15）+ draft 12 ms（~5）+ CPU selector 2.3 + 注入 3 + 间隙 10。
  3. **AL 差 0.8x**（2.9 vs 3.4）⇒ 拒绝采样是放大器。
- **追平公式（重写）**：
  ```
  tg = 1000 / (轮开销 / AL)   ⇒  达标 124.6  ⇒  轮开销 ≤ 27 ms @ AL 3.4
                                   或 轮开销 ≤ 40 ms @ AL 4.6
  ```
  ⇒ **主攻 = 轮开销（78.6 → 30 ms）**（内核融合 + GPU selector + 重叠），**放大器 = AL（2.9 → 3.4+）**（拒绝采样）。组合后 tg 113–126 = **边缘达标区间**。
- **E10 噪声确认**：干净配置 3 rep，tg 32–46、轮 65–87 ⇒ **墙钟噪声 ±11 ms/轮**（R378 的 ±9 ms 复证）⇒ 判据一律 `[OP]` GPU 时间。
- **对策（阶段重排）**：
  | 阶段 | 目标 | 手段 | 预期 |
  |---|---|---|---|
  | A 拒绝采样 | AL 2.9 → 3.4+ | `sampling.cpp` REJ（已接通，待构建） | tg 37 → 45 |
  | B-1 内核（verify/draft） | 轮 78.6 → 45 ms | v100-skinny MT=2（M=8 GEMM）、sm70-attn Split-D 修 small-prefill | tg 45 → 75 |
  | B-2 GPU selector | 轮 45 → 40 | 1cat `TP4_M1_FAST_SELECTOR` | tg 75 → 85 |
  | C 重叠/融合 | 轮 40 → 30 | combo_kernels（1cat inductor 思路）、注入重叠 | tg 85 → 113 |
  | D 验收 | ≥124.6 | 同口径 + 显存 ≤ 64 GB | — |

### R381 ★★★ 阶段 A 拒绝采样（`LLAMA_SPEC_REJ`）：**AL 提升有效（3.28–4.09，超 BL1 3.4），但 IMA 崩溃复现（非偶发）⇒ 保持默认关**；轮开销无差异（81–83 ms）（2026-09-25）

- **A/B（e12）+ 稳定性（e13，REJ=1 × 4 rep）**：
  | 臂 | tg | tpot ms | AL | 接受率 | 轮成本 |
  |---|---|---|---|---|---|
  | REJ=0（贪心匹配） | 33.4 / 44.3 | 29.9 / 22.5 | 2.66 / 3.75 | 23.9% / 40.3% | 79.6 / 84.5 |
  | **REJ=1（拒绝采样）** | **49.2 / 40.3 / 40.1** | 20.3 / 24.8 / 24.9 | **4.09 / 3.32 / 3.28** | **44.4% / 33.5% / 32.8%** | 82.9 / 82.1 / 81.7 |
- **✓ 有效**：REJ=1 的 AL **3.28–4.09**（均值 3.56，最高 4.09）vs REJ=0 的 2.66–3.75 ⇒ **拒绝采样提升 AL ✓，AL 4.09 已超 BL1 的 3.4**（1cat 的 `draft_sample_method: probabilistic` 同款机制）。
- **✗ Bug**：`CUDA error: illegal memory access`（IMA）在 `llama_decode → process_ubatch → synchronize`（device 3）——**e12 rep2 + e13 rep4 两次复现**（REJ=0 未崩）⇒ **非偶发，REJ 路径有 bug**。嫌疑：`result_probs` 与 `result` 长度错位（`p_min` break 时同步性）、或 `common_sampler_accept` 提前接受改变 sampler 状态、或 `idxs` 索引错位。
- **判定**：**REJ 保持默认关**（不入采用链）——① AL 是放大器，轮开销（81 ms）才是主因；② AL 4.09 @ 轮 84 = tg 49，**AL 4.09 @ 轮 31 = tg 132** ⇒ AL 价值须待轮开销下降后兑现；③ 默认关 ⇒ 门值/生产零风险。
- **遗留**：IMA 定位（ASAN 或 assert 边界）→ 修复后重测；本轮转向阶段 B（内核，主攻轮开销 81 → 31 ms）。

### R382 ★ 阶段 B-2（Split-D mma decode）判负：**mma 核对 decode（q=8）不适用，−84%**（E1 教训复现坐实）（2026-09-25）

- **先验**（staged-baseline 流程）：假设 Split-D mma 核（`mma.sync.m8n8k4`，TC）比 TILE（SIMT）快；预期 FA 25→15 ms（−40%）、轮 78.6→68 ms；若为负则 mma 对 q=8 长 KV 不利（TC 利用率 + 固定开销）⇒ 回 TILE、记灰。
- **改动**：`fattn-sm70-d256.cu` 的 `sm70_d256_min_q()` 加 `FISHLIKEXIE_BLACK_MAGIC=2`（decode 准入 min_q=1）⇒ env 门控同源 A/B。构建 `BUILD_RC=0`、门值 `bcda0092…` 绿。
- **A/B（e15，ABBA 4 臂，256K spec-on）**：
  | 臂 | 序 | tg t/s | tpot ms | 判定 |
  |---|---|---|---|---|
  | A（TILE，`BLACK_MAGIC=0`） | 1 | 39.27 | 25.43 | ✓ |
  | B（mma decode，`=2`） | 2 | **5.88** | **170.14** | ✗✗ |
  | B（mma decode，`=2`） | 3 | **5.47** | **182.78** | ✗✗ |
  | A（TILE） | 4 | 34.21 | 29.20 | ✓ |
- **判定**：**B 判负（−84%）**——A 臂两臂差 12.9%（墙钟噪声 ±11 ms，R378），但 B 臂 −84% 是**确定性退化**（远超噪声）⇒ 不采用。
- **灰色名单（为什么被否）**：**mma 核（`mma.sync.m8n8k4`）对 decode（q=8）不适用**——probe 显示 Split-D 只对 prefill（q=2048）正常（`Q=(256,2048,24,1)`），q=8 时固定开销主导（E1 的 `small-prefill` 判负教训复现坐实）⇒ **mma decode 路线关闭**。
- **A 臂基线复核**：tg 34.21–39.27、tpot 25.43–29.20 ms（与 E10 的 32–46 / 22–31 一致）⇒ 当前基线不变。
- **下一步**：B-2 关闭 ⇒ 转 B-1（GEMM MT=2，MUL_MAT 25 ms）或 C（combo_kernels 融合 + 注入重叠）。

### R383 ★ CPY+ADD 融合：代码保留、**默认关**（未测量不得进基线）；同源 A/B 开关 `GGML_CUDA_FUSE_CPY_ADD=1`（2026-09-26）

- 实现：`cpy.cu` 新 kernel `cpy_add_scalar_contiguous<src_t, other_t, dst_t>` + `ggml_cuda_op_cpy_add_fused`；`ggml-cuda.cu` 新谓词 `ggml_cuda_should_fuse_cpy_add` + `try_fuse` 分支（`{CPY, ADD}`，`return 1`）。
- 把关：谓词排除非连续 / 非 F16-F32 组合 / 别名写入，**不满足即不融合**（不靠断言崩溃）。门值两轮绿（`bcda0092…` 全等）⇒ 数值正确。
- **默认关的理由**：收益未测（墙钟噪声 ±11 ms 分辨不了 ~3 ms 效应），未测改动留在默认路径会污染后续所有 A/B 的基线（staged-baseline 铁律）。开关留作同源 A/B；待 W3（小算子）一击时启用测量。
- 基线恢复构建：`libggml-cuda.so` = `ebf9f721ab2a9b737328ee7aa2d96c0a`、`MANIFEST_DIFFS=0`、门值绿。

### R384 ★★★ 项目重审 + 吐字内核攻坚方案（用户 2026-09-26 指示"重新审视整个项目、解决吐字问题"；kernel-first）

**一、问题的定量表述（本轮全部实测，256K 同口径）**

| 指标 | BL1（vLLM+DFlash2） | 我们（llama.cpp+DFlash2） | 比值 |
|---|---|---|---|
| 轮成本 | **27.2 ms** | **78.6 ms**（65–87，23 rep） | **2.9x** |
| AL | 3.4 | 2.84–2.99 | 0.85x |
| tpot | **8.0 ms** | 22–31 ms（中位 27） | 3.4x |
| tg | **124.6 t/s** | 32–46（中位 37） | 3.4x |
| TTFT（spec-on） | 152.5 s | 174.0–175.1 s | 1.15x |
| 纯解码 tpot（spec-off） | 24.4–27.0 ms | 33.0–33.6 ms | **1.25x** |

**二、带宽地板（4×V100，每卡每轮）—— 决定性视角**

| 流量 | 体积 | @751 GB/s（V100 实测权重流率 R212） |
|---|---|---|
| 权重（Q8_0 27 GB / TP4） | 6.75 GB/卡/前向 | **9.0 ms** |
| KV（16 全注意力层 × 570 MB/层，K+V） | **9.1 GB/卡/轮** | **12.1 ms** |
| 地板合计（层内 GEMM → FA 串行，不重叠） | | **≈21–22 ms** |

⇒ **BL1 的 27.2 ms 就在地板上**（地板 + host）；**我们不是**：MUL_MAT 25.2 vs 9.0 = **2.8x off（36% 屋顶）**、FA 25.4 vs 12.1 = **2.1x off（48% 屋顶）**。**吐字问题 = 两个大核没跑到内存地板**，非模型/硬件问题（§1.0.1 原理判据）。

**三、目标拆解（到轮 ≤ 33 ms，再靠 AL 收口）**

| 成分 | 现在 | 目标 | 省 | 内核手段（含可抄件） |
|---|---|---|---|---|
| **FA** | 25.4 | **12–13** | −12 | q=8 长 KV 核：**KV 切分给足 CTA 并行度** + `mma.sync.m8n8k4`（D=256 k-slice）。抄件 `v100-refs/sm70-attn/`（Split-D N32 **D256 同头维**、同架构）。⚠ E1/E6/E15 已证：只"准入 q=8"不够（routing 到 small-prefill → 4.2x 慢；split floor 无效；mma decode −84%）⇒ **必须修并行度与流水，不是开关** |
| **GEMM** | 25.2 | **10–12** | −13 | Q8_0 版 **MT=2**：两个 m8n8k4 row-tile **共享一次权重流** + fp32 累加，M≤16 专用。抄件 `v100-refs/v100-skinny/kernels/skinny_kernels.cu` 的 `skinny_fp8_qpn8_mt2` / `skinny_nvfp4_qpn2`（V100 实测）⇒ 需写 **q8_0 codec 变体**（我们格式是 q8_0） |
| 小算子 | 17（CPY 4.7 / ADD 2.5 / RMS_NORM 1.4 / SCALE 1.3 / UNARY 1.2 / MUL 1.05 / CONCAT 0.8 / CONT 0.66 / GLU 0.6 / SET_ROWS 0.6 / ROPE 0.42 / views 0.5） | 5–6 | −11 | 融合模式扩展（已有 14）+ **图谱层消除冗余拷贝**（先查清 CPY 4.7 是哪些：KV 写 / rope 输出 / GDN state） |
| host/gap | 10 | 3 | −7 | GPU selector（1cat `TP4_M1_FAST_SELECTOR`；我方 in-graph selector 启动崩 = **待修 bug，非判负**）、tail graphs、注入重叠 |
| **合计** | **78.6** | **30–33** | **−45~−48** | |

**四、AL 是另一半（乘法因子）**：`tpot = 轮 / AL`。
- 轮 33 @ AL 3.4 = 9.7 ms（tg 103，不够）；轮 33 @ **AL 4.1** = **8.05 ms（tg 124）✓**
- **REJ 已有实证 AL 3.93 / 4.09**（1cat `draft_sample_method=probabilistic` 同款），历史最高 AL 5.9–6.03 ⇒ **修好 REJ 的 IMA（W0）是最便宜的杠杆**：单此一项 tg 37 → 51（+38%）
- ⇒ **W0（AL）+ W1/W2（两大核）互补，缺一不可**：只做核 → tg ≈103；只做 AL → tg ≈51

**五、执行序（按 (收益 × 成功率)/成本）**

| # | 工作项 | 目标 | 预期 | 成本 |
|---|---|---|---|---|
| **W0** | **REJ IMA 根因修复**（ASAN / CUDA_LAUNCH_BLOCKING 定位；env 门控已通） | AL 2.9 → 3.9+ | tg 37 → 51 | 0.5–1 天 |
| **W1** | **FA q=8 长 KV 核重写**（KV 切分并行 + mma D=256；抄 sm70-attn） | 25.4 → 13 | −12 ms | 2–4 天 |
| **W2** | **GEMM Q8_0 MT=2 核**（抄 v100-skinny，写 q8_0 codec） | 25.2 → 11 | −14 ms | 3–5 天 |
| **W3** | **小算子**：先查清 CPY 4.7 构成，再融合/消除 | 17 → 6 | −11 ms | 2–3 天 |
| **W4** | **host/gap**：修 in-graph selector 崩溃 + tail graphs + 注入重叠 | 10 → 3 | −7 ms | 2 天 |
| **W5** | **TTFT**：草稿链前向与 target 预填充重叠（175.2 → ≤152.5 s） | 1.15x → ≤1.0x | −23 s | 1–2 天 |

**六、测量纪律（冻结）**

1. **判据 = `[OP]` GPU 时间**（R378：墙钟噪声 **±11 ms/轮**，≤5 ms 级效应全被淹没）；墙钟只作旁证，且必报 **轮成本 + AL 并列**（R375）。
2. **门值**：greedy sha 逐位 = `bcda0092bfbaaa9d30db21d4a30ccc1bc1e6e68c3ff3101f86402069a7450bb8`；任一臂破 = 不采用（无论多快）。
3. **同源 A/B**：一 build + env 门控（`GGML_*`），四库 md5 记录；ABBA ≥2 臂；对照臂差 <0.1% 才认。
4. **构建**：一律 `p4-build.sh` 清单门（`MANIFEST_DIFFS=0`）+ `BUILD_MANIFEST.txt`。
5. **预填充冻结**：W1/W2 若触公共组件，必须 env 门控 + 预填充门值重过（spec-off 151.7 s、门 `bcda0092…`）。
6. **默认关纪律**：未测量的改动**不得**进默认路径（R383 教训）。

**七、判负 ≠ 关闭（§1.6）**：E1/E6/E15 的结论是"**那种做法**不行"，不是"FA/GEMM 没救"：
- E15 的 −84% 真因待查（嫌疑：q=8 时 CTA 数 = 6 头 × 1 tile = **6 CTA/层/卡**，并行度灾难）⇒ **正解是 KV 切分**（W1 核心）。
- E6 的 split floor 无效 ≠ split 无效：那是 TILE 核的 floor，不是长 KV 并行度的解法。

### R385 ★★★ 用户观察（1cat 跑时 CPU 500–1000% 满载、我们不满）的代码级查证 + 方案三处修正（2026-09-26）

**用户观察**：BL1 运行时 nvtop 里不只 GPU 满载，**CPU 也 500–1000%**（= 5–10 核），"GPU 对应的 CPU 也在用"；我方 llama.cpp 无此现象。

**查证结论（勿再猜）**：
1. **不是 n-gram/lookup 辅助**：`vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:478-480` 由 `speculative_config.ngram_assist` 门控，**默认 False**；`:_lookup_enabled`（:528）随之 False、`_ngram_assist is None`（:473）。**正式服务 `--speculative-config` 未设该字段 ⇒ BL1 的 124.6 t/s 里 n-gram 辅助是关的**（R384 的"AL 靠 lookup 辅助"推断作废）。
2. **真实成分**：TP4 = **4 个 worker 进程 + engine core 进程**的常驻轮询/逐步记账（`np.logical_not(is_prefilling_np)`、CPU token 缓冲 `all_token_ids_cpu/sampled_token_ids_cpu`、lattice walk 的 `.item()` 逐步同步）+ NCCL/自定义 all-reduce 的 CPU proxy 线程。**作用 = 主机侧高并发把延迟全藏起来**（launch/sync/采样被吞，GPU 队列不空），不是"算得快"。
3. **对我方的意义**：我们是单主循环、主机侧并行度低 ⇒ 那 **10 ms host/gap 是真暴露的**（不是测量假象）⇒ **W4 优先级上调**。

**方案三处修正**：
| # | 修正 | 依据（1cat 自己的记录） |
|---|---|---|
| **W4-1（新，最便宜）** | **tail CUDA graphs：把 q1…q8 全部形状图化**（不只 q8） | `docs/design/sm70_dflash2_tail_graphs_20260911.md`：`VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS` **默认 True**，把 target-only B1 的 **q1–q7 与 q8 一起注册**；原文 **"the eager tail was the dominant round cost at the capacity"**；图谱清单 `max_context: 262144, query_rows: [2,3,4,5,6,7,8]` |
| **W4-2（新）** | **查我方 256K 下 q≠8 形状的注意力准入**：他们修过"long-attention loader 只准入 q8 到 132096"这类容量期 bug | 同上文档：容量点下 partial-verifier 与 q1 步"lose both full CUDA graphs and optimized attention dispatch" |
| **W1（细化）** | FA 长 KV 切分的**具体形态**：q1 算子 = **256 partitions × 1024 token**；grouped 候选 = **80 logical splits + K16 compensated QK + N32 online updates + 完整 FP32 workspace**；布局契约 **B1/H6/D256/page3296**（H6/D256 与我方每卡 6 Q 头 + D=256 同构） | 同上 + `sm70_dflash2_nvfp4_17ms.md`（"FULL target and draft CUDA Graphs"、Flash-V100、FP8 E5M2 target KV/FP16 draft KV） |
| **W0（不变）** | REJ 仍有效（我方实测 AL 3.93/4.09）；BL1 确实用 `draft_sample_method=probabilistic`（其 fork 对 dflash 的默认） | `vllm/engine/arg_utils.py:1825-1831`（mtp→greedy、其余→probabilistic 默认） |

**1cat 自有工程基线（重要参照，非 256K 口径）**：`sm70_dflash2_nvfp4_17ms.md` 冻结负载 = Qwen3.8-27B / 4×V100 / TP4 / batch-one DFlash2 / 七草稿 / selector K=16 / target top-k=20 / FP8 E5M2 target KV + FP16 draft KV / Flash-V100 / FULL target+draft CUDA Graphs ⇒ **已接受基线 512-token 请求 18.465–18.537 ms/轮、AL 4.686、251.60 token/s**；1024-token 18.587–18.603 ms。⇒ 其 256K（27.2 ms）比 512-token（18.5 ms）只多 ~9 ms，**那 9 ms 就是我方 FA 25.4 ms 的战场**。

### R386 ★★★ 重审发现：1cat decode 核并非"没救"，R366 判死有范围错误 —— **W1 的正确打法是"零拷贝分页复活"**（2026-09-26）

**史实（R351→R368 已做过的移植）**：`sm70-long/`（vendored 1cat `scalar-attention.cu` + `fp8_kv_utils.cuh`/`fused_mma.h`/`flash_v100_traits.cuh`/`paged_kv_utils.cuh` + `sm70-long-atenshim.h` + LICENSE）已编译通过（R358 零 torch）、已进构建（R361 隔离 OBJECT + `nm` 可见 `sm70_long_decode_f16`）、已接线 FA 分发（R362 `BEST_FATTN_KERNEL_SM70_LONG=501`，env `LLAMA_SM70_LONG_DECODE=1` 默认关）、**R366 判死、R368 封存**。

**R366/R368 判死的两条理由，各有致命范围错误**：
| 判死理由（原文） | 范围错误 | 证据 |
|---|---|---|
| "5K 下解码注意力总共 ~0.2–1 ms/步 ⇒ 无收益空间" | **在 5K 下测的**；我方目标是 256K | R377：256K 的 `FLASH_ATTN_EXT` = **25.4 ms/步 = 轮 37.6%**（头号单项之一）⇒ 目标场景下**空间极大** |
| "即使选中，我方集成每调用转全 KV 到 f16（O(kv) 流量）" | **这是适配层缺陷，不是核的缺陷** | 256K 下每次调用搬 ~9.1 GB/卡；1cat 生产是**分页 FP8 KV 直读零拷贝** ⇒ 该集成方式注定赢不了，也极可能是 IMA 根源 |

**⇒ W1 正确打法（内核级复活，非小修）**：
1. **KV 指针 + 页表直传**，删除 `sm70_long_stage_kv` 的 O(kv) 拷贝（`fattn-sm70-long.cu`）；1cat 的 `paged_kv_utils.cuh`（46 行纯 CUDA）原生支持分页。
2. **给 KV 编解码加 q8_0 变体**（现支持 FP16 / FP8 E4M3 / FP8 E5M2）：`fp8_kv_utils.cuh` 仅 **88 行纯 CUDA**，q8_0 = int8 + per-32 fp16 scale，照同一形状扩展 ⇒ **不动已测的 KV 配置**（避免"换 KV 格式=变相调参"）。备选：改用 BL1 同款 fp8_e5m2 KV（其核原生支持）。
3. **修 IMA**：4 处 ABI 修复已在手（头检查 24/4→GQA 6:1、scratch 固定 [80,8,6,256]、block_table/seq_lens 按 query token 逐行、每 KV 头平面偏移）；残余用 compute-sanitizer 定位。
4. **判据 = `[OP]` 的 FA 时间**（R378：非墙钟）。

**旁证（1cat 自己的记录）**：`docs/design/sm70_dflash2_tail_graphs_20260911.md` 明示其 256K 路径 = **B1/H6/D256/page3296 布局 + 256 partitions×1024 token（q1 标量核）** ⇒ 长 KV 是**分页 + 大分区**读的，与"零拷贝分页"结论一致；其 `sm70_dflash2_nvfp4_17ms.md` 冻结负载与我方同构（TP4/七草稿/FP8 E5M2 target KV/FP16 draft KV/Flash-V100/FULL graphs），512-token 轮 **18.465–18.537 ms、AL 4.686、251.60 t/s**，256K 为 **27.2 ms**（KV 增量仅 +9 ms）。

**另记（内核母矿，供 W2）**：`csrc/moe/marlin_moe_wna16/` = **SM70 Marlin**：`marlin_template.h`（2001 行，**0 torch 引用**）+ `kernel.h`（42 行，0 torch）+ `sm70_marlin_gemm.cuh`（2887 行）+ `sm70_marlin_u8_gemm.cu`（U8=int8，471 行）/`sm70_marlin_u8b128_gemm.cu` ⇒ **W2 = 移植 Marlin U8 并加 q8_0 布局 repack（group 32 保零量化误差）**。许可：`sm70_grouped_long/LICENSE` = BSD 3-Clause (c) 2025 D.Skryabin ✓。

### R387 ★★★ W1 施工定案：**给 1cat 长注意力核加 Q8_0 codec（零拷贝直读我方 KV）** —— 已落第一块代码；路线与依据（2026-09-26）

**一、为什么不能"照原样用"（三条硬约束，已实测/读码确认）**
| 约束 | 事实 |
|---|---|
| 布局不兼容 | 1cat 的 `_staged` launcher 契约 = `k_cache[n_pages, page_size, 1, 256]` f16，stride `(page_size*256, 256, page_size*n_pages, 1)` = **头主连续平面 + 页分组**；我方 KV = `[d][t][h]` **头交错**（nb1=1088=4×272, nb2=272）|
| 数据类型不兼容 | 该 launcher 硬取 `reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>())` ⇒ **只吃真 f16**；我方 q8_0 是"2B scale + 32×int8 = 34B/块"，**stride 变换解不出来** |
| staging 必死 | f16 KV 每卡每轮 = 17.2 GB（q8_0 的 1.9 倍）⇒ 地板 23 ms；staging 另加 18.2 GB 搬运 ⇒ 注定输。**R366 只在 5K 试过，所以从未暴露** |

**二、也不用"改成 FP8 KV"**：**llama.cpp/ggml 全树没有 FP8 类型**（`F8_E5M2`/`F8_E4M3` 零匹配）⇒ 走那条要新增端到端 ggml 量化类型（类型/量化核/`set_rows`/`--cache-type` 管线）+ **改被冻结的 79T 预填充 KV 读路径** ⇒ 改动面过大。

**三、定案路线（零拷贝 + 只动新核，env 门控默认关，预填充零影响）**
1. **加 `KV_CACHE_DTYPE_Q8_0` codec**（已落：`sm70-long/fp8_kv_utils.cuh` 加常量 + `q8_0_to_float` + `load_kv_cache_float_unscaled` 的 q8_0 分支；`load_kv_cache_half` 自动经 `load_kv_cache_float` 复用，无需改）。自包含、不引 ggml 依赖、**默认构建（未开 `GGML_CUDA_SM70_LONG`）零影响**。
2. **寻址能天然对上（关键洞察）**：index 契约定为 `index = row*256 + d`，行 = 一个 (token, kv head)。我方布局的字节偏移 = `(t*4 + h)*272` ⇒ 只要传 **stride_t = 1024（=4 头 × 256）、stride_h = 256**（以"虚拟 256 元素/行"为单位），则 `index>>8 = t*4 + h = 行号`、`index&255 = d` ⇒ codec 内按 `row*272 + (d>>5)*34` 取块，**零拷贝直读我方 q8_0 KV** ✓
3. **待办（下轮起）**：① 把该核里其余 dtype 分支放行 Q8_0（`dot_qk_cache<D,KV_DTYPE>`、`load_xqa_tc_kv_vector`、pair-load；注意 :810/:843 的 `static_assert` 只允许 E4M3/E5M2，需扩展）② 用 `LAUNCH_XQA_PARTITION(Q8_0)` 实例化 launcher 族 ③ raw entry 传我方 stride（去掉 `sm70_long_stage_kv` 的 O(kv) 拷贝）④ `-DGGML_CUDA_SM70_LONG=ON` 单目标编译验证（R361 先例）⑤ compute-sanitizer 定位残余 IMA ⑥ 判据 `[OP]` FA 时间 + 门值。

**四、预期**：KV 流量回到 9.1 GB/卡 ⇒ 地板 12.1 ms；该核按 ~700 GB/s 有效带宽 ⇒ **FA 25.4 → ~13–15 ms（−10~−12 ms/轮，≈ 轮的 −15%）**。这是当前最大的单项刀。

**五、铁的纪律提醒**：`_staged` 的残余 IMA（R368 修完 4 处 ABI 后仍崩）最可能源 = **按错误布局喂 KV**；改 codec + 我方 stride 后此因消失，若仍崩再用 compute-sanitizer 定位，**不得回退到 staging**（staging 在 256K 注定输）。

### R388 ★★★ W1 施工图细化：**目标核族 = grouped verify（80 splits × q≤8 × 6 heads）**；剩余活单已精确到行（2026-09-26）

**一、已确认（读码，逐条可复核）**
1. `dot_qk_cache<D, KV_DTYPE>`（`grouped-attention.cu:1003-1037`）的 **else 分支走通用标量路径** `load_kv_cache_float_unscaled<KV_DTYPE>` ⇒ **R387 加的 q8_0 codec 已自动接上，此函数无需改** ✓
2. KV 装载器（`:780-827`）按 dtype 分流：FP16 用 `uint4`、FP8 用 `uint64`+`fp8_*_vector_to_half8`，且 `:810-812` 有 `static_assert(只允许 FP16/E4M3/E5M2)` ⇒ **实例化 Q8_0 时需扩展该 assert 并加 Q8_0 分支**（8 元素 = 1 个 uint64 内的 8 int8？注意 q8_0 每 32 元素 34 字节，故 32-块内每 8 元素连续，需按 `chunk = (row*256+d)/8`、`row = chunk>>5`、块内 `(chunk&3)` 取 uint64 + 同块 scale）。
3. **寻址契约已验证成立**：传 `block_stride = page_size*1024`、`token_stride = 1024`、`head_stride = 256` ⇒ `physical_offset = t*1024 + h*256 + d = row*256 + d`（row = t*4+h = 我方 KV 行序）✓ 与 codec 契约逐字吻合。
4. **两个核族分清（关键）**：
   | 族 | 入口 | 形状 | 归属 |
   |---|---|---|---|
   | **grouped verify** | `flash_attention_grouped_verify_*`（`:4971` grid `dim3(1,80)`；combine 核 `:4978`） | **80 logical splits × q≤8 × 6 heads**；workspace `partial[80,8,6,256] f16` / `lse[80,8,6] f32` / `online_rescales[8,6,80] f32` | **我方 verify q=8（旧 raw entry 用的就是这族）** |
   | partition | `launch_flash_attention_decode_paged_xqa_tc_256_wide`（`:4257`，模板 `:4250-4256`）| 256 partitions | 我方 q=1 步 |
5. 与 1cat 文档互证：`sm70_dflash2_tail_graphs_20260911.md` 的"**80 logical splits、K16 compensated QK、N32 online updates、完整 FP32 workspace**"= grouped verify 族 ✓；"**256 partitions of 1024 tokens**"= partition 族 ✓。

**二、剩余活单（下轮机械执行）**
1. `fp8_kv_utils.cuh`：加 `q8_0_vector_to_half8(uint64 raw, __half scale)`（镜像 `fp8_e5m2_vector_to_half8`）。
2. `grouped-attention.cu` KV 装载器：`static_assert` 放行 Q8_0；加 Q8_0 的 uint64 分支（按上述 chunk/row/块内偏移取 8 int8 + 同块 half scale）。
3. **grouped verify 族**加 Q8_0 实例化（`:2047` 与 `:2623` 两个 kernel 定义的 KV_DTYPE 参数处 + 其调用点）；`FP8_PAIR_LOAD=false`、`E4M3_SHARED_LUT=false` 走通用路。
4. raw entry 改为 **Q8_0 + 我方 stride、零拷贝**（删 `sm70_long_stage_kv` O(kv) 拷贝；block_table 用恒等页表 page_size=256、n_pages=1024）。
5. 编译验证：**standalone `nvcc -std=c++17 -arch=sm_70 -DSM70_LONG_RAW`**（R358 先例，不动 cmake 缓存、不影响 p4-build 基线）⇒ 让编译器枚举剩余 static_assert/特化障碍。
6. 通过后：compute-sanitizer 定位残余 IMA → 门值 `bcda0092` → `[OP]` FA 时间 A/B（`LLAMA_SM70_LONG_DECODE=1`）。

**三、收益不变**：KV 流量 9.1 GB/卡 ⇒ 地板 12.1 ms；该核 ~700 GB/s ⇒ **FA 25.4 → ~13–15 ms（−10~−12 ms/轮）**。

### R389 ★ W1 首个可验证增量：**q8_0 codec 在 vendored 核内编译通过**（`NVCC_RC=0`，1.15 MB `.o`）（2026-09-26）

- **改动**：`fp8_kv_utils.cuh` 加 `KV_CACHE_DTYPE_Q8_0` + `q8_0_to_float` + loader 分支（R387）；`grouped-attention.cu` 的 `SM70_LONG_RAW` 区加冒烟核 `sm70_long_q8_0_probe`（强制实例化 codec，兼作"解量化值 vs CPU 参考"的校验核）。
- **验证方法**（R358 先例，不动 cmake 缓存、不碰 p4-build 基线）：`export PATH=/usr/local/cuda-12.4/bin:$PATH; nvcc -std=c++17 -arch=sm_70 -DSM70_LONG_RAW -I sm70-long -I . -O1 -c sm70-long/grouped-attention.cu` ⇒ **`NVCC_RC=0`**、`/tmp/sm70long-q8.o` = **1154328 字节**；输出仅 6 条 `#177-D declared but never referenced` 告警（含 1cat 自带的未用函数），**零 error** ✓
- **意义**：q8_0 读路径（块寻址 + scale 乘法）在真实核内可用；**默认构建（未开 `GGML_CUDA_SM70_LONG`）零影响**，故当前基线/门值不受牵连。
- **注意（工具路径）**：服务器 `nvcc` 不在 PATH，须用 `/usr/local/cuda-12.4/bin/nvcc`。
- **下一步**：按 R388 活单继续——装载器 `static_assert` 放行 Q8_0 + uint64 分支；grouped verify 族 Q8_0 实例化；raw entry 改 Q8_0 + 我方 stride 零拷贝；然后 sanitizer → 门值 → `[OP]` FA 时间 A/B。

### R390 ★★ W1 第二增量：**Q8_0 向量装载器分支编译通过**（K16/QK 装载路径已就位）（2026-09-26）

- **改动**：`grouped-attention.cu` 加 `q8_0_vector_to_half8(raw8, scale)`（沿用 `fp8_e5m2_vector_to_half8` 的 `uint4`=4×half2 位模式约定）；KV 装载器 `else` 链插入 **Q8_0 分支**（`chunk = physical_offset/8 + vec_col`；`row = chunk>>5`；`sub = chunk&31`；字节基址 `row*272 + (sub>>2)*34`；块内 `uint64` 取 8 个 int8 + 同块 half scale ⇒ `q8_0_vector_to_half8`）；并用 `decltype(&load_xqa_tc_kv_vector<4,false,Q8_0>)` **取地址强制实例化**（odr-use）。
- **块内安全性论证**：`physical_offset` 8 对齐 ⇒ 每个 8 元素 chunk 必落在单个 32-元素 q8_0 块内，**不会跨块**。
- **验证**：`nvcc -std=c++17 -arch=sm_70 -DSM70_LONG_RAW -O1 -c` ⇒ **`NVCC_RC=0`、零 error**（仅 `#177-D` 未引用告警）；`nm /tmp/sm70long-q8b.o` ⇒ **`T sm70_long_q8_0_probe` 在 `.text`（含 `__device_stub__`）**，180 个符号 ⇒ codec 真实例化（非静默跳过）✓
- **基线影响**：零（该目录不进默认 glob）。
- **下一步**：grouped verify 族的 Q8_0 实例化（`:2047`/`:2623` 两处 kernel 的 KV_DTYPE）+ raw entry 改 Q8_0/我方 stride 零拷贝 ⇒ 再 nvcc ⇒ sanitizer ⇒ 门值 ⇒ `[OP]` FA A/B。

### R391 ★★★ W1 坐标确定：**grouped verify 核族模板参数表 + Q8_0 实例化的确切插入点**（2026-09-26）

**核族真身（`grouped-attention.cu`）**
```cpp
// partial：QK+softmax 分区阶段
flash_attention_grouped_verify_e5m2_partial_kernel<
    8, false, /*页步长*/ 1648 | 3296, false, false, false,
    /*KV_DTYPE = 第 7 个模板参数*/ flash_v100::KV_CACHE_DTYPE_FP8_E4M3,
    false, float, true, true, false>
// full_q8：整 q8 一次过
flash_attention_grouped_verify_e4m3_full_q8_kernel<8, false, 3296, …, KV_DTYPE, …>
// combine：不含 dtype（在 partial 的 fp32 结果上归并）
flash_attention_grouped_verify_e5m2_combine_kernel<8, false, float, true>
```
**三条关键事实**
1. **KV_DTYPE = 第 7 个模板参数** ⇒ Q8_0 实例化 = 派发链加一条（名字里的 `e5m2`/`e4m3` 是历史标签：同一个核分别被 E4M3/E5M2 实例化，证明名字不绑 dtype）✓
2. **第 3 个参数 = KV 页步长编译期常量**（`1648`/`3296` = 文档里的 `page3296`），且绑定 kernel 的 smem tiling；运行期 `k.stride(0..2)` 另由 launcher 传入。
3. 派发点 `:4966-4983`，按 `k.size(1)`（KV 步长）选择 kernel + `paired` 变体。

**⇒ Q8_0 的实际植入方案（下轮执行）**：新增一条实例化，参数 = `KV_DTYPE_Q8_0` + **我方页步长 262144**（= 页 256 token × `token_stride` 1024 虚拟元素）+ 恒等页表（`page_size=256`、`n_pages=1024`），运行期传我方 strides（`block/token/head = 262144 / 1024 / 256`）。**若该核的 tiling 要求页步长落在 {1648,3296}，则退化为"给核加一个能接受任意页步长的实例"——仍是内核级改法，不做 staging。**

**数据背书**：`benchmark_sm70_dflash2_grouped_verify.py`、`test_sm70_flash_v100_dflash2_swa.py` 是这一族的微基准/测试，可作正确性对照。

### R392 ★★★ W1 执行规格完备：**不需要新页步长常量**（`PAGE=0` = 运行期 strides）；Q8_0 raw entry 的完整启动配方已确定（2026-09-26）

**关键简化**：真实实例化里第一个部分核的**第 3 模板参数取 `0`**，而 `1648`/`3296` 只是为 1cat 自家 paged 布局做的特化（`page_size` = 3296 token ⇒ 256K/3296 ≈ 80 页 ⇒ 与 `dim3(1, 80)` 的 80 对上）。⇒ **我方扁平 q8_0 布局直接用 `PAGE=0` + 我方运行期 strides 即可，无需新常量** ✓

**启动配方（照抄 `private_grouped_e4m3_fp32_paged` 的 :4985-5011，只换 dtype/stride 描述）**
```cpp
constexpr int kGroupedVerifyThreads = <见 :1820 附近>;
constexpr int kCompensatedSmemBytes =
    sizeof(GroupedVerifySmem) +
    kGroupedVerifyRows * kGroupedVerifyProbStride * sizeof(__half) +
    kGroupedVerifyBlockN  * kGroupedVerifyKVStride   * sizeof(__half);
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kCompensatedSmemBytes);
cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
kernel<<<dim3(1, 80), kGroupedVerifyThreads, kCompensatedSmemBytes, stream>>>(
    q_half, k_ptr, v_ptr, block_table_int, row_lengths_int,
    partial_f32, lse_f32,
    q_rows,                      // q.size(0) = 8（verify）
    block_table.size(1),         // max_num_blocks = n_pages
    k.size(1),                   // ★ 我方传 256（页内 token 数）⇒ 不命中 1648/3296 ⇒ 走 PAGE=0 运行期变体
    k.s0, k.s1, k.s2,            // ★ 262144 / 1024 / 256（虚拟元素单位）
    v.s0, v.s1, v.s2,
    scale * k_scale, v_scale, nullptr, 1, row_lengths_int);
flash_attention_grouped_verify_e5m2_combine_kernel<8, false, float, true>
    <<<dim3(q_rows, 6), kGroupedVerifyThreads, 0, stream>>>(
        partial_f32, lse_f32, row_lengths_int, out_half, q_rows, row_lengths_int);
```
- **Q8_0 实例化**：`..._e5m2_partial_kernel<8, false, 0, false, false, false, KV_CACHE_DTYPE_Q8_0, false, float, true, true, {true|false}>`，q==8 时用 `..._e4m3_full_q8_kernel<同上>`；`paired` 由 strides 的 16 对齐决定，两者都实例化最稳。
- **我方 strides 的推导**（已在 R387/R390 论证）：虚拟行 = 一个 (token, kv head) 的 256 元素；`token_stride = 4 头 × 256 = 1024`、`head_stride = 256` ⇒ `physical_offset = t*1024 + h*256 + d = row*256 + d`，`row = t*4 + h` 正是我方 `[d][t][h]` 缓存的物理行序；页取 256 token ⇒ `page_stride = 256*1024 = 262144`。
- **恒等页表**：`block_table[1][1024]` = 0..1023（int32），`row_lengths[1]` = kv 长度；`partial[80,8,6,256] f16`、`lse[80,8,6] f32`、`online_rescales` 不再需要（改用 partial/lse 组合路径）。
- **删除**：`fattn-sm70-long.cu` 的 `sm70_long_stage_kv`/`sm70_long_stage_q`/`sm70_long_scatter_out` 与 O(kv) 拷贝（零拷贝直指我方 KV）。
- **下一步**：写入 `sm70_long_decode_q8_0` raw entry（放在 `#if defined(SM70_LONG_RAW)` 块）⇒ `nvcc -DSM70_LONG_RAW` 编译 ⇒ 若 `PAGE=0` 与运行期 strides 不兼容（断言/错值），再按 R391 的退路给核加"任意页步长"实例 ⇒ sanitizer ⇒ 门值 ⇒ `[OP]` FA A/B。

### R393 ★★ W1 最后一处契约：**页表按 4-token 微块索引**（`BLOCK_SIZE==4`）⇒ 用数值实验定案，不猜（2026-09-26）

**族常量（已取得，`grouped-attention.cu:1838-1864`）**：`kGroupedVerifyQ8MaxQ=8`、`kGroupedVerifyRows=48`、`kGroupedVerifyBlockN=32`、`kGroupedVerifyKVStride=264`、`kGroupedVerifyProbStride=32`、`kGroupedVerifyQ8Splits=80`、`kGroupedVerifyThreads=512`
⇒ `kCompensatedSmemBytes = sizeof(GroupedVerifySmem) + 48*32*2 + 32*264*2 = sizeof(...) + 19968`（公式见 :4985-4988）。

**新发现的契约**：KV 装载器 `load_xqa_tc_kv_panel` 的 `BLOCK_SIZE == 4` 分支写死
```cpp
logical_block = token_offset >> 2;   block_offset = token_offset & 3;
```
⇒ **页表（`page_ids`/`block_table`）是按 4 个 token 一个微块索引的**，`BLOCK_SIZE` 是编译期模板参数，而 `k.size(1)`（= 运行期 `block_size`）另有用途（疑似"页内 token 数"，但 1cat 的 3296 与该猜法不自洽，故**不猜**）。

**两个候选假设（下轮用数值实验判决）**
| # | 假设 | 我方参数 |
|---|---|---|
| **H1** | 页表按 **4-token 微块**索引，`stride(0)` = 微块步长 | `block_stride = 4 × token_stride = 4096`、`token_stride = 1024`、`head_stride = 256`、`k.size(1) = 256`、`block_table[1][65536]` = 0..65535（256K/4） |
| **H2** | 页表按 **页内 token 数**索引，`stride(0)` = 页步长 | `block_stride = 262144`、`token_stride = 1024`、`head_stride = 256`、`k.size(1) = 256`、`block_table[1][1024]` = 0..1023（256K/256） |

两假设在**恒等映射**下都给出 `physical_offset = t*1024 + h*256 + d = row*256 + d`（`row = t*4+h` = 我方 `[d][t][h]` 物理行序）⇒ **两者对地址的最终结果一致**，差别只在页表长度与 `stride(0)` 的取值；**H2 若与微块索引冲突，取 H1**。

**判决方法（廉价、决定性强）**：写一个 standalone CUDA 小程序（`nvcc -DSM70_LONG_RAW` 可直接复用核），用**小规模合成 KV**（如 128 token、1 KV 头、pattern = 已知函数）跑该核，与 CPU 参考逐元素比对 → 同时验证（a）页表语义、（b）q8_0 解量化正确性、（c）无 IMA。**通过后再上实机**（避免 R368 那样"131 次干净后 IMA"的盲撞）。

**注意**：`kGroupedVerifyBlockN=32` ⇒ 每个 split 处理 32 token 一块，`kGroupedVerifyQ8Splits=80` ⇒ 80 个 split；256K/80 = 3276.8 token/split（≈1cat 的 3296 页，非整除 ⇒ 说明 split 边界由 `total_kv`/`split_tiles` 动态算，不依赖整除）。

### R394 ★★★ W1 数值实验 1 通过：**q8_0 codec 逐位精确**（`max_abs_diff=0`，768/768 元素全对）（2026-09-26）

- **方法（standalone，不碰实机）**：`nvcc -DSM70_LONG_RAW -arch=sm_70 -c grouped-attention.cu` 产出目标文件（含 `sm70_long_q8_0_probe` 符号）→ 用独立测试程序 `q8_0_codec_test.cu` **链接**该目标文件并调用 probe → 与 CPU 参考逐元素比对。合成数据：3 行 × 256 元素，每行 8 个 q8_0 块（`[half scale][32 int8]`），scale = 0.5/0.75/1.0，int8 填 `((r*7+d) % 255) - 127`（覆盖正负与全量程）。
- **结果**：`LINK_RC=0`；`Q8_0_CODEC_TEST rows=3 elems=768 mismatches=0 max_abs_diff=0 => PASS`；`RUN_RC=0`。
- **证明了**：① vendored 核的 q8_0 块寻址（`row*272 + (sub>>2)*34`）**逐位正确**；② 行契约 `index = row*256 + d` 成立；③ 该 standalone 数值测试骨架可复用（下轮扩到"页表语义"实验）。
- **资产**：`1cat-vllm-v100-study/p0-scripts/q8_0_codec_test.cu`（可重复执行的验证器）。
- **下一步（数值实验 2）**：给 `load_xqa_tc_kv_vector` 加一个 dump 型 probe，用小规模合成页表判定 **H1（按 4-token 微块索引）vs H2（按页内 token 数索引）**——这决定页表长度与 `stride(0)` 的取值。通过后再写 raw entry（参考 R392 启动配方 + R393 常量）。

### R395 ★★★★ W1 执行规格**完全确定**（模板表全取得）：H1/H2 问题消失——**页表粒度由我们定**（`PAGE_BLOCK_SIZE=0` ⇒ 运行期 `page_block_size`）（2026-09-26）

**一、H1/H2 之争由代码直接解决（无需实验）**：装载器
```cpp
if constexpr (BLOCK_SIZE == 4) {...} else if (BLOCK_SIZE == 16) {...} ... /* 特化分支 */
else { logical_block = token_offset / block_size;   // ← 运行期 block_size
       block_offset  = token_offset % block_size; }
const int physical_block = page_ids[logical_block];
```
⇒ **给非特化值（`PAGE_BLOCK_SIZE = 0`）即走 `else`，页表粒度 = 运行期 `page_block_size`（= launcher 传的 `k.size(1)`）** ⇒ 粒度**由我们选**，页表长度 = `ceil(total_kv / granularity)`。`CONTIGUOUS_HKV1_LAYOUT=false` ⇒ 不受 `static_assert` 的 BLOCK_SIZE 集合限制 ✓

**二、grouped verify partial 核的完整模板表（`:2071-2076`）**
```cpp
template <int MAX_QUERY_TOKENS, bool TWO_PASS, int PAGE_BLOCK_SIZE = 0,
          bool SINGLE_QUERY = false, bool CONTIGUOUS_HKV1_LAYOUT = false,
          bool STAGE_PARTITION_PAGE_IDS = false,
          int KV_DTYPE = KV_CACHE_DTYPE_FP8_E5M2,
          bool SPARSE_PAGE4 = false, typename PARTIAL_T = __half,
          bool ROW_SEQLENS = false, bool COMPENSATE_P = false, bool PAIR_E4M3 = false>
```
**★ 硬约束**：`COMPENSATE_P`（他们的 compensated-P 精度特性）被 `static_assert` **限定为 FP8 E4M3** ⇒ Q8_0 必须 `COMPENSATE_P=false`（**这是 Q8_0 相对 E4M3 的精度让步，需在数值验证时留意**）。

**三、我方 Q8_0 实例化（定稿）**
```cpp
flash_attention_grouped_verify_e5m2_partial_kernel<
    8,      // MAX_QUERY_TOKENS = 8（verify）
    false,  // TWO_PASS
    0,      // PAGE_BLOCK_SIZE = 0 ⇒ 运行期粒度
    false,  // SINGLE_QUERY
    false,  // CONTIGUOUS_HKV1_LAYOUT
    false,  // STAGE_PARTITION_PAGE_IDS
    flash_v100::KV_CACHE_DTYPE_Q8_0,
    false,  // SPARSE_PAGE4
    float,  // PARTIAL_T
    true,   // ROW_SEQLENS
    false,  // ★ COMPENSATE_P = false（E4M3 专用）
    paired> // PAIR_E4M3 ← 按 strides 的 16 对齐二选一，两者都实例化
```
**运行期参数**：`page_block_size = 256`（我方粒度）、`block_table[1][1024]` 恒等 0..1023（int32）、`row_lengths[1] = kv 长度`、strides（虚拟 256 元素行单位）= `block/token/head = 262144 / 1024 / 256`、grid `dim3(1, 80)`、threads 512、smem `sizeof(GroupedVerifySmem) + 48*32*2 + 32*264*2`；combine 核 `<8,false,float,true>` grid `dim3(q_rows,6)`。
**地址自证**：`physical_offset = (t/256)*262144 + (t%256)*1024 + h*256 + d = t*1024 + h*256 + d = row*256 + d`，`row = t*4+h` = 我方 `[d][t][h]` 物理行序 ✓（与 R390 的 codec 契约逐字一致）

**四、剩余活单（下轮一次写完）**
1. 写 `sm70_long_decode_q8_0` raw entry（`#if defined(SM70_LONG_RAW)` 块内）：建 at::Tensor 描述（q/k/v/out/partial/lse/block_table/row_lengths）→ 选 kernel → set smem attr ×2 → launch → combine。
2. 删 `fattn-sm70-long.cu` 的 `sm70_long_stage_q/kv/out`（**零拷贝**）。
3. `nvcc -DSM70_LONG_RAW` 编译 → 数值实验 2（小规模合成页表跑 partial 核 vs CPU 参考）→ sanitizer → 门值 → `[OP]` FA A/B。

### R397 ★★★★ W1 战略转折（证据驱动）：**1cat grouped-verify 族与 E4M3 咬死** ⇒ 正解是"抄他们的 KV 格式"（加 FP8 KV），不是"把 q8_0 塞进他们的核"（2026-09-26）

**证据（全文件只有 4 处 dtype 硬约束，两两咬合）**
| 行 | 断言 | 含义 |
|---|---|---|
| 837 | `KV_DTYPE ∈ {E4M3, E5M2}` | 向量装载器（**R390 已加 Q8_0 分支绕过** ✓） |
| **870** | 同款，在 **`load_xqa_tc_kv_panel`** | grouped-verify 核真正调用的装载器 ⇒ 仍拦 Q8_0 |
| **2507** | `static_assert(COMPENSATE_P && warps==16, "PV reuse is isolated to six-head compensated E4M3")` | partial 核**无条件段** |
| **3140** | 同款 | full_q8 核 |
配合 `:2092` 的反向断言（`COMPENSATE_P ⇒ KV_DTYPE==E4M3`）⇒ **两核都要求 `COMPENSATE_P=true` ⇒ 都要求 E4M3**。

**编译实证**：Q8_0 实例化（`COMPENSATE_P=false`）在 `:2507` 与 `:3140` 触发断言（两个 `paired` 分支都触发 ⇒ 该段**不在 `if constexpr(COMPENSATE_P)` 内**，无条件执行）；此前 5 处仅是**指针类型不匹配**（已修），说明**模板本身接受 Q8_0，是那段代码路径不接受**。

**两条路的诚实对比**
| | (A) 把 q8_0 塞进他们的核 | **(B) 抄他们的 KV 格式（加 FP8 KV）** |
|---|---|---|
| 要做的 | ① panel 装载器加 Q8_0 分支 + 放宽 870 ② **在 2507/3140 处凭空写出"非 compensated PV"尾段路径**（发明代码，非抄） | ① llama.cpp 加 FP8（E4M3/E5M2）KV 类型（含量化/写路径/`--cache-type` 管线）② **抄** 1cat 的 KV 写与读（其 `sm70_79t/prefill.cu` 与 decode 核**原生支持** E4M3/E5M2，`fp8_kv_utils.cuh` 已在手）③ 预填充 79T 的 KV 读同步换（env 门控 + 门值重过） |
| 性质 | **发明**（在别人的优化核里写新路径） | **抄**（整条格式与核一起搬） |
| 风险 | 高（精度机制被绕过，后果未知） | 中（面更广，但每步都是抄现成） |
| 与 BL1 口径 | 我们独有 q8_0 KV | **与 BL1 一致（fp8_e5m2）** ⇒ 同口径更硬 |
| 连带收益 | — | KV 流量 8.6 GB/卡（比 q8_0 的 9.1 更小）⇒ 地板 11.5 ms |

**⇒ 采用 (B)**（符合 §1.6④"能抄就抄"、不发明不将就）：**给 llama.cpp 加 FP8 KV，然后 1cat 的核零改动即用**（`COMPENSATE_P=true` 的 compensated 精度也一并拿到）。R387/R390 已完成的 q8_0 codec/loader 分支**保留为已证资产**（若将来还要 q8_0 KV 可用），但不再是 W1 的主路径。

**代价与纪律**：新增 ggml 类型属"公共组件"改动 ⇒ 必须 ① 默认行为不变（env/类型门控，默认仍 q8_0）② 预填充门值重过（spec-off 151.7 s、门 `bcda0092…`）③ 验收标准不变（Q8_0 权重、4×V100、256K）。

**下轮**：① 摸清 1cat 的 FP8 KV **量化/写路径**（`cache_kernels*.cu` 里的 fp8 量化 + `nvfp4_kv_cache_kernels.cu` 参考）② 评估 ggml 新增 FP8 KV 类型的最小面 ③ 再决定"新类型"还是"借用现有 8-bit 类型位"。

### R398 ★★★★ W1(B) 落地路线（比预想更简单）：**FP8 KV = uint8 + 一个 scale（非块量化）**；核零改动；**预填充冻结面可完全避开**（2026-09-26）

**一、格式本质（读 1cat `csrc/libtorch_stable/cache_kernels.cu`）**
- FP8 KV = **uint8 存储 + 一个 float scale**（`k_scale`/`v_scale`，形状 `[1]` 或 `[num_heads]`）—— **不是块量化**，比 q8_0（每 32 元素一个 half scale）**更简单**。他们服务里 `kv_cache_dtype: fp8_e5m2` 即此（`reshape_and_cache_flash`，`:346`）。
- 另有块量化变体（`quant_block_size` + `dst_scale[num_tokens, head_dim/qbs*4]`，`:1562+`，DS-MLA 用）—— **我们不需要**。
- ⚠️ 他们 **E5M2 不用 PTX cvt**：自写 `fp16_bits_to_e5m2_satfinite_rn`（`:250`，饱和 + RN）—— 抄这个才能逐位一致。

**二、已具备的抄件（许可 BSD-3）**
| 件 | 位置 | 用途 |
|---|---|---|
| `fp8::scaled_convert<OutT,InT,kv_dt>` | `quantization/w8a8/fp8/nvidia/quant_utils.cuh` | 量化/反量化原语 |
| `CopyWithScaleOp` | `cache_kernels.cu:238-244` | 逐元素转换 + scale |
| `fp16_bits_to_e5m2_satfinite_rn` | `cache_kernels.cu:250` | E5M2 逐位一致量化 |
| `reshape_and_cache_flash` | `cache_kernels.cu:346` | KV 写核（NHD/HND、按头 scale） |
| `cp_gather_and_upconvert_fp8_kv_cache` | `cache_kernels.cu:1265+` | **fp8→bf16 上转**（读侧兜底） |
| 注意力核的 E4M3/E5M2 路径 + `k_scale`/`v_scale` 入参 | 已在手（`sm70-long/`） | **核零改动** ✓ |

**三、★ 关键洞察：预填充冻结面可完全避开**
`O(kv)` 上转在 **decode 致命**（每步一次），但在 **prefill 只发生一次**：9.1 GB 上转 ≈ **12–24 ms**，对 **152,500 ms** 的预算是 **0.016%** ⇒ **不必动被冻结的 79T 预填充**（可先用"fp8→f16 一次性上转 + 现有 79T"；后续若有余力再抄他们的 fp8-aware prefill）。

**四、施工四步（下轮起）**
1. **ggml 新增 FP8 KV 类型**：8-bit、无块结构（1 字节/元素）+ 独立的 `k_scale`/`v_scale` 张量；管线（类型定义 / `ggml_row_size` / `--cache-type-k|v` 接受 / 量化-反量化桩）。默认 **不启用**（仍 q8_0）⇒ 门值/预填充零回归。
2. **KV 写核**：抄 `CopyWithScaleOp` + `fp16_bits_to_e5m2_satfinite_rn` 接到 llama.cpp 的 KV 写路径（含 scale 计算策略：先用单一 amax scale，后续可抄按头/按块）。
3. **decode 接线**：raw entry 传 `KV_DTYPE=E5M2` 或 `E4M3` + `k_scale/v_scale` + 我方 strides（R395 规格已定；`COMPENSATE_P=true` 此刻**合法**）。
4. **prefill**：先走"一次性上转"，跑门值（spec-off 151.7 s + `bcda0092`）确认零回归。

**五、收益不变且更硬**：KV 流量 8.6 GB/卡（vs q8_0 9.1）⇒ 地板 11.5 ms；FA 25.4 → **~12–14 ms**；且 **KV 格式与 BL1 一致** ⇒ 同口径验收更硬。

### R399 ★★★ FP8 KV 类型已入 ggml 且**行为中立**（`BUILD_RC=0`、门值 `bcda0092` 全等）；另查清一条运维事实（2026-09-26）

**改动（路线 B 第 1 步）**
- `ggml.h`：新增 `GGML_TYPE_F8_E4M3 = 43`、`GGML_TYPE_F8_E5M2 = 44`，`GGML_TYPE_COUNT` 43→45（**末尾追加，不复用已移除的旧号 31/32/33/36/37/38**，避免污染旧 GGUF）。
- `ggml.c` 的 `type_traits[]`：两行照 `GGML_TYPE_I8` 的样板——`blck_size = 1`、`type_size = 1`、`is_quantized = false`、**无 to_float/from_float**（flavor 与外部 scale 是消费者的契约：KV 写核 + 注意力核）。
- **验证**：`MANIFEST_DIFFS=0`、`BUILD_RC=0`、`ERROR_LINES=0`；`libggml-cuda.so` = `9c05fc80fd253f9fe49521b00ae9d752`；**门值两臂 `bcda0092…` 逐位一致** ⇒ 新增类型**行为中立**，且编译器未要求补其它表（traits 表是主表）。

**★ 运维事实（重要）**：`GGML_CUDA_SM70_LONG` **在构建树里是 ON**（R361 的 `cmake -D` 留在 CMakeCache）⇒ **vendored `sm70-long/` 在默认构建内**，对它任何改动都必须能过 p4-build。本次即因此暴露：R396 的 Q8_0 raw entry（`COMPENSATE_P=false`）触 `:2507` 断言**打挂了强制构建** ⇒ 已用 `#if defined(SM70_LONG_Q8_0_ENTRY)` 整体置于**未定义宏之下**（保代码不编译，等路线 B 落地后再决定去留）。

**下一步**：路线 B 第 2 步——抄 `CopyWithScaleOp` + `fp16_bits_to_e5m2_satfinite_rn` 接入 KV **写**路径（f32→fp8 + scale），仍默认关；然后第 3 步 decode 接线（`KV_DTYPE=FP8_E5M2` + `k_scale` + 我方 strides，此刻 `COMPENSATE_P=true` 合法）。

### R400 ★★★ 路线 B 第 2 步定位：KV 写路径 = `set_rows`（需加 FP8 分支）；★ 判定 **FP8 KV 必须是 E4M3**（2026-09-26）

**一、写路径（读码确认）**
| 环节 | 事实 |
|---|---|
| llama 侧 | `ggml_set_rows(ctx, k, k_cur, k_idxs)`（`llama-kv-cache.cpp:1354/1389/1410`），`k_cur` 为 **f32** |
| CUDA 分派 | `ggml-cuda/set-rows.cu` 的 `set_rows_cuda<float, int32_t>`（止于 `:322`）= **白名单**（F32/F16/BF16/Q8_0/IQ4_NL）+ `GGML_ABORT` ⇒ **FP8 需新增分支** |
| 量化子路径样板 | `set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>`（`:300-309`）——**每块现算 scale 并写进块头**，与 FP8「无块头 + 外部 scale」语义**不同** |
| `is_quantized` 的影响 | 只门控"注意力旋转（Hadamard）"（`llama-kv-cache.cpp:324/337`）⇒ FP8 取 `false` 无害，且我们不想要该旋转 |

**二、★ 决定性判定：FP8 KV 必须用 E4M3（不是 E5M2）**
grouped-verify 核尾段**无条件**要求 `COMPENSATE_P`（`:2507`/`:3140`），而 `COMPENSATE_P` 被 `:2092` 断言限定为 **`KV_DTYPE == FP8_E4M3`** ⇒ **E5M2 同样无法进入该核**。虽然 BL1 的 service 配置写的是 `fp8_e5m2`，但**我们的核只认 E4M3** ⇒ 取 E4M3。
（附带收益：E4M3 尾数 3 位 ≈12% 相对误差，优于 E5M2 的 2 位 ≈25%；且 compensated-P 精度机制只在 E4M3 上存在——**我们反而拿到比 BL1 更好的 KV 精度**。）

**三、scale 形态**：1cat = **外部标定**的 per-tensor / per-head scale，作为 launch 入参传给核（`k_scale`/`v_scale`，形状 `[1]` 或 `[num_heads]`）；其写核 `CopyWithScaleOp` 只做 `src*scale → fp8`。我们照抄该形态，**首版取 scale=1.0**（E4M3 范围 ±448，KV 量级通常远低于此；若出现饱和再引入标定）。

**四、下一步（第 2 步实现）**
1. `set-rows.cu` 给 `set_rows_cuda<float,int32_t>` 加 **F8_E4M3 分支**：写 1 字节/元素（`uint8_t`），转换抄 1cat 的 fp8 原语（E4M3 用 `fp8::scaled_convert<uint8_t, float, kFp8E4M3>`；E5M2 若将来要用，必须连同 `fp16_bits_to_e5m2_satfinite_rn` 一起抄以保逐位一致）。
2. 默认**不启用**（只有显式 `--cache-type-k/v f8_e4m3` 才走到）⇒ 门值/预填充零回归。
3. 然后第 3 步 decode 接线（`KV_DTYPE=FP8_E4M3` + `COMPENSATE_P=true` + 我方 strides）。

### R401 ★★★ 路线 B 第 2 步完成：**FP8 KV 写路径已入 `set_rows`**（行为中立，门值全等）（2026-09-26）

**实现（零改动复用基核的巧法）**
- 新增 `struct fp8_e4m3_t { uint8_t v; fp8_e4m3_t(float x) : v(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)) {} }`（`set-rows.cu`）。
  ⇒ 基核 `k_set_rows<src_t,idx_t,dst_t>` 的元素步长来自 `sizeof(dst_t)` 与写入语句 `dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00])` ⇒ **1 字节结构 + float 转换构造即可直接实例化，4D 索引逻辑零改动** ✓
- `set_rows_cuda<float,int32_t>`（张量级特化）新增 `else if (dst->type == GGML_TYPE_F8_E4M3)` 分支，调用形式**照抄同函数既有分支**（靠模板推导，不写显式实参）。
- 踩坑记录：① `__nv_cvt_float_to_fp8` 返回 `__nv_fp8_storage_t`（= `unsigned char` 别名）**没有 `.__x`**；② 构造函数**必须非 explicit**（`convert.cuh:64` 的 `ggml_cuda_cast` 走隐式转换）；③ 显式模板实参反而导致 "no instance matches" ⇒ 用推导。

**验证**：`MANIFEST_DIFFS=0`、`BUILD_RC=0`、`ERROR_LINES=0`；`libggml-cuda.so` = `2cf1588a0027b5f2380da2b7e378b059`；**门值两臂 `bcda0092…` 逐位一致** ⇒ 行为中立（尚无 cache type 会选到 f8_e4m3）。

**scale 现状**：写路径只做 `f32 → E4M3`（饱和 RN），**scale 由调用方给**——与 1cat 的 `CopyWithScaleOp`（外部标定 scale）契约一致，首版取 1.0。

**下一步（第 3 步）**：① 放行 `--cache-type-k/v f8_e4m3`（llama.cpp 侧类型校验 + 名称解析）② decode 接线：raw entry 传 `KV_DTYPE=FP8_E4M3` + `COMPENSATE_P=true` + 我方 strides + `k_scale`（R395 规格已定稿）③ 数值实验 2（合成数据 vs CPU 参考）④ 上机 A/B（判据 `[OP]` FA 时间）。

### R402 ★★ 路线 B 第 3 步侦察：**`--cache-type f8_e4m3` 无需改校验**；只剩三处接线（2026-09-26）

**一、类型校验链路（读 `llama-context.cpp:3840-3876`，结论：零改动）**
- 校验是**通用**的，没有类型白名单：① 仅 MLA/DS4 要求 `type_k == type_v`（不涉及我们）② `ggml_is_quantized(type_v)` ⇒ 强制开 FA ③ `ggml_is_quantized(type_k)` ⇒ 要求 `blck_size` 整除 `n_embd_head_k`。
- 我方 FP8 类型 `is_quantized = false`（R399 设定）⇒ **②③ 两条门都跳过**（且 `blck_size = 1` 本就整除一切）⇒ **`f8_e4m3` 可直接被接受** ✓
- 名称解析走 `type_traits[]`（已含 `"f8_e4m3"`）⇒ `--cache-type-k/v f8_e4m3` 能被解析 ✓

**二、剩余三处接线（第 3 步的全部工作）**
| # | 位置 | 内容 |
|---|---|---|
| 1 | `sm70-long/grouped-attention.cu`（`SM70_LONG_RAW` 块） | 新增 `sm70_long_decode_fp8` raw entry：与 R396 的 q8_0 版同构，但 `KV_DTYPE = KV_CACHE_DTYPE_FP8_E4M3`、**`COMPENSATE_P = true`**（此刻合法）、并接收 `k_scale`/`v_scale`；KV 张量描述用 `at::kByte`（1 字节/元素）+ 我方 strides（`262144/1024/256`、页 256 token、恒等页表 1024 项） |
| 2 | `fattn-sm70-long.cu`（我方适配层） | 放开 K/V 类型检查（现为"neither f16 nor q8_0"），加 fp8 分支；**删掉 `sm70_long_stage_kv` 的 O(kv) 拷贝**（零拷贝直指我方 KV；这是 W1 收益的关键） |
| 3 | `fattn.cu` 分发 | `K->type == GGML_TYPE_F8_E4M3` 时路由到 `BEST_FATTN_KERNEL_SM70_LONG`（env 门控仍默认关） |

**三、约束复述（防遗忘）**：FP8 KV **必须 E4M3**（核尾段要求 `COMPENSATE_P`，而其断言限定 E4M3）；E5M2 进不去该核。scale 由调用方给（首版 1.0）。

**下一步**：写 1→2→3，然后 `nvcc -DSM70_LONG_RAW` 编译 → 数值实验 2（合成数据 vs CPU 参考）→ 上机 A/B。

### R403 ★★★★ W1 核心构件编译通过：**FP8 E4M3 零拷贝入口 `sm70_long_decode_fp8`**（`NVCC_RC=0`，符号在 `.text`）（2026-09-26）

**实现**：`sm70-long/grouped-attention.cu` 的 `SM70_LONG_RAW` 块新增 `sm70_long_decode_fp8`——与 R396 的 q8_0 版同构，但：
- `KV_DTYPE = KV_CACHE_DTYPE_FP8_E4M3` + **`COMPENSATE_P = true`**（`TWO_PASS=false`、`ROW_SEQLENS=true`、`PARTIAL_T=float` 全部满足 `:2092` 与 `:2507` 两处断言的合取）
- KV 张量 = `at::kByte`，**strides 就是张量的自然字节步长**：`token = n_kv_heads*256 = 1024`、`head = 256`、`page = page_tokens*token = 262144`（全 8/16 对齐 ⇒ `paired = true`，走向量装载路径）
- 传 `k_scale`/`v_scale`（首版 1.0），qk_scale 按 1cat 的折叠方式传 `softmax_scale * k_scale`
- 地址自证：`physical_offset = (t/256)*262144 + (t%256)*1024 + h*256 + d = t*1024 + h*256 + d` = 我方 `[d][t][h]` 扁平布局的字节偏移 ✓ **零拷贝**

**验证**：`nvcc -std=c++17 -arch=sm_70 -DSM70_LONG_RAW -I sm70-long -I . -O1 -c` ⇒ **`NVCC_RC=0`**、零 error、`/tmp/sm70fp8.o` = 1,159,624 字节；`nm` ⇒ **`T sm70_long_decode_fp8` 与 `T sm70_long_decode_f16` 并列在 `.text`**（共 181 符号）✓

**意义**：这是**该核族首次以生产形态（E4M3 + compensated）在我方树内编译成功**——此前所有尝试都被 `COMPENSATE_P ⇒ E4M3` 的断言挡住（Q8_0 路线即因此改道）。**路线 B 的核侧已通**。

**剩余**（第 3 步余下）：① `fattn-sm70-long.cu` 适配层加 fp8 分支 + **删 `sm70_long_stage_kv` 的 O(kv) 拷贝** ② `fattn.cu` 分发路由 `GGML_TYPE_F8_E4M3` ③ 数值实验 2（合成数据 vs CPU 参考）④ 上机 A/B（判据 `[OP]` FA 时间）。

### R404 ★★ 适配层 FP8 分支的精确设计（下轮一次写完，避免半截改动）（2026-09-26）

**一、现状（读 `fattn-sm70-long.cu`）**
- `sm70_long_ws`（:122-138）持久工作区：`k/v/q/o/bt/sl/act/mxl/exs/ors/part` + 容量 + `stats_ok`（R368 立：图捕获内禁止 per-call `cudaMalloc`）。
- `supported()`（:145-198）的类型白名单在 `:182-185`：**只收 `F16` 或 `Q8_0`**。
- `decode()`（:206+）现有实现：staging（`sm70_long_stage_q/kv` + `scatter_out`）后调 `sm70_long_decode_f16`。

**二、设计结论（重要，缩小改动面）**
| 成本 | 量级 | 处置 |
|---|---|---|
| **KV staging**（`sm70_long_stage_kv`） | **O(kv)**（256K = 9.1 GB/卡/轮） | **必须删**，FP8 路径直接传 `K->data`/`V->data`（零拷贝） |
| Q staging（`[t][hq][d] → [q][h][d]` f16） | O(q×h×D) = 8×6×256 | **保留**（微小，且核的布局契约需要） |
| 输出 scatter（核写 `[q][h][d]` f16 → 我方 `dst [D][n_q][hq]` f32） | 同上，微小 | **保留** |

**三、FP8 分支需要的工作区缓冲**（复用 `sm70_long_ws`）
- `bt`：`int32[n_pages]`，**恒等页表**（`bt[i] = i`），`n_pages = ceil(kv_len / 256)`；kv 长度不变时**只填一次**（用 `bt_cap` 缓存判别）。
- `sl`：`int32[1]` = `kv_len`（`row_lengths`）。
- `part`：`f32[80*8*6*256]` = 3.93 MB（`PARTIAL_T=float`，与 1cat 的 E4M3 实例化一致）。
- `lse`：`f32[80*8*6]`。
- `o`：`half[n_q_pad*hq*256]`。
- **均已在 `sm70_long_ws` 中有对应字段**（`bt/sl/part/o`），只需按 fp8 语义调整容量与填充方式；`mxl/exs/ors/act` 在 FP8 路径**不再需要**（改走 partial/lse 组合）。

**四、插入点**
1. `supported()` `:182`：白名单加 `K->type == GGML_TYPE_F8_E4M3`（**必须与 decode 分支同时落地**，否则会把 FP8 引到只支持 f16/q8_0 的 staging 路径）。
2. `decode()` 开头加早分支：`if (K->type == GGML_TYPE_F8_E4M3) { …零拷贝路径… return; }`。
3. `alloc_size()`：按 fp8 路径的 workspace 需求（part 3.93 MB 为主）给出。

**五、调用形态（与 R403 的 raw entry 对齐）**
```cpp
sm70_long_decode_fp8(q_staged_f16, K->data, V->data, ws.o,
                     ws.bt, ws.sl, ws.part, ws.lse,
                     n_q, n_kv_heads, n_q_heads_per_kv,
                     /*page_tokens*/ 256, n_pages,
                     /*k_scale*/ 1.0f, /*v_scale*/ 1.0f, scale, stream);
// 然后 scatter ws.o -> dst（沿用 sm70_long_scatter_out）
```

**六、为什么本轮不动手**：单改 `supported()` 会把 FP8 路由到不支持的 staging 路径（危险）；而完整分支涉及 workspace 容量、页表缓存、早分支与 scatter 四处联动，需连续性——**留作下一轮一次写完并 nvcc 验证**，避免半截改动污染构建。

### R405 ★★ 适配层结构读通 + 一处必须用数值实验定案的语义（下轮的入口）（2026-09-26）

**一、`decode()` 结构（`fattn-sm70-long.cu:206-392`）**
| 段 | 内容 |
|---|---|
| :229-241 | 维度与 `n_pages = ceil(kv_len/256)`、`n_parts`（R367 已修过 512 上限截断 ⇒ 256K 需 1024 页） |
| :246-267 | 持久工作区分配（按 `ws.kv_cap`/`ws.q_cap` 缓存，容量不足才 `cudaFree/cudaMalloc`）——**图捕获内禁止 per-call malloc（R368）** |
| :268-273 | 指针绑定 + memset |
| **:282-285** | **已有的 `k_direct` 快路径**：条件 `!k_q8 && hkv == 1 && K->nb[1] == D*sizeof(half)` ⇒ 直接传 `K->data` **免 staging** |
| :287-300 | 其余情况走 `sm70_long_stage_kv`（**O(kv)，就是它**） |
| :301-309 | Q staging（按 KV 组，`[n_q][gqa][256]` 平面，O(q×h×D) 微小） |

**★ 发现**：我方 KV 是 `hkv = 4`、`nb[1] = 1024`（FP8）/`1088`（q8_0）⇒ **恰好不满足 `k_direct` 的 `hkv == 1` 条件** ⇒ 必然走 staging ⇒ 9.1 GB/卡/轮的搬运。**这就是 W1 收益的全部来路**（也解释了 R366 在 5K 下"无收益空间"的观测：5K 时 O(kv) 只有 0.2 ms）。

**二、已定的部分**（FP8 分支可直接用）
- 工作区复用：`bt`（恒等页表 int32[n_pages]）、`sl`（int32[1] = kv_len）、`part`（f32[80·8·6·256]）、`lse`（f32[80·8·6]）、`q`/`o`（f16）；`mxl/exs/ors/act` 在 FP8 路径不再需要。
- Q staging 与输出 scatter **保留**（O(q×h×D)，微小；核的布局契约需要）。

**三、★ 必须用数值实验定案的语义（不要猜）**：`head_group` / combine 网格的索引语义。
- 我 R403 的 raw entry 把 `q` 声明为 `[n_q_pad, n_q_heads_per_kv, 256]` 单组布局，并让 `grid.x = n_kv_heads`（4）；但适配层的 `sm70_long_stage_q` **把各组写成拼接平面**（`Q_f16 + j*total_g`，平面步长 `total_g = D*n_q_pad*gqa`）。
- 二者是否等价，取决于核内 `head_group`（`blockIdx.x`）到底如何偏移头轴、以及 combine 核网格该取 `(q_rows, 6)` 还是 `(q_rows, hq)`——**这只能由数值实验回答**。
- **解决工具（下一轮第一件事）**：扩展 R394 的 standalone 数值测试→**数值实验 2**：小规模合成（如 `hkv=4`、`hq=24`、`kv=512`、`q=8`）跑 partial+combine，与 CPU 参考逐元素比对 ⇒ 一次同时定案（a）q/头索引语义、（b）页表语义、（c）q8/fp8 解量化、（d）有无 IMA。

**四、纪律**：不做半截改动——`supported()` 白名单与 decode 早分支必须同时落地，且**先过数值实验 2 再上机**（避免 R368 的"131 次干净调用后 IMA"）。

### R406 ★★★★ 数值实验 2 首次运行：**FP8 路径端到端执行、零 IMA**（里程碑）+ 索引语义定案（2026-09-26）

**运行结果**（standalone，机器空闲，未触碰服务）
```
FP8_VERIFY_TEST q=8 hq=24 hkv=4 kv=512 page=256 np=2
  worst_rel=0.79618 worst_abs=0.0039062 at (q=0, head=18, d=0) => FAIL
```

**① 里程碑：不崩。** `/tmp/sm70fp8.o` 链接成功、`sm70_long_decode_fp8` 端到端跑完、`cudaDeviceSynchronize` 无错 ⇒ **此前封存（R368"131 次干净调用后 IMA"）的故障不属于 fp8 路线**；我参数化的 fp8 + 扁平 `[d][t][h]` 缓存 + 恒等页表 + 4 KV 头组合**不触发 IMA**。**这是该核族首次在真实数据流上执行成功。**

**② 索引语义定案（R405 遗留问题的答案）**：核**每个 KV 头启动一次**（`grid.x = 1`，处理该头的 6 个 Q 头）——与旧适配层注释"launched once per KV head with that head's 6 Q heads"及 1cat 的 `dim3(1, 80)` 完全一致。

**③ 定位到我的入口 bug**：我一次启动就 `grid.x = n_kv_heads`（4）、`q/out` 只声明一组 6 头、combine 网格也只覆盖 6 头 ⇒ **24 头里只写了 6 头，其余是未初始化内存** ⇒ 失败点落在 `head=18`（第 4 组）**完全吻合**。

**④ 修正方案（下轮实施）**：入口内部**循环 4 次**（每 KV 头一次）：`grid.x = 1`、K/V 基址 **+ j×256 字节**（我方 `[d][t][h]` 的 head 步长）、q/out 用**按组拼接平面**（`[j][n_q_pad][gqa][D]`，正是旧 `sm70_long_stage_q` 的产出布局）、combine 网格 `dim3(q_rows, gqa)` 并对 `out` 加组偏移。
**地址自证（零拷贝仍成立）**：基址 `j×256` + `page×262144 + tok_in_page×1024 + 0×256 + d` = `j×256 + t×1024 + d` = 我方扁平布局第 j 头第 t 位置第 d 维的**字节偏移** ✓

**⑤ 测试侧同步**：`fp8_verify_test.cu` 的 Q 需按 `[j][t][gqa][d]` 摆放、输出按同布局比较（镜像适配层将要做的 staging）。

### R407 ★★★★ 数值实验 2 收敛：**partial→combine 耦合没对上**（转储给出决定性证据：每组只有第 0 头被写，其余 5 头恰为 0）（2026-09-26）

**诊断过程（三次迭代，逐次缩小）**
| 迭代 | 结果 | 排除/确立 |
|---|---|---|
| 1 | `worst_rel=0.796, abs=2^-8`，最差点 `(q=0,h=18)` | 头索引（已修：每 KV 头一次启动） |
| 2 | 加分级判据后 `=> PASS` 但 **`significant=0`** | **我的测试设计有缺陷**：V 取零均值格点 ⇒ 输出相消趋零、相对误差退化 ⇒ 判据形同虚设 |
| 3 | V 改强正偏（E4M3 精确：`0.25/0.5/0.75/1/1.5/2/-1`）⇒ 输出 O(1) | `43008/49152 = 7/8` 错，`worst_abs=0.7168` ⇒ **真问题**（非精度） |
| 4 | 转储 `(j=0, h6=0..5, t=0..1, d=0)` | **★ 结论**：`h6=0` 全对、`h6=1..5` **恰好 0**（非置换、非错值）；q 行（t=0/1）均正确 |

**★ 结论**：`partial[80,8,6,256]` + `lse[80,8,6]` 与 **combine 核的耦合没对上**——combine 只覆盖 6 个头中的 1 个，其余位置从未写入（输出缓冲区我未清零 ⇒ 读出 0）。
**已排除**：q 平面布局（`[t][h][d]` vs `[h][t][d]` 两种都试过，q 行都正确）、头索引（R407 第 1 轮已修）、精度/解量化（h6=0 逐位精确）。

**剩余嫌疑（下一轮读 combine 核体定案，`grouped-attention.cu:3283+` 约 100 行）**
1. combine 的 `gridDim.y` 语义：1cat 的调用是 `dim3(q.size(0), 6)`——但那是"每次启动一个 KV 头、partial 只含该头 6 个 Q 头"的语境；我每头各启动一次也传 `(q_rows, 6)`，语义可能重复或错位。
2. combine 的输出寻址：是 `[h][t][d]`（头主，plane 步长 `q_rows*D`）还是 `[t][h][d]`（token 主，头步长 `D`）——我按头主摆放但只得到 1/6 的写入，说明**输出地址的头步长不匹配**。
3. `partial`/`lse` 的头轴步长：`po_st = {8*6*256, 6*256, 256, 1}`（我按 `[split][q][h][d]`）是否与核内一致。

**纪律**：不动半截代码；下一轮先读 combine 核体把这三条一次定案，再改，然后重跑本测试（已具备转储与分级判据）。

### R408 ★★★★ combine 核体定案：三处硬契约（**推翻 R407 的"每头一次启动"判断**）（2026-09-26）

**核体（`grouped-attention.cu:3283-3357`）**
```cpp
token_idx = blockIdx.x;  head_idx = blockIdx.y;  request_idx = blockIdx.z;
if (token_idx >= query_len || head_idx >= kGroupedVerifyHeads) return;   // ★ 全头数
partial_lse += request * kSplits * MAX_Q * kGroupedVerifyHeads;
lse_idx = (split * MAX_Q + token) * kGroupedVerifyHeads + head_idx;
split_lse[t] = partial_lse[2*lse_idx];        // ★ 成对：(lse, sum)
split_sum[t] = partial_lse[2*lse_idx + 1];
out[(token * kGroupedVerifyHeads + head_idx) * D + d] = ...;             // ★ out = [t][h][d]
```

**三处我搞错的契约**
| # | 契约 | 我做的 | 正确 |
|---|---|---|---|
| 1 | 头轴 = **`kGroupedVerifyHeads`（全头数 24）** | 6（每组头数） | **24** |
| 2 | `lse` 每 (split,token,head) = **2 个 float**（`lse` 与 `sum` 成对，且 `ROW_SEQLENS` 下走 `split_sum` 分支） | 1 个 | **×2** |
| 3 | `out` = **`[t][h][d]`，头步长 `D`**（token 主） | 改成 `[h][t][d]` | **`[t][h][d]`** |

**★ 结论：partial 是全尺寸共享工作区 `[80][8][24][256]` f16**（每个 KV 头的启动只填自己那 6 个头），combine 一次性读全部 24 头并把权重归并到 `out[t][h][d]`。
⇒ **我最初的"单次启动 grid = dim3(n_kv_heads, 80)"形式本来就是对的**；R407 改成"每头一次启动"是被错误现象误导——真实原因是 **partial/lse 尺寸不对（6 宽 vs 24 宽、lse 缺一列）造成写越界/踩踏**，才表现为"只有第 0 头有值"。

**下轮修正（四条）**
1. 入口回到**单次启动**：`grid = dim3(n_kv_heads, 80)`，q 传**整张** `[n_q_pad][hq][D]`（头步长 `D`），K/V 传整张（loader 按 `head_group` 取 KV 头）。
2. `partial` = `[80][8][kGroupedVerifyHeads=24][256]` f16。
3. `lse` = `[80][8][24][2]` f32（**成对**）。
4. combine：`grid = dim3(q_rows, 24)`，`out` = `[t][h][d]`。
5. 测试同步：q/out 回 `[t][h][d]`，partial/lse 尺寸按上式（**并改用 `cudaMemset` 清零输出**，让"未写入"与"写零"可区分）。

**下轮先做的一件小事**：grep 确认 `kGroupedVerifyHeads` 的确切数值（推断 24 = 4 KV 头 × 6），避免又一次猜。

### R323 ★★★ **T1-C 第一硬里程碑：79T 引擎编译通过（BUILD_RC=0）——错误收敛 101→21→15→3→1→0，抄袭链四世同堂闭环**（2026-09-24）

- **装配终账**：prefill.cu 6893 行 + CUTLASS 2.11 全集（OBJECT target 隔离 = 双 cutlass drift 实证后正解）+ `_79t_defs` 圣经 36 宏**移至文件首**（晚于 :19 cublas 门 = 前一轮假绿根因）+ `cutlass::GemmSoftmax` = **CUTLASS example 35 头**（抄袭链闭环：FA 抄 example → 1cat 抄 FA → 我方抄 1cat，vintage 与 2.11 天然同代）+ swizzle `get_tile_offset` static→成员（v2.11 API vintage 差）。
- **条件链关键事实**：`:389 PREFIX_QK_RAW_FIXED_SHIFT` 门 = QKGemm typedef **恒编**（与 CUBLAS_RAW 无关）⟹ GemmSoftmax 必备；`:19` cublas include 门早于源内 defs ⟹ 原构建靠 -D 注入。
- **下一步**：ggml 接线（入口 + 准入门 q%256/kv%32/Hq6 组 + ub 段喂入 + state_max/sum 续算 + 尾段回落）→ 门值+PPL 双关 → FA/TTFT A/B。**生死门：TTFT ≤ 152.5 s**（当前 169.5）。
- **R322 编译集成灾训五条（今日最贵一课 + 恢复实录）**：① Compress-Archive 反斜杠条目老坑（tar 为准）② tar 跟随 symlink 乱链 ③ **⚠️ Copy-Item 合并目录 = untracked 污染、git checkout 不清 untracked、git clean 才是净**（"checkout=净"假象；已致 cute 裁剪集混装 100 错、git clean+净版重传救活 BUILD_RC=0）④ pwsh=PS5.1（Set-Content -NoNewline 不支持）⑤ CMake 源属性静默失效 → 已验证三件套 = **#define 直写源 + 平铺 + target_include_directories（:124 同款 = 有效机制）**。**第二波组装清单**：defs 直写 prefill.cu 顶部（edit 工具）→ 平铺改名 fattn79t-prefill.cu（glob 收）→ include 行改 "cutlass/..." → **cutlass 2.11 留 fattn-79t/include（不碰 sm70-vendor）** + target_include_directories 加它。
- **R321b 前缀引擎节拍解剖 + 尾段数学破解（T1-C 终形清单）**：per-block 节拍 = `prefix_qk（CublasQKLauncher，转置 score [kBlockN,rows]）→ stable_max 分块归约 → prefix_pv（FP32 输出）→ stable_merge_prefix（24K 在线合并 = 守卫④）` + 尾段双流重叠（concurrent_tail_scores）；四守卫全内建编排。**尾段泛化破解**：kTailTileTokens=256（Q8192 尺寸族）⟹ 2048/256=8 tiles 整除、kTailTasks=36（三角公式 Q-线性）⟹ assert 改 %256==0 即可 = **泛化温和**（"重写尾部数学"判再修正）。T1-C 清单：前缀编排 ~200 行 + stable 三核（现成）+ 尾编排 ~150 行 + workspace 裸化/ggml 接线 ~150 行。
- **R321 死墙 + T1-B 破墙案（workspace 解剖判决）**：`static_assert:6036` **只准 Q=8000/8192**（尾 tile 三角任务表 = Q 的编译期函数）⟹ Q8000 核心进不了 Q2048 红线门。**破墙 = T1-B 三分解**：① **前缀 24K 块引擎**（:6284-6485 编排 + cuBLAS QK + GEMM-softmax PV + FP32 在线合并 + 四守卫 = **75 TFLOPS 形态本体、~90% FLOPs、与 Q 解耦**，rows=12288@Q2048 仍 4x 于 T2）② 尾段 2048 三角 = 我方融合内核/float_raw 现成 ③ state 续算 = prefix_accumulator（rows×256 FP32）语义 = chunk 间咬合。**红线合规（Q2048）+ 形态红利 90% 双得**。workspace 解剖账：scores=[kBlockN=24576, rows]f16 转置、key_transposed=[256,KV]、stable_rows 四守卫张量系（value_scaled/center、block_max/sum、prefix_accumulator）。
- **R320b 诚实修正（撤回"-60%"）**：float_raw 合同（tail.cu:814-852）读出 = **Split-D 逐头内核**（Sm70D256SplitDTraits = 我方 fattn-sm70-d256.cu 祖先同族 ≈45.9 级）= adapter 兜底引擎**非 75 TFLOPS 前沿**；前沿 = **prefill.cu Q8000 核心**（GQA 打包 + cuBLAS QK + 24K FP32 块 + 双流 + `_79t_defs` 拓扑）⟹ **T1 回归原案工作量**（300 KB 引擎+编排段移植）。新收获真实：float_raw ABI 合同全落袋（D-inner 跨距 :843-848、state 在线语义、q%kBlockM==0/kv%kBlockN==0 对齐边界、scale×log2e 预乘）= Q8000 编排的尾段接口现成。
- **R320 ★★★ **T1 路径大简化（adapter 合同 + cmake 圣经读出）：79T 前沿 = `onecat_v37_dense_state_float_raw`（v37 float-state 引擎，tail.cu:814 已在树）——非 300KB torch 体；状态式分块接口与 llama chunked prefill（ub 2048 + 前缀状态续算）天然咬合。T1 工作量 -60%**（2026-09-24）

- **三件真相**：① cmake:58-60 重命名 = Split-D 源（我方 fattn-sm70-d256.cu 祖先）→ `legacy_raw` 兜底；② adapter 全文 = 固定尾壳（kTail=8000/Hq6/Hkv1）转调 **float_raw** + fp32→fp16；③ 形态优势本体 = **v37 float 引擎**（gemm_with_softmax 融合 GEMM + 24K FP32 块在线合并 + stable_rows 四守卫），全部源在树 ✓。
- **T1 终案**：编译 `v37/tail.cu` 引擎 + ggml 直调 `onecat_v37_dense_state_float_raw(q,k,v,state_max,state_sum,out,q_len,kv_len,hq,hkv,scale,unnormalized,stream)`；`LLAMA_SM70_V37F=1` 门控 + 准入检查（D256/causal/GQA 组/KV 32 对齐）；逐 ub 段状态续算 = chunked prefill 天然形态。**_79t_defs 圣经**（tile 形状 + 旋钮）照抄编译。
- **R319 立项辨析（防 R299/300 误杀 T1）+ q8-direct 灰录**：① **R300 证伪的是粗糙 T2 非 79T**——T2 S 物化 10.8 GB/call ⟶ 79T 紧凑转置 workspace 0.6 GB（rows×kBlockN×2B）；T2 M 维 3072 塌方 ⟶ 79T GQA 打包 rows=12288（4x 缓解）；79T 另有 FP32 块输出+在线合并+四守卫 = T2 全无。**79T = T2 正确实现形态，判例不冲突**。② q8-direct 性能**灰**（TTFT 185.8/191.2 = -8~-13%，窄块读 > 转换成本，与 q4 同向更甚）；显存 -2.2 GB 收益真实、门控留（VRAM 紧场景）。③ **红线校准（用户 2026-09-24 二次）**：4 卡合法（≤BL1）、显存 ≤BL1 实际占用、参数可等价齐平（批 ≤2048）⟹ **T1 = 79T 形态 @ Q2048 + TP4 适配**（rank 单 KV 组 = workspace 减半 + 并行 +33%）。源码注记（8/23-24 实测）：q4-direct 预填充 **-3.6%**（窄 u16 块读 > dequant 转换成本）、**镜像转换全成本仅 ~0.1s/770s（0.013%）**⇒ "KV 流量减半 ⇒ FA +30-60%" = 流量模型错误（FA 读次数不变、字节减半≠时间减半）。**q8-direct 模式已实现**（`LLAMA_SM70_Q8_DIRECT=1`，Path B 遗产）⟹ 降级为 **T1 显存使能件**（回收 ~470 MB f16 镜像 → score workspace 预算）。**主刀回归 T1 = 79T 形态 @ Q2048**（45.9→60+ TFLOPS 形态优势 = 唯一量级空间；红线边界：批≤2048、卡≤4、显存≤14.5GB/卡、KV≤8bit）。
- **R317 ★★★ **组合药方命中（R308×R310 = "被否项组合重测"活案例）：wait 457→0.03 ms、evnull→0；sync 289 ms = 真 GPU 等待（气泡 = GPU 影子）⇒ host 侧到物理边际；末刀 = R318 ub 加倍（气泡 ∝ 调用数）**（2026-09-24）

- **机制闭环**：`GGML_SCHED_COPIES=2`（R308 单用无效 = 当时 meta 无 event_new）× **R310 meta events**（单用无效 = n_copies=1 从不创建，:2093 条件 = **最后一行之谜落定**）⇒ events 非空 ⇒ :1728 `event_wait` GPU 侧非阻塞 ⇒ **wait 457.0→0.03 ms ✓✓**（五分探针 evnull=3→0 实证）。门值四臂 ×2 全绿（c4e11b2d）。
- **新面孔与终判**：sync 320→289 ms（COPIES=2/4）= **7 次 user 同步中 1 次真等 ≈ 280 ms = 上一调用 GPU 完成等待**——475 ms/调用节拍 ≈ GPU 串行工时本体 ⟹ **wait/sync 优化只是气泡挪房间，host 侧到物理边际**（171.5→169.5 = -1.2% 为全部所得）。COPIES=4 深排无效（事件新旧非变量）。
- **末刀 R318**：气泡总量 ∝ 调用数（~340 ms × 245 ≈ 83 s）⇒ **ub4096 = 调用减半 = 气泡减半**（mask 4.2 GB 贴边可试、从未测）⇒ 预期 TTFT **~135-145 s**。
- **R316 五分探针 = 终局定罪（算术完美闭环）**：`copy=487.6｜wait=456.96 sync=0.11 cpy=30.5｜evnull=3.0 evwait=0.0`——wait = 全部残差（487.6−30.6≈457 ✓）、**events 恒 NULL → 每次 else 全阻塞 `backend_synchronize`（3×152=456 ✓）**。**真凶 = 空 events 的全阻塞兜底**；R310 补建未被用上 = **events 创建处（:2005 区）条件未放行**（该创建逻辑从未细读 = 最后一行之谜）。修后预期 **TTFT 171→~135 s ⇒ 反超 BL1**。战训固化（四犯）：**写账本先于 git，顺序死规矩**。
- **R315 gate 跳过 = 无效（gate 正名：从不触发，n_inputs>0）**：copy 486 原封、**门值四臂全绿（c4e11b2d ×4）= 安全性无忧**。**最后未计时块浮出（R311 代码记忆重审）**：else 分支开头 **:1728-1732"覆写前等待"**（`event_wait` 或 events 空时 `backend_synchronize` 全阻塞）——3 次 × 152 ms = 456 ✓ 与 fb=3 严丝合缝。**下刀 = 五分探针**（该块计时 + events 空值计数）最后一击。战训上墙：**提交前自问"工作树有变化吗？"**（漏写账本三犯）。
- **R314 + R313c 终案 v2（计数钥匙）**：R314 权重一次性拷贝**无效**（copy 486 原封）——但 `user=7 fb=3`（10 次全落两计时分支 ≈ 30 ms）⟹ **456 ms 无家可归者唯余 :1702 gate（split 间等待）**；算术双合 **456 = 2 × 228 ms**（splits=2 跨界等待 ≈ GPU 单 split 时长）= **split2 等 split1 的调用内管线串行化（气泡本体）**。gate 存在理由 = 防跨 split 分配区复用踩踏——**R302c-3 收养后地址稳定、复用不发生 ⟹ gate 可免**。**修法 G = gate 门控跳过（env 判值）** ⇒ 预期 copy 487→~31 ms、**TTFT 171→~137 s ⇒ 反超 BL1**。（R314 权重一次性拷贝代码无害留树、门控默认关；MoE 段嫌疑正式撤销。）
- **R313b 真凶终案（宏定义反转 + 算术双合）**：`tensor_copy` = **哈希宏（μs 级）**——456 ms 不在查找 ⟹ 未计时分支仅剩 **MoE 专家拷贝段**（:1735-1822：`:1746` 源后端全同步 + ids D2H 同步 + 逐专家 memcpy；Qwen3.8 = MoE 架构，R293 `moe_total` 早有此名）⇒ 10 权重输入 × 45 ms 天衣无缝；**R312 async=0 真因同步落定** = MoE 分支先于 else-async 命中。**修法 F' = WEIGHTS 拷贝一次性化**（静态权重首拷后跳过，env 判值）——预期 copy 487→~35 ms、**TTFT 171→~135 s ⇒ 反超 BL1**。
- **R313 四分探针判决（拼图最后一块）**：`sync=0.12 + cpy=30.5 ms`（分支近零罪）、**456 ms 藏于逐输入 :1713-1716 段（10 次 × 45 ms）** = `tensor_copy()`/后端查询嫌疑；`async=0`（R312 委托零命中 = guard 早退，歧义①向）；:1702 闸排除（n_inputs>0 不触发）。**下一读 = tensor_copy 本尊定终案**。战训：账本条目"漏写就 commit"二犯（提交前自问"工作树有变化吗？"）。
- **R312 meta cpy_tensor_async 分片委托 = 无效（灰）**：copy=486.1 ms 原封、TTFT 171 持平。**歧义两说未分**（第五次"测后再修"）：① `is_meta(src->buffer)` 开头即 false（input 在普通 CUDA 缓冲上）→ fallback 全走；② simple 侧 async 内部自等。**下刀 = 四分探针**（async 命中 / fallback 次数 / 等待 / 纯拷贝）一跑定音再修。构建+作业形态注记：iface 槽位"加塞而非填坑"曾致 too-many-initializers（表本有空槽——填坑前先看表）。
- **R311 字节定量判决（机制三分终局）**：`copy=487.7 ms/call、251.7 MB/call、10 次拷贝`——带宽病 ❌（516 MB/s = D2D 的 3-4%）、launch 开销 ❌（49 ms/次拷 25 MB）、**逐拷贝同步等待 ✅（49 ms/次 ≈ GPU split 时长）**。**真凶链补全**：meta `cpy_tensor_async` 缺失 → 10 次/调用全走 :1826-1833 fallback → **:1827 `ggml_backend_synchronize(input_backend)` 源后端全同步**——R310 只换了该分支一半、此行未动 = events 无效之谜破案（少改一行）。**修法 D = meta 补 `cpy_tensor_async`（委托 simple 后端 cudaMemcpyAsync）**⇒ fallback 不走、拷贝全异步；预期 copy 段 488→~20-50 ms、**TTFT 171 → ~115-125 s ⇒ 反超 BL1（159.0）**。
- **R309 CPROF 分相判决（定量定罪）**：`copy=489.5 / compute=155.9 / other=0.0 ms/call`（splits=2.0）⇒ **输入拷贝段 = 76% 的 compute 段时间**。机制假说（可测）：meta backend **无 event 对象** → :1728-1832 兜底 `ggml_backend_synchronize(全后端)` = **每输入全同步等 GPU**（host 被 GPU 阻塞的真身；双缓冲 R308 无效亦吻合——等待不在 copy-slot 面）。**候选修法 B = events 补建 → `event_wait`（GPU 侧）替代 host 全同步** ⇒ 预期 copy 段 489→<50 ms、**TTFT 171 → ~120-130 s ⇒ 直接超 BL1（159.0）**。备选 A = 输入拷贝融合（中改）。下轮读 events 创建逻辑定 B 可行性。

### R304-P ★★★ **天花板实测（DPROF 分相）：cuBLAS QK 在小 N 形状塌方 = 0.167 TFLOPS（正常 30-60 的 1/200）——T2 隐藏主凶补全、Path A 融合价值再确认**（2026-09-24）

- **分相账（q6=3072, kbn=512 浅段块）**：pack 0.32 / **qk 4.81** / soft 0.04 / rest 0.00 ms ⇒ QK GEMM 本体 0.805 GFLOP / 4.81 ms = **0.167 TFLOPS**（cuBLAS 该形状 = 小 N(512) 启发式塌方；softmax 无辜）。
- **三重意涵**：① T2 慢的隐藏主凶补全（S 物化 + **每块 GEMM 小 N 形状塌方**双重税）；② Path A 融合内核（45.9 TFLOPS）**远超** GEMM 组合形态 = 融合 mma 形态本身就是当前水位的持有者；③ **P-P3-T 论证修正：cuBLAS 不是形态上限参考**（同形状更慢）⇒ 41% 峰值利用率或已是 m8n8k4 形状真实水位，显式 mma 空间须**内核内实验**（非 GEMM 对标）。
- **待补**：深段块（kbn=24576）分相采样被 `nd<64` 截断（PPL 前 64 调用全浅段）——一行改深段采样定全形状结论。
- **R304-P 补记（工具线插曲后收线）**：DPROF 探针在 llama-perplexity 第 ~65 调用（kv_len 首增处）稳定崩于 `ggml_cuda_compute_forward`（异步归因 = 探针自身路径，零块守卫不治根）；**无探针的 decomp 运行全部正常** ⇒ 探针之罪、计算无恙。64 条浅段样本有效（0.167 TFLOPS 结论保持）；深段数字**对 P-P3-T 判决非必需**（40:1 + 41% 峰值 + M1 已闭环）——**不再追探针，主线先行**。工具线战果：mcp-ssh 常驻 shell + tmux 具名会话 + 结束符秒查 = 新作业形态（长活 tmux、绝不久睡）。
- 教训链补全："比率是数据" + "形状是数据"（GEMM 效率是形状的函数，微基准的形状组合必须逐一对齐生产负载——T2 用生产形状跑出 0.167 TFLOPS 即为实证）。

- **T1 实现蓝图（作废：被 R303-R 证伪，只改灰不删）**：smem 50→82 KB ✓ 免缩 kBlockM；寄存器 ≈150→222 ✓ <256（minBlocks=1）；**配对陷阱已识别**：必须组内配对 `head_q = j·gqa + 2c' + pass`（跨界对 (5,6) 跨 kv-head 禁忌），grid.z' = hkv·gqa/2；**mask = 解析式因果**（:1016 `Mask<true,false,false>(kv_len, kv_len-kv_offset)`——sm70 内核不读 kq_mask 张量！T2 学到的张量 mask 语义不适用于此内核，重要辨析）；三接缝 = ① Q 暂存双 tile（sQ×2）② QK-softmax-PV 双 pass（acc_s 复用、row 状态/o_storage/P-smem 逐 pass）③ K/V 预取提至 n_block 顶（tKrKNext/tVrV0-1 加载一次、双 pass 共读 smem）；epilogue 逐 pass（row_sum allreduce + out 写带 `head_q = z'·2+kPass`）。
### R303-R ★★★ **GQA-摊销证伪（比率核算救场，动刀前夜止步）：compute:memory = 40:1 ⇒ 摊销天花板 2.1%；真瓶颈 = MMA 形态效率（45.9 = 峰值 41%）⇒ 回归 P-P3-T 显式 mma PV 原义**（2026-09-24）

- **比率核算（决定性）**：256K 单层 FA = 287.5 ms @45.9 TFLOPS = **V100 fp16 峰值（~112 TFLOPS）的 41%**；K+V 全读（含 6x 组内重复）= 6.4 GB ⇒ 900 GB/s = **7.1 ms**（compute:memory ≈ **40:1**）⟹ GQA 摊销理想上限 = 省 5.9 ms/层 = **2.1% 天花板**。且逐 n_block K/V 工作集 = 32 KB ⇒ **6x 读本来就是 L2 命中**（非 DRAM 税）——流量论点的物理前提本身不成立。
- **双重否定闭环**：M1（79T 打包 GEMM = Path A 持平）+ 本核算（2.1% 天花板）⇒ **R303 GQA-摊销入灰**（证伪前的 T1 蓝图同步作废——它救下了 2-3 轮注定白做的内核手术）。
- **真瓶颈定案**：45.9/112 = 41% 峰值利用率 = **MMA 形态效率问题**（m8n8k4 喂不满 + smem→reg 碎片搬运开销）= **P-P3-T 显式 mma PV 的原始定义**（R272-V2 同号 +1-2ms/轮先例 = 正统后继路径）。杠杆清单：① PV 段显式 mma.m8n8k4 双 HMMA 四碎片形态（R272-V2 已有可复用基础）；② QK 段 gemm<> 助手的碎片装载开销；③ kDChunk/warp 行配比调参。
- 教训：**流量类论点必须先算 compute:memory 比**（本轮 40:1 一条算术否定三天蓝图）——"时间是数据"的姊妹律："比率是数据"。
- **R303 预登记（[S4] 沿用 + 本条）**：FA 算子 ≥+20% 或 256K TTFT ≥+7%（175.8 → ≤164 s）；**门值必须原样**（`c4e11b2d`/`f3edac19` 双谱系）；纯步不回退 >2%；显存不升。及格 = 全过；失败入灰。
- **实现路径（T1 阶段）**：现有 dense 内核模板化 `kHeads=1|2`（头循环包住 QK/softmax/PV，K/V 块暂存提至循环外）+ launcher 分派（`LLAMA_SM70_GQA2` 判值门控，默认 =0 走现路径）。

### R302c ★★ **收养设计尝试：概念成立（结构键 + 逐张量验证 = 断言教训闭环），整合面两坑 ⇒ 止损回退 V2；两关键账本修正**（2026-09-24）

- **两账本修正（重要）**：① **alloc 只计 rebuild 调用**（[RT] 计数器语义再确认）——68.1 s/91 rebuild = **748 ms/rebuild**（非 139 ms/call 的均摊）；② rep1=176.2 / rep2=177.1（无差）= 收养在 SLOTS=1 下未生效（prefill/decode 两形态计划被单槽互相驱逐 ⇒ rep2 仍逐步重放置）。
- **收养设计（概念成立）**：结构键 + 逐张量 `size_max` 验证（= ggml-backend.cpp:2359 断言要求的验证语义，断言教训完成设计闭环）+ 收养沿用旧地址（免 sync）+ rep1 即预热（MAX 放置向下适配）。
- **整合面两坑（新知识入册）**：① **`ggml-backend-meta.cpp:1821 bufs.back() != nullptr` 断言**——收养跳过 reserve 侧效应 × meta/T8 缓冲记账失配（SLOTS=2 实测崩溃）；② SLOTS=2 与单池 disarm 的交互未明。⇒ **止损：工作树回退 V2 已知良好态**（git checkout + 重构建 ✓ 175.8 s 采用态复原）。
- **下一步转向**：R302c 整合专项（meta 缓冲记账机制读码）排后；**主刀 = P-P3-T GQA-smem 摊销**（FA 78% GPU、KV 流量 ×6 = 最大奖）。

### R302b ★★★ **alloc 33 s 真身终局：重建路径全后端同步（隐藏 GPU 等待第三案）——逐层剥洋葱定案**（2026-09-24）

- **剥洋葱链（每层实证）**：① [GAT] 埋点：reserve_n_impl 仅 **2.6 ms**（3963 节点）、buf_loop **µs 级**（单池 v2 完美生效）⇒ reserve 无罪；② `GGML_SCHED_SPLIT_CACHE=1` **零效果**（33.04 vs 33.03 s）⇒ split_graph 无罪（且发现指纹 split 缓存 = 更早战役遗产、默认 OFF、P10 探针真身 = `[SCHED]`）；③ cudaMemset 逐张量嫌疑**条件排除**（仅非-COMPUTE 量化张量，不在每调用路径）；④ **算术闭合终局**：91 rebuild × ~700 ms ≈ 64 s（双 rep）= **`:1654-1658` 重建路径 `ggml_backend_synchronize(all backends)`**（注释"re-allocation may cause split inputs to move"）= **隐藏 GPU 等待**（R298-P compute-async 家族第三案）；ub 缩放表象（28 vs 139 ms/调用）= GPU 越忙等越久 ✓ 全解释。
- **R302c 候选（下轮定设计）**：a) **形状类预热**（启动期对全部 bucket 类 × 形态预 reserve → 运行期全走命中路径 = 零 miss 零 sync ✓ 零语义风险，首选）；b) 桶阶梯加粗（24 类→12 类，sync 省 8.4 s vs FA padding +5-13% = 平/负，灰）；c) 放置稳定性/异步安全（深水区）。
- 工具链注记：paramiko 通道 ~5 min 超时（长活必须 nohup + 分段收）；GAT2 埋点哑因未明（GAT 同 env 同流有效——疑作用域/优化差异，不阻塞主案）；`[SCHED]` %256 阈值打印两次未现身（探针可靠性注记）。

### R302 ★★ **alloc 45.6 s 战役：增长余量（零效果）→ key 粗化（assert 证伪）→ 机制全图落定，单池解耦 v2 立案**（2026-09-24）

- **目标**：ub2048 采用栈 alloc = 45.6 s/46 rebuild（900 ms/次整池 free+malloc）= 墙钟 25%；砍到 ~3 s ⇒ TTFT ~150 s ⇒ **反超 BL1 spec-off（159.0）**。
- **证据链两轮**：① **增长余量（ggml-alloc.c realloc 分支 chunk max_size ×1.5+16MB）= 零效果**（alloc 45.6 不动）——因 `realloc = buffers[i]==NULL`（:1183）= **新 plan key 必然全新分配**，T8 槽对"单调增长 + 46 个一次性 key" = 逐 key 驱逐（slot_clear :610 **vbuffer_free 整池**）+ 重建，增长余量被架空。② **key 尺寸量化（2 的幂档）= assert 证伪**：`ggml-backend.cpp:2359` 张量放置越界 assert——**plan 地址布局不可跨尺寸复用**（"re-validates all sizes" 实为总量级弱验证），灰名单。
- **机制全图（读码定案）**：plan_key 哈希含全量 `ne[d]`（:570）⇒ bucket 步进 = 新 key；slot 驱逐 free 整池 ⇒ 46 key × 900 ms。**plan 缓存对单调增长工作负载 = 结构性无效**。
- **v2 实测采用（R302-V2）**：单池解耦七处刀（驱逐缴械移交池 + 池承接 + teardown 收池 + key 粗化撤销）——**TTFT 182.6 → 175.8 s（-3.7%）/ 1344 t/s、纯步 34.27 ms 持平、门值 `c4e11b2d` 四臂全绿（分配器零数值影响 ✓）**。alloc 45.6 → 33.0 s（-28%，未到 3-5s 预期 = 重放置 walk/残余分配成本待 R302b 深挖）。**距 BL1：1.15x → 1.11x**（159.0 vs 175.8，spec-off 口径）。显存注记：carry 池常驻 + 增长余量峰值未单测（粗算合包络）。
- 教训：③ 测试服务必须收尾清场（孤儿 server 连续两次触发 BUSY_SERVER 拦截——互斥纪律两次救场）；④ "re-validates"类注释要按代码实证强度解读。

### R301 ★★★ **刀序终局定案（分账先行）：FA = GPU 78%（R264 账本级）⇒ P-P3-T 优先；算子流量分析定出真大奖 = GQA-smem 摊销（grid.z=gqa 令 KV 全局流量 ×6）**（2026-09-24）

- **分账（既有账本 + 本轮 [RT] 解剖）**：FA = 峰值 ubatch GPU **78%**（R264）⇒ GDN ≤22% ⇒ **P-P3-T > GDN 侦察**（刀序有账可依）。ub2048 采用栈 [RT]：rounds=246（116 预填充调用）、compute 129.9 s = 墙钟 71%（GPU-bound 维持）、纯步 34.28 ms（三测一致）。
- **★ GQA 流量病根（P-P3-T 真大奖）**：Path A `grid(q_len, nb*hkv, **gqa**)` = 6 个 Q 头各自独立实例、**各读一遍同组 K/V**（242 MB/层 >> L2 40 MB ⇒ 真 6x 全局流量）。⟹ **GQA 打包（6 头共享一次 K/V smem 载入）= 流量 ÷6 杠杆 = 1cat 梯子 45.9→60.8 的机制本体**。上限理论：FA 45.9 → 60+ TFLOPS 级。
- **P-P3-T 刀序（融合形态内强化 Path A）**：① GQA-smem 摊销（大奖）→ ② prefix/tail 分离（零 mask 大块 + 因果尾批 + 首 64 修复 = mask 流量 + 分支税）→ ③ PV 显式 mma（R272-V2 同号 +1-2ms 先例）→ ④ kBlockN 调参 + RegP 组合（免费）。
- **新刀点（ub2048 解剖红利）**：**alloc 41.7 s/46 rebuild = 900 ms/次 = 2.1 GB mask 缓冲每 rebuild 重建**——T8 槽未吸收 ub2048 新形状（重建原因多为调用间形状微变）⇒ **mask 缓冲定尺复用 = 低垂的 5-10% CPU 侧**（与 R282 统一 mask 论同源，收窄版）。
- ⚠ GGML_PROF 探针触发名失考（strings 无果）——ops 级表待复原；本轮以 R264 账本级份额定序（有账可依原则）。

### R300 ★★★ **战略定案：P-P3 分解版证伪入灰（双重否定）；ROI 重排：ub2048×桶化叠加快测上桌（外推 TTFT ~165 s = 距 BL1 1.08x）**（2026-09-24）

- **P-P3 分解版 = 死路（灰名单，只改灰不删）**：① **M1 早证**——79T 分解在 m=2048 生产形状 = 287.5 ms 与 Path A 完全重合（预登记 +25% 前提与自家数据矛盾，当时失察）；② **T2P 实证 + 流动物理**——S 物化总流量 = 12B×q6×kv **与分块无关**（块不变量三重互证），打不过 score 不出 smem 的融合内核；f16 S 减半仍 -30% 量级，无解。
- **幸存资产收编为 P-P3-T（融合形态内强化 Path A）**：PV 显式 mma（R272-V2 同号 +1-2ms 先例）、GQA 打包 smem 摊销（1cat 思想入融合内核）、prefix/tail 分离（零 mask 大块 + 张量尾）、kBlockN 调参、RegP 组合。预期 +5-15% FA。
- **新头号候选（一个 env）→ 实测采用**：**UB=2048 × 桶化叠加（R300 采用）**——同口径（spec-off、SLOTS=1、r=1.08）双 rep：U512 **190.1 s / 34.25 ms** vs **U2K 182.6 s / 34.35 ms**（TTFT **+4.1%**、纯步 **+0.3% 不回退** ✓、离散 ≤0.1%）⇒ **采用（UB=2048 入栈）**。外推修正：ub 大批量收益大头已被桶化收割（同一 host 根病），残余 +4-5% = GPU 效率项。R295 的 tg -20% 之谜同步消解 = 当年口径混杂（SLOTS/显存），同口径零回归。显存 9.35 GB/卡（SLOTS=1）包络更松。**距 BL1：spec-off 1.21x → 1.15x**（159.0 vs 182.6）。教训：预言的坑（R295 mask OOM @SLOTS=3）第一跑又踩一次（先验写了复查却照跑）——预言类先验必须转成脚本里的硬约束。
- **新侦察项**：GDN 内核（48 层、42-50 GB/s 未开发大项）立项。

### R299-T2P ★★ **T2 性能首测为负（先验"为负"分支如实触发）：PPL 时长 +57%，F1 压测 71 t/s；真主凶 = S 矩阵物化流量（块大小不变量实锤）**（2026-09-24）

- **ABBA 中止决策**：C1 对照臂落袋（TTFT 208.0 s ≈ R297 采用值 208.9，漂移 0.4% ✓ 量具可信）；F1 = **71 t/s（-94%，ETA 55 min）** ⇒ 立即止损杀序列（保 C1）。
- **双成本项分离（时间也是数据——PPL 时长早有预警，本轮补读，教训入档）**：
 - **S 物化（真主凶）**：score 全矩阵 [q6, kv] f32 物化 = ~10.8 GB/调用（写+读+P+PV）vs K/V 打包 0.9 GB（12 倍）；**块大小不变**（8192/24576/65536 三档 PPL 时长逐位同 = 3:45）——与 S 总流量 ∝ q6×kv 与 kbn 无关的算术完美互证。
 - **工作区 realloc**（次凶，修复 ✓）：grow 判据遇上 `kbn_max=min(kBlockN,kv_len)` 逐步递增 ⇒ 每调用 cudaFree+cudaMalloc × 5.5 万次；修 = 固定 cap。PPL 时长未变 ⇒ realloc 非 PPL 主项，但 F1 压测跑的是病版（-94% 的放大项待分离）。
- **正确性不受影响**：PPL 1.0027 vs 1.0025（+0.02%）、块不变性、门值 `c4e11b2d`×2 位级同（T4 不重立，=0 门值 `f3edac19` 保留）。
- **下一步（先验预案）**：① 单臂 F1b 复测（realloc 修复版）分离两成本项的真实占比；② 视结果走 **T3 = S 不物化改造**（f16 S + 子块流水窗，或完全融合 tile 流水 = P-P3-T 单算子化路线）。灰色记录：T2 首版性能 = 不采用。

### R299-T2V ★★★ **P-P3 分解 T2 数值验收通过：PPL +0.02% ≤ 0.1%（1.0027 vs 1.0025），块尺寸不变性逐位自洽；真凶 = K/V 头行交错布局**（2026-09-24）

- **破案链（dump-diff 定量武器 + nb 实测收官）**：① 块尺寸不变性（8192/24576 PPL 逐位同）⇒ 块不变量数学域；② dump-diff ⇒ 结构性错（凸组合越界 +1.081、整列零）；③ **nb 实测终局**：target K `Kne=256,768,2 Knb=34,544,272` ⇒ **物理行 r = 2j+h（头行交错）**，我按"头在外 ne1×256"切组 = 读交错错行 ⇒ gather 打包（Kp/Vp per-block）+ 步长按 nb 推导（f16 直读/暂存保交错/nc 打包序三分支）+ 头索引 gkv≠g0 修正。
- **PPL 陪审终局**：decomp=0 **1.0025±0.00036** vs decomp=1 **1.0027±0.00038**（kbn 8192/24576/65536 **三值逐位相同**）⇒ **+0.02% ≤ 0.1% 判据 ✓ 数值验收通过**；块不变性 ✓ 合并数学自洽。
- greedy 哈希分叉 = 预期内求和序舍入差（近平局翻转），归 **T4 门值重立条款**（唯一配额）管。
- **阶段 bug 尸检（累计 8 个真 bug，各带实证）**：mask f16 类型 / Q 布局 [d,h,q] / 多卡单例交叉 / GEMM 混型禁令 / 块 softmax rescale 反向+全 mask 块 / 步长语义反转（Q vs dst）/ **K-V 头行交错（本案主犯）** / 头索引 gkv。方法论战果：块不变性二分、dump-diff 定量、nb 实测三板斧全程禁臆想。
- **下一步 = T2 性能 A/B**（预登记及格线：FA 算子 ≤230 ms（+25%）或 256K TTFT ≤194.7 s（+7%），基线 r=1.08 采用值）→ T4 收口（门值重立须报）。

### R299-T2 ★★ **P-P3 分解 T2 阶段记录：全链跑通（576/576 进新路径、零崩溃）但 PPL 陪审未过（+1.68% > 0.1%）⇒ 数值 bug 待 dump 二分定位**（2026-09-24）

- **已跑通**：GQA 打包（Q 组打包 kernel，[d,h,q] 实测布局）+ 双 cuBLAS GEMM（QK/PV，fp32 累加）+ 精确块 softmax + 散写 dst；per-device workspace/cublas 句柄；probe 576/576 ACCEPT、0 REJECT。
- **连环修掉 5 个真 bug**（每条都有实证定位）：① mask f16/f32 类型链；② Q 布局 [d, h, q]（诊断打印 `qnb1=6144=1024×ne[2]` 定案，"预转置"必须件第三次显灵）；③ **多卡单例交叉**（ws/cublas 句柄全局 → per-device，97 调用后跨卡指针崩）；④ cublasGemmEx **禁止混型操作数**（A32F×B16F = NotSupported → Q 打包直接出 f16）；⑤ 块 softmax **rescale 方向反 + 全 mask 块垃圾因子**（桶尾 -inf 块）。
- **PPL 陪审（判据 ≤0.1% 相对差）**：decomp=0 **1.0025±0.00036** vs decomp=1 **1.0193±0.00071** = **+1.68%（超 17 倍）⇒ 数值真 bug，T2 不采用**。e2e 贪心第 ~10 token 分叉（0.185 overlap）与此一致。
- **下一步（已定方法）**：dump 二分定位——env 门控 dump S（QK 后）/P/O/dst 首值，S 已错 ⇒ QK/打包几何错；S 对 O 错 ⇒ softmax/PV 错。禁止继续盲推（第五定律二连）。
- 工具链战痕累计（本 feature）：pwsh `$( )` 本地抢算、`tail -3` 裁掉 PPL 正式行、编辑失配 ×2（注释措辞/已替块）——全部自抓，数据未污染。

### R299-T1 ★★ **P-P3 分解 T1（门控 + 私有常驻 workspace + 直通）验收全过：=0/=1 双判据位级等价**（2026-09-24）

- **实现**（commit `0da82d593`，分支 `feat/p3-decomp`，fattn.cu 零改动）：`LLAMA_SM70_FA_DECOMP`（判值）+ 私有常驻 workspace（static 设备缓冲，不入图防多槽翻倍 = P3-RECON [S2]-1）+ T1 直通（Path A 本体照跑）。
- **五臂门**：gb-off/gb-off-nospec/gb-on/gb-on-nospec 全绿（无回归）+ t1-decomp 臂 `f3edac19…` ——但 **probe=0 自曝验证空洞**：门臂 greedy q<256 不进 sm70 路径 ⇒ =1 判据未覆盖。
- **补 q≥256 端到端对照（t1-e2e.sh，greedy/seed 固定，~4500 tok prompt，双臂）**：

| 臂 | probe_hits | 输出 sha256 |
|---|---|---|
| DECOMP=0 | 0（默认路径零触碰 ✓） | `86707d24…` |
| **DECOMP=1** | **576**（真实 FA 调用全进新分支 ✓） | **`86707d24…` 逐字相同** |

- ⇒ **T1 双判据验收完成**；教训入档：**门臂形状必须覆盖新码路径**（q<256 走 stock = 无效验证，probe 计数自证是抓这类空洞的关键）。
- 下一步 T2：GQA 打包 QK（cuBLAS 块）+ PV（显式 mma，FP32 块输出）+ score 布局（p3-decomp spec [S2]-2/3/4）。

### R298-P ★★ **enqueue 税解剖（P6/P10 量具，256K 单请求）：巨兽不是 meta（7 ms/call），是 sched 每调用图规划（263.7 ms/call = 墙钟 69%）**（2026-09-24，前哨）

- **账（r=1.08，rep=1，505 调用）**：`enqueue 133.1 s = 263.7 ms/call`（墙钟 69%）｜`alloc 28.5 ms/call`（T8 已治）｜`setin 22.5 ms/call`｜`build 0.25 ms/call`。
- **[META] 对照**：`total=7.049 ms/call`、sub/call=15.8、ar/call=14.8、dev_hist 中位 25µs、85%<60µs ⇒ **meta 派发只是零头**；老 E 系列"meta 税"叙事在 256K 服务负载下让位于 **ggml_backend_sched 每调用图规划**（split/指派 O(大图)，prefill 图 5000+ 节点）。
- ⇒ **桶化已让图复现（rebuild 66）但 sched 规划仍逐调用重做** —— r=1.08 收益饱和的真因。
- ⇒ **下一刀 R298 = sched 规划缓存/免重做**（按图指纹缓存 split/assign 结果，复用轮直达 submit）。**天花板测算**：enqueue 264→~30 ms/call ⇒ TTFT 192→**~100-115 s（spec-off）⇒ 反超 BL1（152.5 s）在射程内**。
- 口径注记：[SCHED] P10 探针无输出（打印条件待查，不影响主账）；[META] calls 口径（4224）与 rounds（505）的映射未完全厘清，7 ms/call 下界结论不受影响。
- **⚠ 修正（同日读码，第三定律）**：`enqueue_us` 打印源 = `t_rt_compute_us` = **`ggml_backend_sched_graph_compute_async` 全程**（`llama-context.cpp:1452/2610`；`graph_compute` 的 `batched` 参数只是线程池选择，**非同步标志**）⇒ **"纯 sched 规划税"解读不成立**：263.7 ms/call 真身 = {sched 规划 O(大图) + split 边界隐式 GPU 等待} 二选一/混合。**R298 刀向暂缓**，双对照臂定性后动刀：(a) GBON2NS + `LLAMA_ROUND_TIMING_SYNC=1`（sync 单列）；(b) r=0 同口径 [RT] 账（桶化收益 77s 的来源闭合）。禁臆想第三定律本轮再立功。
- **★★ 终局定性（SYNC 分离臂，R298-P 收口）**：`sync_us=190.2s`（= compute 起点到同步完成）− `compute 37.6s` ⇒ **真实 GPU 等待 152.6 s = 预填充墙钟 192 s 的 79%**；CPU 侧（dispatch 37.6 + alloc 4.9 + setin 10.5）≈ 53 s = 21%。**⇒ 预填充 = GPU-bound 定性（账全闭合）**：桶化 +38% = 纯 CPU 侧收益（CPU ~115→40s，算术闭合）；**剩余最大肉 = GPU 内核（FA 占 GPU ~60-70%）⇒ 主刀最终定案：P-P3 内核线 > R298 调度线（仅剩 CPU ~15-30s 小头）> 吐字轮结构**。无 SYNC 臂的 compute=133.1s = 调度 + split 边界隐式等待的混合体（口径注记）。

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
