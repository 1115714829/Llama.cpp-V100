# AGENTS.md — 路由层（自动加载）

> 本文件只保留：**① 强制工作项 ② 路径地图**。其余一律外链，勿在此扩写。
> 新会话顺序：本文件 → [`docs/v100-dev/README.md`](docs/v100-dev/README.md) → 按需下表。

---

## 1. 强制项（必须做，不可省）

### 1.0 ★ 终极验收目标（用户 2026-09-23 定，凌驾所有其他目标）

> **折腾 llama.cpp 的唯一意义 = 追平 1cat-vLLM。否则不如直接用 vLLM。**

| 项 | 定义 |
|---|---|
| **标准线 BL1** | 1cat-vLLM 标准服务（`vllm-1cat.service`：FP8+DFlash2、TP4、**基本启动参数**）的 **256K 压测实测值** = 极限目标 = 硬件能力证明。实测后填入下表 |
| **追平判据** | llama.cpp 在同口径 256K 实际负载下（同 prompt/采样/请求形状，见 `.dsh/skills/stress-256k/`），**全面能力 ≥ BL1：吐字（tg t/s）、预填充（TTFT/pp t/s）、实际使用直观体感（tpot / ms 每轮 / 响应流畅度）** |
| **显存包络（不合格线）** | 总显存 ≤ vLLM 部署的 **4×V100-16GB**（约 64 GB）；**用更少卡达标 = 优，超过 4 卡 = 不合格** |
| **调测许可** | 小模型（Q2_K 等）可用于开发/基线/压测调参（内核与运行时收益随全量模型带走）；**验收必须在 Q8_0 全量模型上** |
| **全自动化授权** | （用户 2026-09-23）本目标内**自主决策执行、不再请示**——含内核级改动、构建、服务启停、任务自分配、**关键节点自主 `git commit`（绝不 push；本地 git 历史 = 备份环境）**；唯一约束 = 不偏离本核心目标。硬纪律（其余红线、度量底线、实机互斥）不变 |

| 256K 指标 | BL1 vLLM（标准线） | BL2 官方 llama | BL3 B5 | 我们（目标） |
|---|---|---|---|---|
| 预填充 TTFT / pp t/s | **152.5 s / 1549 t/s**（152.26/152.82） | 待测填入 | 待测填入 | **≥ BL1** |
| 吐字 tg t/s（附 AL） | **124.6 t/s**（118.0/131.1） | 待测填入 | 待测填入 | **≥ BL1** |
| 体感 tpot / ms 每轮 | **≈8.0 ms**（8.47/7.63） | 待测填入 | 待测填入 | **≥ BL1** |

> **BL1 口径（2026-09-23 实测，stress-256k 技能）**：ctx 262144 的 **90.14% 填充**（prompt 236,313 tokens；每 rep 加盐防 prefix cache、句序号防投机刷分）；gen 128；thinking on、chunked prefill 2048、DFlash2、TP4 卡 0,1,3,4；服务**原样启动参数**（`vllm-1cat.service`）。比 1cat 文档 2438 t/s 低是因为那是 chunk 8192 pure prefill 契约——本表是**标准服务基本参数实测**，即标准线本体。

### 1.1 维护 [`1cat-vllm-v100-study/PLAN-GRAPH.md`](1cat-vllm-v100-study/PLAN-GRAPH.md)

1. **先读图再动手**；汇报用节点 id（N0/X5/P-D1…）。
2. **试过的都要上图**；证伪**只改灰、不删**。
3. **每完成一件事回头复读图**：补节点？补粉框条件结论？补边？
4. 外部想法登记进图 **§3** 并连线。
5. 对外交接图：[`docs/v100-dev/4-规划拓扑图.md`](docs/v100-dev/4-规划拓扑图.md)（与上图保持同步）。

### 1.2 比结论 / 报数前先读

| 必读 | 原因 |
|---|---|
| [`docs/v100-dev/00-红线.md`](docs/v100-dev/00-红线.md) | 红线与度量底线（**不可覆盖**） |
| [`1cat-vllm-v100-study/BASELINE-LEDGER.md`](1cat-vllm-v100-study/BASELINE-LEDGER.md) | 采用链 / 唯一对比尺 |
| [`docs/v100-dev/3-失败记录.md`](docs/v100-dev/3-失败记录.md) | 勿重做清单 |

### 1.3 范围（已定）

**V100 二次开发**（代码/内核/运行时）≠ 启动参数调参。主攻格式 **Q8_0** → [`05-量化范围.md`](docs/v100-dev/05-量化范围.md)。

---

## 2. 路径地图

| 路径 | 是什么 | 怎么用 |
|---|---|---|
| [`docs/v100-dev/`](docs/v100-dev/README.md) | **当前有效文档（3+1+范围）** | 对外/决策只认这里 |
| [`1cat-vllm-v100-study/`](1cat-vllm-v100-study/) | 过程档案、账本、E 系列判决 | 证据与细节；冲突以 `docs/v100-dev/` 收口 |
| [`llama.cpp/`](llama.cpp/) | **交付物**（二次开发代码） | 改码前读 [`llama.cpp/AGENTS.md`](llama.cpp/AGENTS.md) |
| [`models/`](models/README.md) | 本机参考模型（vocab GGUF） | 大权重在服务器 `/mnt/3.84t/**` |
| [`olddoc/`](olddoc/README.md) | 旧私有/散件回档 | `qwen-private/QWEN.md` **含口令，禁止外传** |
| [`v100-refs/`](v100-refs/) | **全部外部参考**（已收拢，只读） | 见下表 |

### `v100-refs/` 内（一律只读、禁止改）

| 子目录 | 含义 |
|---|---|
| [`vllm/`](v100-refs/vllm/) · [`1cat-vllm/`](v100-refs/1cat-vllm/) | 官方 vllm 与 1cat 分支 |
| [`vllm-1ca-vllm分支版本差异文件/`](v100-refs/vllm-1ca-vllm分支版本差异文件/) | **分叉点基线**（原 `vllm-forkpoint`）= 1cat 起分支时的 vllm；对照此目录看 **1cat 相对分叉点改了什么** |
| 看差异的命令 | `git -C v100-refs/1cat-vllm diff 4ff865c38..HEAD -- <path>`（分叉点在 1cat 对象库内） |
| 其余 | `flash-attention-v100/` `sm70-attn/` `v100-skinny/` `ninfer-v100/` `jusko-…/` `sglang-V100/` `xllama.cpp/` `qwen38-v100-serve/` — V100 内核/服务参考 |

### 研究档案深链（按需）

| 需要 | 打开 |
|---|---|
| 可信度 / 旧结论更正 | [`AUDIT-2026-09-20-dsh.md`](1cat-vllm-v100-study/AUDIT-2026-09-20-dsh.md) |
| 当下实况交接 | [`HANDOFF.md`](1cat-vllm-v100-study/HANDOFF.md) · [`HANDOFF-2026-09-22.md`](1cat-vllm-v100-study/HANDOFF-2026-09-22.md) |
| 吐字战役报告 | [`DECODE-CAMPAIGN-REPORT-2026-09-23.md`](1cat-vllm-v100-study/DECODE-CAMPAIGN-REPORT-2026-09-23.md) |
| 已突破 / 计划 / 失败 | [`2-已突破`](docs/v100-dev/2-已突破方向.md) · [`1-计划`](docs/v100-dev/1-计划突破方向.md) · [`3-失败`](docs/v100-dev/3-失败记录.md) |

---

## 3. 一句话现状

**终极验收 = 追平 BL1（§1.0）**：256K 预填充+吐字 ≥ 1cat-vLLM 标准线，且显存 ≤ 4×V100-16GB 包络。当前 B5 = 45.94 ms/轮（tg 113/94/146，门 `f3edac19…`）；解码硬指标 tg≥150（≤37.0 ms/轮）。详见 [`docs/v100-dev/README.md`](docs/v100-dev/README.md)。
