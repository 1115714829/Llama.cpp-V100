# AGENTS.md — 路由层（自动加载）

> 本文件只保留：**① 强制工作项 ② 路径地图**。其余一律外链，勿在此扩写。
> 新会话顺序：本文件 → [`docs/v100-dev/README.md`](docs/v100-dev/README.md) → 按需下表。

---

## 1. 强制项（必须做，不可省）

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
| [`v100-refs/`](v100-refs/) | 外部 V100 参考实现 | **只读** |
| `vllm/` `1cat-vllm/` `vllm-forkpoint/` | 对照源 | **只读，禁止改** |

### 研究档案深链（按需）

| 需要 | 打开 |
|---|---|
| 可信度 / 旧结论更正 | [`AUDIT-2026-09-20-dsh.md`](1cat-vllm-v100-study/AUDIT-2026-09-20-dsh.md) |
| 当下实况交接 | [`HANDOFF.md`](1cat-vllm-v100-study/HANDOFF.md) · [`HANDOFF-2026-09-22.md`](1cat-vllm-v100-study/HANDOFF-2026-09-22.md) |
| 吐字战役报告 | [`DECODE-CAMPAIGN-REPORT-2026-09-23.md`](1cat-vllm-v100-study/DECODE-CAMPAIGN-REPORT-2026-09-23.md) |
| 已突破 / 计划 / 失败 | [`2-已突破`](docs/v100-dev/2-已突破方向.md) · [`1-计划`](docs/v100-dev/1-计划突破方向.md) · [`3-失败`](docs/v100-dev/3-失败记录.md) |

---

## 3. 一句话现状

**B5 = 45.94 ms/轮**（tg 113/94/146，门 `f3edac19…`）；硬指标 tg≥150（≤37.0 ms/轮）。详见 [`docs/v100-dev/README.md`](docs/v100-dev/README.md)。
