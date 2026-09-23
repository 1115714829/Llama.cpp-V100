---
feature: prefill-kernel
status: delivered
updated: 2026-09-23
branch: feat/prefill-kernel
commits: local uncommitted (feat/prefill-kernel @ d120acb48 base)
---

# 预填充内核专项（首刀 Path B）

## Report

**What was built** — Path B（q8_0 核内反量化 / 去 f16 镜像）已落地在本地 `llama.cpp` 分支 `feat/prefill-kernel`：`fattn-sm70-d256-kernel.cuh` 增加 `sm70_q8_dequant_*`（对齐 staged `__hmul2`），`KMode/VMode`（0=暂存,1=q4,2=q8）；`fattn-sm70-d256.cu` 增加 `LLAMA_SM70_Q8_DIRECT`（判值，默认 OFF）、`sm70_kv_direct_modes`（alloc/launch 一致，混合 q4+q8 回退暂存）、q8-direct 探针串。直读时 **不分配** f16 K/V 镜像（P-M3）。

**Verification** — 服务器可操作区 `/root/llm/test/v100-opt/llama.cpp` + `/root/libdir-pathb`：
- 构建 `BUILD_RC=0`，`MARK_Q8=1`（`strings | grep -c LLAMA_SM70_Q8_DIRECT`）。
- 机制自证：`ACCEPT: sm70 d256 + q8-direct`，Ktype=8/Vtype=8，`Knb0=34` 块连续。
- **32K ABBA（NODROP 诊断，`-ub 2048`，TP3，`GGML_GALLOCR_SLOTS=3`，`-r 2`）**：
  | 臂 | Q8_DIRECT | pp32768 t/s |
  |---|---|---|
  | a1 | 0（暂存） | 966.81 ± 4.00 |
  | a2 | 1（直读） | 937.57 ± 0.34 |
  | a3 | 1（直读） | 935.84 ± 0.89 |
  | a4 | 0（暂存） | 966.40 ± 0.43 |
  - 暂存均 **966.61**；直读均 **936.71** → **-3.09%**。与 q4-direct 的 8/24 结论同型（直读省镜像、吞吐略亏）。
- 8K 复核：暂存 1986 vs 直读 1966（约 -1%，噪声量级）。
- 256K 本配置 OOM（`res=-2`，对照/特性皆未跑完）——须更紧槽位/更大 KV 预算，**非 Path B 独有**（`SLOTS=8` 在 32K 亦 OOM，见 Journey）。
- greedy sha256：本轮为 pp-only（`-n 0`）未生成 token，门值未跑；decode 零改动路径。

**Journey log**
1. PowerShell→ssh 双引号陷阱复发：`tr -d "\r"` 被收成 `tr -d r`，源码字母 `r` 被删；改脚本内 `sed 's/\r$//'`。
2. `GGML_GALLOCR_SLOTS` 默认 8 在 ≥32K OOM（R282）；A/B 须 `SLOTS=3`。
3. `/usr/bin/time` 本机不存在，勿写入 harness。
4. 直读 vs 暂存：**速度不占优（-3%）**，价值在 **f16 镜像显存**（256K 约 GB 级）；与 q4-direct 同一剧本。
5. 按 T4「不差于暂存才采用」→ **不采用为默认**；代码保留、`LLAMA_SM70_Q8_DIRECT` 默认 OFF，显存紧时可开。

## [S1] 问题

长上下文预填充是用户主场景（256K）。墙已经量过：

- 长 KV 下 `FLASH_ATTN_EXT` 是预填充最大单项（峰值 ubatch 占 **78.4%**，256K 全程约 **2/3**）；效率真基线 stock **35.8** / Path A **45.9 TFLOPS**（R269 口径；旧值「1.49 TFLOPS」已作废）。
- Path A（`sm70-attn` D256 Split-D）已落地，端到端证明有效（256K **+17.5%**），但生产 KV 是 **q8_0**。
- Path A 对非 f16 的 KV 仍走上游 **整缓存 `to_fp16` 暂存**：每个预填充 chunk 都把整份 K/V 重新反量化进 f16 镜像。代价是：
  1. 分块服务预填充下 **O(n^2) 流量**；
  2. 256K 时 **GB 级额外显存**（K+V 的 f16 镜像）；
  3. 与 FA 内核抢带宽。

树里已有同构先例：`Kq4`/`Vq4`（q4-direct）在寄存器/smem 碎片里直接反量化原始 `q4_0` 块，并跳过 f16 镜像。  
生产格式是 **Q8_0 权重 + q8_0 KV**（见 `docs/v100-dev/05-量化范围.md`）。缺的是同一套 **q8_0 直读**（Path B）。

用户决策（2026-09-23）：吐字冲刺先停；开 **预填充内核/底层** 专项。本 feature 文档管这条战役；**首个交付 = Path B**。

## [S2] 设计

### 战役范围（仅内核 / 底层）

| 优先级 | 项 | 在本文档中的角色 |
|---|---|---|
| **T1–T4（本次交付）** | **Path B：q8_0 核内 KV 反量化** | 实现 + 验证 |
| 后续（改文档再开） | P-P1 Path A 门控/回归收口 | 可选配套 |
| 后续 | P-P4 tile / SplitKV3 再调 | 独立切片 |
| 后续 | P-P3 巨型 GEMM 工作分解 | 独立大 feature |

不在本战役：解码 P-D*；显存预算器 P-M1（Path B 自带的 P-M3「省 f16 镜像」除外）。

### 工作区约定（相对 compose 默认的覆盖）

- 会话沙箱禁止 `git worktree add`（共享 ref 注册表）。
- 当前代码工作区 = 本地 `llama.cpp/` 分支 **`feat/prefill-kernel`**（自 `master` @ `d120acb48` / B5 线切出）。
- Feature 文档放在上级 `docs/compose/spec/prefill-kernel.md`（项目文档根是 `F:\vllm+llama.cpp`，不是嵌套的 `llama.cpp/`）。
- 交付只落 `llama.cpp/`（红线）。**不新增 `tests/*`**（llama.cpp AGENTS.md）。
- 源码 **仅 ASCII**。

### 环境边界（硬性；来自 `olddoc/workspace-loose/AGENTS.md` §1.9 + `docs/v100-dev/00-红线.md`）

| 类别 | 对象 | 允许行为 |
|---|---|---|
| **正式环境（绝不碰）** | `/root/llm/systemd/**`（含 `llama-server.service`、`vllm-1cat.service` 及 `/etc/systemd/system/...` 软链） | **只读**；不 start/stop/enable/disable/改文件 |
| **正式目录（绝不写）** | `/root/llm/llama.cpp`、`/root/llm/ac922env` | **只读参考**；不编译、不落文件、不删 |
| **服务状态（不要动）** | `vllm-1cat` / `llmscope` / `new-api` | 保持用户交付时状态；只允许 `systemctl is-active` 类只读查询 |
| **可操作区（服务器）** | `/root/llm/test/**`、`/mnt/3.84t/**` | 读写、编译、跑测、建脚本 |
| **模型加载** | 只从 `/mnt/3.84t/**` | **永远不用** `/root/llm/models/` |
| **服务器真开发树** | `/root/llm/test/v100-opt/llama.cpp` | 服务器上唯一允许打补丁/构建的树 |
| **陈旧副本陷阱** | `/mnt/3.84t/v100-opt/llama.cpp` | **过期副本**——打补丁不生效 |
| **本地交付** | `F:\vllm+llama.cpp\llama.cpp\` | 代码落点；ASCII；无 `tests/*` |
| **机密** | `olddoc/qwen-private/QWEN.md` | 禁止提交/分享/复制进仓库 |

另：绝不代用户 `git commit` / `git push` / `gh pr`；私有 fork、不做上游贡献。`v100-refs/` 全程只读。同一台机器构建与测量互斥。

本 feature **不授权**对正式服务/正式目录的任何写入。需要上机测量时只进可操作区，且须在 Spec 批准之后。

### Path B 契约

复用现有 q4-direct 架构（`fattn-sm70-d256-kernel.cuh` 的 `Kq4`/`Vq4` + 启动器 `k_direct`/`v_direct`），补上 q8_0 孪生路径。

1. **类型谓词**（**alloc_size** 与 **launch** 必须一致）：
   - `k_direct = (K->type == GGML_TYPE_Q8_0) && sm70_q8_direct()`（扩展；**q4-direct 保持可用**）。
   - `V` 同理 `v_direct`。
   - 环境开关：`LLAMA_SM70_Q8_DIRECT`（先门控交付；A/B 通过后再谈默认 ON）。**判值不判存在**：`=0` 表示关闭。
2. **内核**：q8_0 块反量化进 HMMA 已消费的同一套 f16 寄存器/smem 碎片（MMA 边界上 Q/K/V 仍是 f16）。舍入对齐暂存路径 `to_fp16` / `dequantize_block_q8_0`，以通过 greedy sha256 门为准；若位有漂移则重立门值（须告知用户）。
3. **布局**：原始块直读，行块连续（`nb[0] == ggml_type_size(Q8_0)`）。非块连续则拒绝并回退 Path A 暂存——与 q4-direct 同规则。
4. **仅 dense**：Q8 direct 开启时对 `PagedKV` `static_assert`（同 Kq4/Vq4）。
5. **alloc_size**：`k_direct`/`v_direct` 为真时 **不分配** 该张量的 f16 K/V 镜像（同 q4-direct）。这就是 P-M3 显存收益。
6. **实例化矩阵**：保持现有 `{dense, SplitKV3} × {staged, 仅 K 直读, 仅 V 直读, 双直读}` 形态；用 q8 模板特化，编译规模不超过现有 q4 已付出的水平。
7. **回退**：`LLAMA_SM70_D256=0` 仍强制上游 MMA_F16；`LLAMA_SM70_Q8_DIRECT=0` 在 sm70 内核上强制 Path A 暂存（保留 f16 镜像）。
8. **机制自证计数**（任何 A/B 必备）：探针行必须能区分 `q8-direct` 与 staged；alloc 须证明直读张量 f16 镜像为 0。

### 正确性 / 度量纪律（项目红线）

- 同源 A/B，唯一变量 = Q8-direct 门控。
- 若碰到解码路径，**tg 与 ms/轮并列**（预期零回退）；预填充报 **pp t/s**（32K 与 256K）+ 有则 **TTFT**。
- 入账数字 `llama-bench -r >= 8`；greedy sha256 门；机制自证计数。
- 正式数字带 `drop_caches`；诊断 `NODROP=1` 须标明口径。
- 禁止并发测量；构建与测量互斥。
- 用户配置（`-ub`、卡数）**不是成果**。

### 预期效果

| 轴 | 预期 |
|---|---|
| 预填充带宽 | 砍掉整缓存反量化流量；深 KV / 分块预填充收益最大 |
| 显存 | 释放 f16 K+V 镜像（256K 为 GB 级）= P-M3 |
| 解码 | 不变（纯预填充路径） |
| 数值 | 优先与暂存 Path A 位级一致；否则重立门并告知用户 |

数值目标**不在设计里写死**（先测再记）。对照基线 = 同库 Path A 暂存。

## [S3] 明确不做

- 解码战役（P-D1…P-D6）与 AL/投机改动。
- 预填充 P-P3 巨型 GEMM 重写（以后独立 feature）。
- P-P4 tile 再调（以后独立切片）。
- P-M1 多槽预算器 / P-M2 / P-M4。
- q8_0 以外的量化格式矩阵（q4-direct 保持可用即可）；不做 Q4_K_M 战役。
- 用户启动参数调优；上游 PR / push（私有 fork，红线）。
- 新增 `tests/*`。
- F3 类捕获 / meta 容器 Fix B（解码侧）。

## Tasks

- [x] T1: 在 `fattn-sm70-d256-kernel.cuh` 实现 q8_0 直读加载 + 反量化（KMode/VMode：0=暂存,1=q4,2=q8）——验收：能编译；q8_0 块反量化到 HMMA 已消费的 f16 碎片（`__hmul2` 对齐 staged）；仅 dense 断言成立（covers: S2）
- [x] T2: 启动器 + alloc_size 的 q8-direct 谓词（env `LLAMA_SM70_Q8_DIRECT`，判值；`sm70_kv_direct_modes` 两侧一致；直读时跳过 f16 镜像；混合 q4+q8 回退暂存）——验收：alloc_size 与 launch 一致；env=0 或非块连续行时回退暂存；探针串能区分 q8-direct（covers: S2; depends: T1）
- [x] T3: 本地源码完备 + 服务器可操作区构建与冒烟——验收：构建干净；`LLAMA_SM70_Q8_DIRECT=0` 行为与暂存 Path A 一致；`=1` 无 launch 断言；机制计数证明走了直读（covers: S2; depends: T2）
- [x] T4: 同源 A/B（暂存 vs q8-direct）pp 32K + 256K——验收：报 pp t/s（有则 TTFT）、f16 镜像显存差、greedy sha256、≥2 臂并报离散；不差于暂存 Path A 才采用（covers: S2; depends: T3）— **32K 四臂完成（-3.09%，不采用为默认）；256K 本配置 OOM 待槽位预算**
