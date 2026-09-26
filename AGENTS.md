# AGENTS.md — llama.cpp SM70 专项优化（2026-09-26 重置）

> 每次会话自动加载。只放目标、角色、硬规则和入口；细则全在 [`sm70/`](sm70/README.md)。
> 2026-09-26 之前的规则与结论已归档、不再生效（见文末「旧档案」）。

## 目标

在 V100（SM70）上对 llama.cpp 做专项优化：**同模型、同硬件、同负载**下，预填充 **TTFT** 与吐字 **TPOT** 都不输 **1cat-vLLM**，显存不超过它的部署包络。
1cat-vLLM 有两重身份：**性能标尺**（服务 `vllm-1cat`，本体系下实测为 REF）和 **配方来源**（源码只读 `v100-refs/1cat-vllm/`）。

## 开工顺序

1. 本文件 → 2. [`sm70/STATE.md`](sm70/STATE.md)（当前真相，一页）→ 3. [`sm70/BOARD.md`](sm70/BOARD.md)（任务看板）
- 执行者（DSH）另读自己的任务包 `sm70/tasks/<ID>*.md`，按技能 `.dsh/skills/sm70-executor` 执行。
- 规则全文 [`sm70/RULES.md`](sm70/RULES.md) · 流程 [`sm70/PROCESS.md`](sm70/PROCESS.md) · 标准测试 [`sm70/BENCH.md`](sm70/BENCH.md)

## 角色

| 角色 | 负责 |
|---|---|
| 主代理（Cursor） | 定序、写任务包、**用 `sm70/tools/dsh-run.ps1` 以 headless 方式驱动 DSH 执行**、审核、内核设计与关键实现、构建、`git commit`；维护 STATE / BOARD / LEDGER |
| 执行者（DSH 模型） | 按任务包执行测量、跑标准脚本、只读调研；只改任务包点名的文件；结果写 `sm70/results/` |
| 用户 | 授权服务启停和标准变更；不需要手动操作 DSH |

## 硬规则（摘要；全文见 RULES.md，冲突以 RULES.md 为准）

1. 服务器 `root@192.168.50.235`：只读 `/root/llm/{systemd,llama.cpp,ac922env,models}`；可写 `/root/llm/test/**`、`/mnt/3.84t/**`。`vllm-1cat` / `llmscope` / `new-api` 保持原状态；`vllm-1cat` 只在用户授权的 REF 测量中启停，测完恢复。
2. 绝不 `git push`、不加 remote、不提 PR；`git commit` 只由主代理做。
3. 同一时刻只有一个上机作业；上机前三查（GPU / 进程 / 锁），加锁 `/root/llm/test/AGENT_LOCK`，结束删锁。
4. 只做代码和内核优化，不做启动参数调参：llama.cpp 启动参数只取 BENCH.md 对齐表里的值。
5. 先正确后速度：没过 L1/L2 正确性门的速度数字作废；单算子收益必须在 256K 标准测试（L3）里保住才算。
6. 只认本体系下的实测数字；旧档案里的数字复测后才能引用。
7. 抄配方，不搬成品：学 1cat 的做法，用我们自己的代码实现；不许「搬外来内核 + 写胶水适配」。

## 旧档案（默认不读）

`olddoc/`（旧规则在 `olddoc/archive-2026-09-26/`）、`docs/v100-dev/`、`1cat-vllm-v100-study/`、`.dsh/tmp/`：只作线索，任务包点名时才按行号查阅。
`llama.cpp/AGENTS.md`、`llama.cpp/CLAUDE.md` 是上游的贡献政策：其 PR 流程不适用（私有 fork 豁免），代码风格约定照样遵循。
`olddoc/qwen-private/QWEN.md` 含口令，禁止提交和外传。
