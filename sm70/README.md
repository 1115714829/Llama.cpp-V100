# sm70 · 标准工作体系

| 文件 | 作用 | 谁写 |
|---|---|---|
| [`STATE.md`](STATE.md) | 当前真相：基线、差距、工作队列、待用户决定 | 主代理 |
| [`BOARD.md`](BOARD.md) | 任务看板 | 主代理；执行者只改自己任务的状态（待派 → 执行中 → 待审） |
| [`LEDGER.md`](LEDGER.md) | 决策账：一行一个决策，追加式，含否决 | 主代理 |
| [`RULES.md`](RULES.md) · [`PROCESS.md`](PROCESS.md) · [`BENCH.md`](BENCH.md) | 规则 · 流程 · 标准测试 | 主代理（改动须用户同意） |
| `tasks/` | 任务包（`_TEMPLATE.md` 为模板） | 主代理 |
| `results/` | 执行结果；原始产物放 `results/raw/<ID>/` | 执行者 |
| `tools/` | 标准脚本；本地为准，`push.ps1` 上传到服务器 `/root/llm/test/sm70/tools/` | 主代理写，执行者验证 |

## 一轮协作

```
主代理：写任务包，看板置「待派」
  → 用户：在 DSH 说「按 sm70-executor 执行看板下一个任务」
  → 执行者：做完写 results/<ID>-*.md，看板置「待审」
  → 用户：告诉主代理「<ID> 待审」
  → 主代理：审核 → 通过 / 驳回；更新 STATE、LEDGER；派下一批
```

## 服务器上的目录

`/root/llm/test/sm70/{tools,logs,runs,gate,op,shapes}`；构建产物在 `/mnt/3.84t/sm70/libs/<commit>-<时间>/`，冻结的 BASE 在 `/mnt/3.84t/sm70/libs/base0/`。
