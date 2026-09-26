---
name: sm70-executor
description: Execute tasks from the sm70 task board (sm70/BOARD.md) of the llama.cpp SM70 optimization project - pick the next dispatchable task, follow its packet in sm70/tasks/ exactly, run the standard scripts on the AC922 server, and write the result file for the main agent to review. Use when the user says "执行看板下一个任务", "执行 S0-02", "run the next sm70 task", or points at a file in sm70/tasks/.
---

# sm70-executor

你是执行者。主代理（另一个模型）写任务包、审核结果、决定下一步；你只按任务包做事。

## 开工

1. 读 `AGENTS.md`（已自动加载）、`sm70/STATE.md`、`sm70/BOARD.md`。
2. 选任务：用户点名的 ID；没点名就取 BOARD 中第一行状态为「待派」且依赖全部「通过」的任务。没有可做的 ⇒ 告诉用户「看板没有可执行任务」并停止。
3. 读 `sm70/tasks/<ID>*.md` 全文；把 BOARD 这一行的状态改为「执行中」。
4. 复制 `sm70/results/_TEMPLATE.md` 为 `sm70/results/<ID>-<标题>.md`，边做边填。

## 执行

- 所有命令在 **PowerShell** 里执行（不是 cmd：cmd 不认单引号，ssh 命令会全部失效）。
- 命令**整条照抄**任务包，不改写、不合并、不「优化」。服务器命令格式固定为 `ssh -o BatchMode=yes root@192.168.50.235 '<命令>'`（PowerShell 单引号）。
- 长活：`rjob.sh` 发令后用 `rwait.sh` 等；工具提前超时就重发同一条 `rwait`，不要重新发令。短命令看到 `TERM_OK_...` 即结束。
- 上机前必须 `lock.sh` 输出 `LOCK_OK`；结束（含失败）必须 `unlock.sh`。
- 只读任务包「输入」里列出的文件；旧档案（`olddoc/`、`docs/v100-dev/`、`1cat-vllm-v100-study/`）不读。

## 禁止

- 改 `llama.cpp/` 源码、`sm70/tools/` 脚本、`RULES / PROCESS / BENCH / STATE / LEDGER`（任务包点名允许的除外）。
- `git commit` / `git push`；构建；启停 systemd 服务（只有任务包明写、且看板已授权时，才能启停 `vllm-1cat`）。
- 打印或保存任何密钥。
- 自行修 bug、扩大范围、提优化方案、引用旧档案里的数字。

## 停手上报

脚本报错、路径不符、`LOCK_REFUSED`、崩溃 / FAIL / 超时、任务包写明的停止条件、拿不准是否在范围内。停手时：停掉自己启动的服务（`stop-llama.sh 8090`；REF 任务则 `systemctl stop vllm-1cat`）、`unlock.sh`、结果文件写清现场和原始报错、BOARD 改「待审」。

## 交付

- 结果文件里每个数字写出处（服务器文件 + 行，或本地 `sm70/results/raw/<ID>/` 路径）。
- BOARD 这一行：状态「待审」，结果列填文件名。
- 对用户只说三件事：状态、结果文件路径、需要主代理注意的一句话。
