# RULES · 规则全文（2026-09-26 起生效；修改须用户同意）

## R1 目标与判据

1. **验收** = L3 标准测试（[BENCH.md](BENCH.md)）的中位数：`TTFT_llama ≤ TTFT_REF` **且** `TPOT_llama ≤ TPOT_REF`。
2. **显存包络**：卡数 ≤ 4；每卡峰值 ≤ **14,827 MiB**（= vLLM `gpu-memory-utilization 0.905` × 16,384 MiB）。用更少的卡或更少的显存达标 = 更优。
3. REF 与 BASE 都在本体系下实测（阶段 0）；阶段 0 结束前不下任何优化结论。
4. 吐字：验收看 TPOT；llama 内部 A/B 看「每轮 ms」并同时报 AL，因为投机解码的 TPOT 随接受长度漂移。

## R2 安全边界（由服务器现状决定，不因重置而失效）

1. **只读**：`/root/llm/systemd/**`、`/root/llm/llama.cpp/**`、`/root/llm/ac922env/**`、`/root/llm/models/**`，以及本地 `v100-refs/**`。
2. **服务**：`vllm-1cat`（平时 inactive）、`llmscope`、`new-api`（active）保持原状态。`vllm-1cat` 只在用户授权的 REF 任务里 `systemctl start/stop`，测完必须回到 inactive；其他服务一律不碰。
3. llama.cpp 用的模型只从 `/mnt/3.84t/**` 加载；服务器可写区只有 `/root/llm/test/**` 与 `/mnt/3.84t/**`。
4. 不提 PR / issue，不向上游（ggml-org 等）推送。`git commit` 与 `git push` 只由主代理做（消息末尾写 `Assisted-by:`）。
   **推送 GitHub（用户 2026-09-26 授权）**：只推用户仓库 `github.com/1115714829/Llama.cpp-V100`——**这是公开仓库**。分支映射固定：本工作区 `master → workspace`；`llama.cpp` `feat/p3-decomp → feat/p3-decomp`（按 URL 推送，不在 `llama.cpp/` 里添加 remote）；`1cat-vllm-v100-study` `master → study-docs`。只做快进推送，禁止 force；每次推送前做密钥扫描（vLLM api-key 精确匹配 + 口令 / token 关键词），命中即停。
5. 密钥不打印、不落日志、不入库（vLLM 的 api-key 由脚本在运行时从单元文件读取；`olddoc/qwen-private/`）。

## R3 机器与等待

1. 同一时刻只允许一个上机作业（构建、测量、服务启停都算）。
2. **上机前三查**：`nvidia-smi`（目标卡显存 < 500 MiB、无计算进程）、进程（llama-server / vllm / test-backend-ops / cmake）、锁文件（`/root/llm/test/AGENT_LOCK`、`/tmp/LLAMA_BUILD_LOCK`）。任一不空 ⇒ 停手上报，不抢、不等。
3. **加锁**：三查与加锁合成一步 `bash tools/lock.sh <任务ID> <卡列表>`（输出 `LOCK_OK` 才能继续）；结束（含失败）必须 `bash tools/unlock.sh <任务ID>`。
4. **卡位与端口**：L3 用卡 0,1,3,4（与 vLLM 相同）；L1 / 单卡 L2 默认 GPU 2。测试 llama-server 固定端口 **8095**；8090 = llmscope、3000 = new-api、8000 = vllm-1cat、8080 = 生产 llama-server，一律不许占用（`llama-std.sh` 遇端口被占会拒绝启动）。
7. **同一时刻只有一个派发者**：执行者只由主代理通过 `tools/dsh-run.ps1` 启动；不要在 DSH 桌面端手动并行执行看板任务。
5. **等待**：长活用 `rjob.sh` 发令（立即返回）+ `rwait.sh` 阻塞（5 s 轮询终止符）；短命令末尾 `echo TERM_OK_<名>`；不裸睡、不叠发。
6. **PowerShell → ssh**：远端命令用单引号串，串内不出现双引号、`$(...)`、`#`；需要给 rjob 传环境变量时用 `rjob.sh <名> env K=V ... bash <脚本>`（不用引号）。逻辑复杂就写进 `sm70/tools/` 的脚本。

## R4 构建（主代理）

1. 唯一构建树 `/root/llm/test/v100-opt/llama.cpp`，必须与本地 `llama.cpp/` 逐字节一致（`tree-check.sh`；漂移即拒绝构建）。
2. 只用 `tools/build.sh`：每次输出到新目录 `/mnt/3.84t/sm70/libs/<commit>-<时间>/`，附 `BUILD_MANIFEST.txt`；绝不覆盖正在使用的库目录。
3. 进入 L3 的二进制必须对应一个 git commit（先提交再构建）。
4. 构建与测量互斥。

## R5 测量

1. **同源 A/B**：同一二进制，唯一变量是 env 开关。新代码路径一律 env 门控、默认关，直到被采纳。
2. **重复**：L1 每点 ≥ 3 个独立进程取中位数；L3 每臂 3 rep 取中位数。
3. **噪声带**：L1 固定 3%；L3 取阶段 0 同臂 rep 离散（(max − min) / median）的 2 倍。差异小于噪声带 = 无差异。
4. **权威口径**：L1 = `test-backend-ops`（test / perf）；L3 = 客户端 SSE 计时 + 响应 `timings`。`GGML_CUDA_OP_TIMING` 等内置计时探针只能粗筛，不能当判据。
5. **可追溯**：每个数字必须能追到原始日志行；报数带库目录与 `BUILD_MANIFEST`。
6. 采纳分两级：L1 + L2 通过 = **暂采纳**（默认开，成为下一轮开发的基线）；下一个 L3 里程碑 A/B 保住 = **确认**；没保住 = 撤销暂采纳（记录原因）。
7. 单卡结论带到 4 卡的边界见 BENCH.md「单卡结论能带到 4 卡的边界」；单卡只看相对变化，不外推绝对值。

## R6 正确性

1. **L1**：新增或修改的内核必须过 `test-backend-ops test`（对 CPU 参考、上游容差），并用真实形状跑。
2. **L2**：贪心门（BENCH.md）与 BASE 参考逐 token 一致 = 通过。
3. 不一致但 L1 通过：用 `llama-perplexity --kl-divergence` 对 BASE 做 KLD 对比，**交用户裁定**；主代理不得自行放宽门槛。
4. 先正确后速度：没过 L1 / L2 的速度数字作废。

## R7 代码

1. 只改 `llama.cpp/`；源码仅 ASCII；不新增 `tests/` 文件；代码风格遵循 `llama.cpp/AGENTS.md` 的约定。
2. **抄配方不搬成品**：先读 1cat 的做法（并行切分、融合、数据路径），再用我们的代码或 llama.cpp 既有内核实现。外来代码只有在接口和形状原生对得上时才可直接复用；禁止「搬核 + 胶水适配」。
3. 内核问题从内核解决，不以「关掉 / 回退 / 绕开」收场。判负的方向写明原因，代码留在 env 门后或回退。
4. **不做启动参数调参**：llama.cpp 启动参数只允许取 BENCH.md 对齐表里的值。

## R8 记录

1. `STATE.md` 是唯一的当前真相（≤ 1 页，主代理维护）；`LEDGER.md` 追加式，一行一个决策（含否决）；`BOARD.md` 是任务看板；其余一律进 `results/`。
2. 证伪也要记；历史行不删，只追加更正。
3. 旧档案只当线索；引用其中任何数字前，必须在本体系下复测。
