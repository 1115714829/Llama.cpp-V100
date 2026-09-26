# BENCH · 标准测试（修改须用户同意）

## 三层

| 层 | 目的 | 硬件 | 模型 | 耗时 | 工具 |
|---|---|---|---|---|---|
| **L1 算子级** | 内核正确性与速度 | 单卡 GPU 2 | 无（真实形状文件） | 秒级 | `tools/op-perf.py` |
| **L2 单卡端到端（日常主回路）** | 贪心门 + 相对速度 | 单卡 GPU 2 | IQ1_S + F16 草稿，CTX 65536 | 分钟级 | `tools/gate.py`、`tools/bench-256k.sh` |
| **L3 256K 标准测试** | 基线、里程碑、验收 | 卡 0,1,3,4 | Q8_0 + F16 草稿 / vLLM 原样 | ≈15 min/臂 | `tools/bench-256k.sh` |

## 单卡结论能带到 4 卡的边界

IQ1_S 与 Q8_0 是同一个基座（Qwen3.8-27B：24 Q 头 / 4 KV 头、head_dim 256、16 层全注意力 + 48 层 GDN、同一个草稿），所以内核内部流程可以在单卡上迭代。边界如下：

| 类别 | 处理 |
|---|---|
| **可以直接带过去** | 注意力 / GDN / 归一化 / 采样 / 草稿等内核的内部实现；投机解码流程；单设备的主机开销 |
| **要换形状复核** | 与并行度有关的改动（KV 切分、block 数、占用率）：单卡每层 4 个 KV 头，TP4 每卡只有 1 个（6 个 Q 头）⇒ 必须在 L1 用 **TP4 每卡形状**再测一遍 |
| **要看路由日志** | 按头数、长度、批大小选路的内核：单卡与 4 卡必须走同一条路径（服务端开 `GGML_CUDA_FA_KERNEL_DEBUG=1`，比对 `[FAK]` 行），否则单卡测的是另一段代码 |
| **单卡测不到** | 权重矩阵乘（IQ1_S 与 Q8_0 是不同内核 ⇒ 用 L1 的 Q8_0 形状测）；多卡通信与 meta 调度（只能 L3）；256K 长度（单卡端到端最多 64K，256K 的注意力用 L1 真实形状测） |
| **数字** | 单卡只看相对变化（开 / 关的比值），绝对值不外推到 4 卡 |

## L2 单卡端到端

- 服务：`llama-std.sh`，`CARDS=2 CTX=65536 MODEL=/mnt/3.84t/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ1_S.gguf`，外加 `GGML_CUDA_FA_KERNEL_DEBUG=1`（路由自证）。
- 负载：`/root/llm/test/sm70/prompt-l2.txt`（= 256K prompt 的前 270,000 字节，约 5.5 万 token，约占 ctx 的 85%）；采样、加盐、max_tokens 128 与 L3 相同；每臂 3 rep。
- 门：见下文「L2 贪心门」的单卡档。

## L3 负载（两个引擎完全相同）

- **prompt**：`/root/llm/test/sm70/prompt-256k.txt`（S0-02 定稿并记录 sha256；约 236K token ≈ ctx 262144 的 90%）。
- 每个 rep 在开头加盐 `run <n>\n`（客户端内置）：两个引擎同 rep 号字节一致；不同 rep 之间没有可复用的前缀，前缀缓存两边都命中不了。
- 请求：`/v1/chat/completions` 流式；temperature 0.7、top_p 0.8、top_k 20、repetition / repeat penalty 1.05；max_tokens 128；thinking 开。
- 每臂 3 rep，rep 之间服务不重启。
- 每臂启动服务前执行 `sync; echo 3 > /proc/sys/vm/drop_caches`（AC922 的显存以 NUMA 节点暴露，页缓存不能占进显存）。

## L3 指标（`tools/summarize.py`）

| 指标 | 定义 |
|---|---|
| TTFT | 请求发出到收到第一个内容或思考 token（客户端计时） |
| pp t/s | prompt_tokens / TTFT |
| TPOT | (t_end − t_first) / (completion_tokens − 1)，毫秒 |
| AL、每轮 ms（仅 llama） | 来自响应 `timings`：rounds = draft_n / NMAX；AL = predicted_n / rounds；每轮 ms = predicted_ms / rounds（公式在 S0-04 核实） |
| 显存 | `nvidia-smi` 每 1 s 采样，每卡峰值 MiB |

报数格式：每 rep 一行 + 中位数行 + `SPREAD` 行 + 各卡 `MEM_PEAK`。

## L3 配置对齐表（llama 启动参数只能取这里的值）

| 项 | vLLM（`vllm-1cat` 单元原样） | llama.cpp（`tools/llama-std.sh`） |
|---|---|---|
| 权重 | Qwen3.8-27B-FP8（e4m3，动态激活量化） | `Qwen3.8-27B-Q8_0.gguf` |
| 草稿 | DFlash2 BF16，`dtype half` | `Qwen3.8-27B-DFlash2-F16.gguf` |
| 草稿长度 | 1cat 默认（S0-05 从启动日志读出） | `--spec-draft-n-max 7`（S0-05 后对齐） |
| 草稿采样 | `probabilistic` | llama 现有实现（差异项，报 AL） |
| 并行 | TP4，`CUDA_VISIBLE_DEVICES=0,1,3,4` | `--split-mode tensor --tensor-split 1,1,1,1`，同一组卡 |
| 上下文 / 并发 | 262144 / `max-num-seqs 1` | `--ctx-size 262144` / `--parallel 1` |
| 预填充分块 | `max-num-batched-tokens 2048` | `--batch-size 2048 --ubatch-size 2048` |
| KV 缓存 | `fp8_e5m2` | `q8_0`（K 与 V） |
| 注意力 | `FLASH_ATTN_V100` | `--flash-attn on` |
| 思考 | enable_thinking、preserve_thinking、reasoning_effort xhigh | `--reasoning on --reasoning-preserve --reasoning-effort xhigh` |
| 前缀缓存 | 开 | 默认（加盐使两边都不命中） |
| 显存 | `gpu-memory-utilization 0.905` | 自然占用，须 ≤ 14,827 MiB/卡 |
| 采样 | 由请求携带 | 由请求携带（服务端不设采样参数） |

**已知差异项**（报告必须列出）：权重格式 FP8 vs Q8_0（同为 8 bit）；KV `fp8_e5m2` vs `q8_0`；草稿采样方式。

## L2 贪心门

- 接口 `/completion`；prompt 取 `tools/gate-prompts.txt` 的 3 行；temperature 0、top_k 1、seed 0、n_predict 128、cache_prompt false、return_tokens true；投机开。
- 两个档位：**单卡** = IQ1_S（`/mnt/3.84t/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ1_S.gguf`）、GPU 2、CTX 32768；**4 卡** = Q8_0、卡 0,1,3,4、CTX 262144。
- 判据：token 序列 sha（`tok_sha`）与 BASE 参考一致 = PASS（`gate.py --ref <参考文件>` 直接给出 `GATE_VERDICT`）。
- BASE 参考在 S0-06 生成，并先验证 BASE 自己两次运行结果一致。

## L1 算子级

- **真实形状**：`test-export-graph-ops -m <gguf> -o <文件> -ub <n> -c <ctx> -ctk q8_0 -ctv q8_0` 导出（单设备、全部头数）；TP4 每卡形状的换算规则由主代理在 S0-07 给出。
- **正确性**：`LIBS=<库> python3 op-perf.py <形状文件> <臂名> 1 test`
- **性能**：`LIBS=<库> python3 op-perf.py <形状文件> <臂名> 3 perf`（3 个独立进程取中位数）
- 原始输出含 `[FAK]` 内核选择行（脚本自动设 `GGML_CUDA_FA_KERNEL_DEBUG=1`）。

## 服务器上的标准路径

| 路径 | 内容 |
|---|---|
| `/root/llm/test/sm70/tools/` | 标准脚本（本地 `sm70/tools/` 为准，`push.ps1` 上传） |
| `/root/llm/test/sm70/logs/` | rjob 日志、构建日志、服务日志 |
| `/root/llm/test/sm70/runs/<tag>/` | L3：`stress.txt`、`gpu.csv`、`summary.md`、`meta.txt` |
| `/root/llm/test/sm70/gate/` | L2 输出与参考 |
| `/root/llm/test/sm70/op/<arm>/` | L1 原始输出 |
| `/mnt/3.84t/sm70/libs/<commit>-<时间>/` | 构建产物（带 `BUILD_MANIFEST.txt`）；`base0/` = 冻结的 BASE |
