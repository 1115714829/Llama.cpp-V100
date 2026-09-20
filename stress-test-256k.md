# stress-test-256k.md — 256k 长上下文端到端压测：原版 vs C4

> 目的：按用户要求，用**同一个模型文件、同一份 256k 上下文配置**，对比原版 llama.cpp 与 C4 版
> （`mmvq.cu` 新增 `MMVQ_PARAMETERS_VOLTA`）在真实长上下文服务场景下的端到端表现。
> 日期 2026-09-20；机器 AC922（6x Tesla V100-SXM2-16GB），GPU 2/5；tensor 双卡（`--split-mode tensor --tensor-split 1,1`）。
> 数据来源：AC922 journal（原始行见 §3），本报告**不采信口头汇总**。

## 1. 结论摘要（保守版）

**能确定的**：C4 的收益由 `llama-bench -p 512 -n 128`（无 MTP、无采样方差）确定：**tg128 36.23 -> 37.4 t/s（+3.4%）**，pp512 不变。

**本 256k 实验不能确定 C4 的净收益**，原因见 §5：原版自己两次跑的 prefill 相差 8.0%，
而 C4 落在两次之间；同时 MTP 接受率在三次运行间大幅波动（0.551 / 0.587 / 0.822）。详见 §5。

## 2. 实验设置（同文件同上下文）

两侧 systemd unit 除 binary 与 port 外**逐字相同**（unit 全文见 §3.4）：

| 项 | 值 |
|---|---|
| 模型 | `/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf`（9.14 GiB, Q2_K, arch `qwen35`） |
| ctx | `--ctx-size 262144`，`--parallel 1` |
| GPU | `CUDA_VISIBLE_DEVICES=2,5`，`--n-gpu-layers 999`，`--split-mode tensor --tensor-split 1,1` |
| attention / KV | `--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0` |
| 投机解码 | `--spec-type draft-mtp --spec-draft-n-max 4`（MTP4） |
| 采样 | `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0` |
| 其它 | `--reasoning on --reasoning-preserve --slot-save-path /home/llama-slot-cache` |
| 原版 binary | `/root/llm/llama.cpp/bin/llama-server`（Sep 10 10:41），unit `llama-server-test`，port 8081 |
| C4 binary | `/root/llm/test/v100-opt/llama.cpp/build/bin/llama-server`（09:46, gcc-toolset-12 + CUDA 12.4），unit `llama-server-c4`，port 8082 |
| 请求 | `/tmp/req256k.json`（1,165,950 B），prompt = `/tmp/prompt256k.txt`（1,165,900 B = 13100 x 那句 20-token 英文句） |

**实际处理的 token 数：三次运行都是 `prompt_n = 248901`、`total = 249029`（128 生成）** -> 可比的。

## 3. 原始证据

### 3.1 原版跑第 1 次（PID 651359，journal 09:08:01 启动）
```
09:18:28  n_gen = 100, tg = 30.23 t/s, tg_3s = 30.53 t/s
09:18:29  prompt eval time = 587182.70 ms / 248901 tokens ( 2.36 ms per token, 423.89 tokens per second)
09:18:29         eval time =   4398.28 ms /   128 tokens ( 34.63 ms per token,  28.87 tokens per second)
09:18:29        total time = 591580.99 ms / 249029 tokens
09:18:29  draft acceptance = 0.58667 (   88 accepted /  150 generated), mean len =  3.32
```

### 3.2 原版跑第 2 次（PID 670192，journal 09:32:53 启动；加了 `--slot-save-path`）
```
09:43:36  n_gen = 101, tg = 31.87 t/s, tg_3s = 32.19 t/s
09:43:37  prompt eval time = 638302.37 ms / 248901 tokens ( 2.56 ms per token, 389.94 tokens per second)
09:43:37         eval time =   4463.12 ms /   128 tokens ( 35.14 ms per token,  28.46 tokens per second)
09:43:37        total time = 642765.49 ms / 249029 tokens
09:43:37  draft acceptance = 0.55063 (   87 accepted /  158 generated), mean len =  3.17
```

### 3.3 C4 跑 1 次（PID 688529，journal 09:56:44 启动）
```
10:07:01  n_gen = 115, tg = 37.40 t/s, tg_3s = 37.72 t/s
10:07:01  prompt eval time = 611062.20 ms / 248901 tokens ( 2.46 ms per token, 407.33 tokens per second)
10:07:01         eval time =   3401.51 ms /   128 tokens ( 26.78 ms per token,  37.34 tokens per second)
10:07:01        total time = 614463.71 ms / 249029 tokens
10:07:01     graphs reused =         28
10:07:01  draft acceptance = 0.82203 (   97 accepted /  118 generated), mean len =  4.23
```
同一请求的响应 JSON `/tmp/resp256k.json` 里也带同样的 timings，可交叉核对：
`"prompt_per_second":407.3251462780712, "predicted_per_second":37.33635944036619, "draft_n":118, "draft_n_accepted":97`

### 3.4 两侧 systemd unit（除 binary/port/alias 外逐字相同）

`/etc/systemd/system/llama-server-c4.service`:
```
Environment="CUDA_VISIBLE_DEVICES=2,5"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64"
ExecStart=/root/llm/test/v100-opt/llama.cpp/build/bin/llama-server \
  --model /root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf \
  --alias Qwen3.8-27B-Q2_K_XL-C4 --ctx-size 262144 --n-gpu-layers 999 \
  --split-mode tensor --tensor-split 1,1 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --reasoning on --reasoning-preserve \
  --slot-save-path /home/llama-slot-cache \
  --host 0.0.0.0 --port 8082 --metrics --api-key-file /etc/llama-server/api-keys
```
`llama-server-test.service` 只有三处不同：`ExecStart=/root/llm/llama.cpp/bin/llama-server`、
`--alias Qwen3.8-27B-Q2_K_XL`、`--port 8081`。

### 3.5 参考：原版**不带 MTP**、短 prompt（PID 642581，08:56:49 启动）
```
08:58:00  prompt eval time =   204.04 ms /       9 tokens ( 22.67 ms per token, 44.11 tokens per second)
08:58:00         eval time =  2939.65 ms /     128 tokens ( 23.15 ms per token, 43.20 tokens per second)
```
**注意：这一行不能与 §3.1/3.2 直接比** —— 上下文长度（9 tok vs 256k）、有无 MTP、采样内容都不同。

## 4. 结果对比（三次运行，全部同一 prompt / 同一 248901 token）

| 运行 | binary | prefill t/s | tg t/s | total s | MTP 接受率 | mean len |
|---|---|---|---|---|---|---|
| 原版 #1 (09:08) | orig | **423.89** | 28.87 | 591.58 | 0.58667 | 3.32 |
| 原版 #2 (09:32) | orig | **389.94** | 28.46 | 642.77 | 0.55063 | 3.17 |
| **C4 (09:56)** | C4 | **407.33** | **37.34** | **614.46** | **0.82203** | **4.23** |
| C4 vs 原版#1 | | -3.9% | **+29.3%** | +3.9% | +40.1% | |
| C4 vs 原版#2 | | +4.5% | **+31.2%** | -4.4% | +49.3% | |

**原版自身两次的漂移**：prefill **-8.0%**（423.89 -> 389.94）、tg -1.4%、接受率 -6.1%。

## 5. 诚实的问题：本实验测不出 C4 的净收益

### 5.1 prefill 差不可判定
原版两次在**完全相同配置**下 prefill 相差 8.0%（423.89 vs 389.94 t/s），说明 256k 长 prefill
在本机有 ~8% 的 run-to-run 漂移（可能来自 NFS/IO、GPU 时钟/温度、其他服务干扰；三次运行期间
GPU 0/1/3/4 的 vLLM 一直在跑）。C4 的 407.33 恰好落在这两次之间 -> **prefill 结论：无差异证据**。

### 5.2 tg 差被 MTP 接受率混淆（关键）
`eval time = 3401.51 ms / 128 tokens` 里的 128 是**含投机接受的最终生成 token 数**，
所以 tg 是"投机解码后的有效吞吐"。它同时受两个因素影响：
- (a) 目标模型+GEMV 的每步速度（C4 声称改进的东西）；
- (b) **MTP 接受率**（一次目标前向平均产出多少 token）。

本实验里 (b) 变化极大：0.551 / 0.587（原版）vs 0.822（C4）。三次运行的 `n_gen` 也不同
（100 / 101 / 115）。C4 的改动只是 GEMV 的 `nwarps`（改变归约顺序 -> 浮点尾数微差 -> 采样出的
token 可能不同），**理论上不该把接受率抬高 40%~49%**。更可能的解释是**内容差异**：

- C4 那次响应（`/tmp/resp256k.json`）是**退化重复**输出：
  `"content":"根据以上材料，根据以上材料，根据以上材料，\n\n<think>\n\n</think>\n\nThe quick brown fox jumps..."`
- 退化/重复文本对 n-gram 类投机极其好猜 -> 接受率自然高。
- 即：C4 的高 tg 很可能**主要来自它那次恰好生成得更好猜**，而不是 GEMV 变快。

-> **结论：256k 下的 tg +31.2% 不能归因给 C4。** 要归因必须消除 (b)（见 §7）。

### 5.3 其它局限
- 每个版本只跑了 1~2 次，无重复统计；无 GPU 独占（vLLM 占 0/1/3/4）。
- MTP draft 的 sampler 落在 CPU（`W spec backend offload failed for seq_id=0; using CPU sampler`，
  三次启动都有），这本身是一个**未消除的性能缺陷**，两侧都受影响。

### 5.4 两侧 binary 其实并非同一上游版本（2026-09-20 追加核实，重要）
- 复核 binary 自报版本：
  - 「原版」`/root/llm/llama.cpp/bin/llama-server`（Sep 10 10:41）→ **`version: 0.4.0-dev (build 1, commit 434ddbb)`**
  - 「C4」`/root/llm/test/v100-opt/llama.cpp/build/bin/llama-server`（Sep 20 09:46）→ `version: 0.4.1-dev (build 0, commit unknown)`
    （源码由 tar 同步进服务器、目录不是 git 仓库，所以 build-info 里 commit 显示 unknown；源 = **b11053** + C4）
  - 注意 `/root/llm/test/llama.cpp/build/bin/llama-server` 与安装版同源（也是 `0.4.0-dev 434ddbb`，Sep 10 10:41）。
- 所以这次 256k 对比实际是**「上游 434ddbb 的原版」vs「上游 b11053 + C4」** —— 混入了两个不同上游
  revision 之间的差异（Sep 10 vs Sep 19，中间隔了数十个 release）。CUDA 后端在这种跨度上并非逐字不变。
- 同样的道理，早先记录的「原版 llama-bench pp512=747.02 / tg128=36.37」与「b11053 baseline pp512=741.16 /
  tg128=36.23」之间的小差异，也可能来自这个上游版本差，而不能全算噪声。
- **结论**：这次 256k 的 C4 收益**既不能证明也不能证伪** —— 变量有两个（上游版本 + C4）。要干净对比
  必须**两侧同源**：已从**同一个 build dir、同一套编译参数**产出两个 binary（§7.1）。

## 6. 因此目前的正确说法
- **C4 = +3.4% tg128**（`llama-bench -p 512 -n 128`，无 MTP / 无采样方差 / 单卡 GPU2）—— 这是唯一干净的数字。
  （baseline 与 C4 都是**同一个 build dir**、同一套编译参数、同一 commit b11053 跑出来的，所以可信。）
- 256k 端到端：**C4 没有变差**（三次里 prefill 在噪声内、tg 在同一次运行里最高），
  但**"长上下文下 C4 收益更大（+31%）"这个说法不成立** —— 它同时被 **MTP 接受率**（§5.2）与
  **两侧 binary 上游版本不同**（§5.4）两个因素混淆。
- **不能**据本报告说"C4 在长上下文有 +31% 收益"，也**不能**说"C4 在长上下文无用" —— 本实验分辨不了。

## 7. 下一步（要拿到可信的长上下文结论）
1. **两侧同源**：从**同一个 build dir、同一套编译参数**产出两个 binary，唯一变量是 `mmvq.cu`：
   - `/root/bin-b11053-c4-server` / `-bench`（b11053 + C4，已存）
   - `/root/bin-b11053-pristine-server` / `-bench`（b11053 原版，用 `/root/mmvq-orig.cu` 在同 build dir 重编）
   > 不要再用 `/root/llm/llama.cpp/bin/*`（那是 `0.4.0-dev 434ddbb`）当"原版"。
2. **关 MTP 做干净 A/B**：`--spec-type none`（已确认合法，且是默认值），原版 vs C4，各跑 2~3 次，
   看 prefill 与 tg 的分布。这条能给出可信的"长上下文 C4 净收益"。
3. **抬 MTP 接受率的对照**：固定采样 seed / 固定 prompt，让两侧生成同样内容再比 tg（消除 §5.2 的混淆）。
4. **修 MTP sampler offload 到 GPU**（`backend offload failed for seq_id=0; using CPU sampler`）：
   潜在收益最大 —— 长上下文服务真正的大头。
5. 每次运行记录 `nvidia-smi` 与时间，规避 / 标记 §5.1 的 8% 漂移。
6. 上**生产模型** `TurboFCFusion-Q4_K_M`（17.23 GiB，**≥3 卡**）做 256K；采样参数按抱抱脸推荐（见 §10）。

## 8. 复现命令（用户自查日志）

```sh
# 看某个 unit 的完整 summary 计时行（TTFT / tg / 接受率 / 总时长）
journalctl -u llama-server-c4   --no-pager -o short-iso | grep -E "prompt eval time|eval time =|total time =|n_gen =|draft acceptance"
journalctl -u llama-server-test --no-pager -o short-iso | grep -E "prompt eval time|eval time =|total time =|n_gen =|draft acceptance"

# 实时跟 prefill 进度（每 ~2000 token 一行）
journalctl -u llama-server-c4 -f | grep --line-buffered "prompt processing"

# 看 MTP 是否掉到 CPU
journalctl -u llama-server-c4 --no-pager | grep -i "offload failed"
```
含义：`prompt eval time` = prefill/TTFT；`eval time` = decode(tg)；`prompt processing, progress` = prefill 进度；
`draft acceptance` = MTP 效果；`graphs reused` = CUDA graph 复用次数。

## 9. 现场状态（2026-09-20 10:48 核对）
- `llama-server-c4` = **active**（port 8082，占 GPU2/5 各 ~12.2 GiB）；`llama-server-test` = inactive；
  `llama-server`（原生产 unit）= inactive；`vllm-1cat` = active（GPU 0/1/3/4，~15.1 GiB/卡，未动）。
- C4 源码：`/root/llm/test/v100-opt/llama.cpp/ggml/src/ggml-cuda/mmvq.cu`；备份 `/root/mmvq-c4-backup.cu` / `/root/mmvq-orig.cu`。
- 产物：`/tmp/prompt256k.txt`、`/tmp/req256k.json`、`/tmp/resp256k.json`。
