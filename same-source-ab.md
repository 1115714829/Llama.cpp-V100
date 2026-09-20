# same-source-ab.md — 同源 A/B 的正确做法（含一个曾让结论作废的坑）

> 目的：给出本项目「原版 vs 优化版」对比**可复现**的方法，并记录 2026-09-20 踩到的坑与修正后的结果。
> 适用：llama.cpp CUDA 后端（`ggml-cuda`）任何改动。机器 AC922（6x V100-SXM2-16GB）。

## 1. 坑：`llama-bench` / `llama-server` 只是**启动壳**，代码在共享库里

2026-09-20 我做过一次"同源 A/B"：把 `llama-bench` 二进制分别存成 `-pristine` / `-c4` 再对比，
得到"C4 毫无效果（36.32 vs 36.30）"。**这个结论是错的**，原因：

- `llama-bench` / `llama-server` 是 **209,944 字节的启动壳**，真正的实现全在共享库里：
  `libggml-cuda.so.0.24.0`（**~125 MB**）、`libllama.so.0.4.1`、`libllama-bench-impl.so`、
  `libllama-common.so.0.4.1` …
- 两个"变体"的壳文件 **md5 完全相同**（`llama-bench` = `eadf39b4…`，`llama-server` = `7db94e34…`）：
  我复制的是同一个文件两次。
- 壳的 `DT_RUNPATH` 是**绝对路径** `/root/llm/test/v100-opt/llama.cpp/build/bin`（`readelf -d` 可见），
  所以壳从哪里复制都**只从 build 目录加载库**。当时 build 目录里是 pristine 库 ->
  两次"变体"实际都在跑 pristine。**A/B 变成了 pristine vs pristine，必然无差异。**

**结论：只复制可执行文件做 A/B 是无效的；必须连共享库一起处理。**

## 2. 正确做法（已实测可行）

1. **`DT_RUNPATH` 在 `LD_LIBRARY_PATH` 之后搜索** —— 验证过：
   ```
   LD_LIBRARY_PATH=/root/libdir-c4 ldd /root/libdir-pristine/llama-bench | grep ggml-cuda
   -> libggml-cuda.so.0 => /root/libdir-c4/libggml-cuda.so.0      # 胜出
   ```
   所以可以**同时保留两套库**，靠 `LD_LIBRARY_PATH` 切换。
   （注意：`DT_RPATH` 会在 `LD_LIBRARY_PATH` **之前**搜索，`LD_LIBRARY_PATH` 对它无效；本仓库用的是
   `DT_RUNPATH`，所以有效。换构建系统时要重新确认。）
2. 变体 A：把 **整个 `build/bin/` 目录**（壳 + 所有 `.so`）快照到 `/root/libdir-<variant>/`。
   变体 B：换源码重编后同样快照。
3. **有效性自检（必做）**：确认两套库里**真正承载改动的那个** md5 不同。
   本项目改动在 CUDA 后端，所以看 `libggml-cuda.so.0.24.0`：
   ```
   pristine=3baf1fbc95c63f2ea511c0e10ea129ff
   c4      =0276aed7c13f0f26bf40ca350b0ee8ce   -> DIFFER  (有效)
   ```
   `libllama.so` / `libllama-bench-impl.so` 相同是**正常**的（改动不在那里）。
4. 运行方式：
   ```
   LD_LIBRARY_PATH=/root/libdir-c4 /root/libdir-c4/llama-bench -m <model> -p 512 -n 128 -ngl -1 -r 3
   ```
   也可以交错跑（A,B,A,B）以抵消漂移。

> 备选做法（历史上一直在用的）：**换 `mmvq.cu` + 原地增量重编 + 立刻测**。因为测量时 build 目录里
> 就是那一版的库，所以它是**有效**的 —— 这也是为什么历史数字（36.23 -> 37.4）其实是可信的。
> 缺点是要为每个变体重编一次（每次 ~6 min），没法交错测量。

## 3. 结果（2026-09-20，同源 —— 只有一个 build dir、一套编译参数、唯一变量 `mmvq.cu`）

模型 `Qwen3.8-27B-UD-Q2_K_XL.gguf`，单卡 GPU0，`-ngl -1 -r 3`（3 次重复），盒子空闲（vLLM 已停）。

### 3.1 默认配置（`llama-bench -p 512 -n 128`，无 `-fa`/`-ctk`/`-ctv`）
| 变体 | pp512 | tg128 |
|---|---|---|
| pristine b11053 | 737.13 ± 15.56 | **36.32 ± 0.06** |
| b11053 + C4 | 728.16 ± 20.65 | **37.35 ± 0.09** |
| 差 | -1.2%（在 ±16 噪声内） | **+2.8%** |

**与历史记录一致**（baseline `tg128=36.23`；C4 sweep 报 `37.4~37.52`）。即历史上那个 "+3.4%" 是**真的**，
之前"无效果"是我的方法错误造成的假象。

### 3.2 `-fa on -ctk q8_0 -ctv q8_0` + 256k 上下文（2 卡同 NUMA，用 3.1 的 libdir 同源法）

| 变体 | pp262144 | tg128 | 备注 |
|---|---|---|---|
| pristine（GPU0/1，`-r 1`） | 383.44 | 35.63 | 23 分钟/轮（含 warmup） |
| c4 | **未完成** | — | 见下 |

- **重要发现：`tg128` 在 256k 上下文下是 35.63，与短上下文的 ~35.6/36.3 几乎相同。**
  原因：目标模型是混合架构，64 层里 **48 层是 GDN（递归/线性注意力，状态大小与上下文无关）**，
  只有 16 层走 KV -> **长上下文几乎不拖慢 decode**。
  -> 推论：**对 decode 而言 L0（短上下文）就是 256k 的可靠代理**，不必每次花 23 分钟跑 256k。
- **C4 那一轮为什么没跑完（我的脚本 bug，记录以免重犯）**：`wait2-ab.sh` 用
  `grep -c pp262144 >= 2` 判断"两轮完成"，但 `big-ab-256k.sh` 每轮收尾时**自己又 grep 追加了一次**
  同样的行 -> 一轮就凑够 2 行 -> 脚本在 pristine 刚结束时把整批任务杀了。
  **教训：计数判定要用"唯一标识"，不要用会被自己重复写入的行。**
- `llama-bench` 比服务端慢：**23 分钟/轮**（warmup + 测量，各一遍）；同一上下文服务端约 10.6 分钟。
- 结论：**256k 的 decode 对比优先级低**（L0 已能代表）；256k 的真正价值在 **prefill(TTFT)** 与最终 L3 验收。

## 4. 复现脚本（都在 `1cat-vllm-v100-study/`）
- `p7-libdirs.sh` — 快照 pristine 库目录 + 换 C4 源码重编
- `p8-real-ab.sh` — 等待编译、快照 C4 库目录、**md5 有效性自检**、跑 A/B
- 相关归档：`stress-test-256k.md`（256k 端到端，含混淆因素分析）、`baseline.md`、`c4-mmvq-volta.patch`

## 5. 教训清单（写进技能与规则文件）
1. 比之前先 `md5sum` 两个"变体"的承载库 —— 相同就说明 A/B 无效。
2. 记录两侧 `--version` 与**源 commit**；两侧必须同一 commit。
3. `LD_LIBRARY_PATH` 覆盖 `DT_RUNPATH`（但不覆盖 `DT_RPATH`）。
4. 单次测量不可信：本机同配置 256k prefill 能漂 8%；`-r 3` + 交错是底线。
