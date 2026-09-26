---
name: stress-256k
description: Standard 256K-context load test for the V100 serving box, compatible with BOTH 1cat-vLLM and llama.cpp llama-server (OpenAI-compatible APIs, identical request shape). Measures prefill (TTFT, pp t/s) and decode (tg t/s, AL, ms/round) and reports against the three project baselines BL1 (vLLM) / BL2 (official llama.cpp) / BL3 (B5). Use when the user asks for a 256K stress test, baseline reference values, TTFT or 吐字 measurements, or any performance report that must carry the baseline values.
---

# stress-256k — 256K 负载压测（vLLM / llama 共用）

## 口径（固定，不可自行更改）

- 上下文 = **256K（ctx 262144），压测填充 = 90%**（用户 2026-09-23 定）：prompt 目标 **~235,930 tokens**，留 10% 头空不撞硬墙（真实值以响应 `usage.prompt_tokens` 为准）。
- **预填充**：`TTFT`（首 token 时延 s）、`pp_tps = prompt_tokens / (t_first_token - t0)`。
- **吐字**：`tg_tps = (completion_tokens - 1) / (t_end - t_first_token)`；llama 侧另附 **AL**（draft acceptance）与 **ms/轮**。
- 采样固定：`temperature 0.7, top_p 0.8, top_k 20, repetition_penalty/repeat_penalty 1.05`，`max_tokens = GEN`（默认 128）。
- 同一 prompt **字节级相同**（客户端首次生成后落盘缓存复用）。

## 三基线（后续一切增幅以它们为分母）

| 基线 | 服务 | 端点 | 认证 |
|---|---|---|---|
| **BL1** | `vllm-1cat.service`（用户标准服务，FP8+DFlash2，TP4 卡 0,1,3,4） | `:8000/v1/chat/completions` | Bearer key 在单元文件内，**禁止打印/外传** |
| **BL2** | 官方 llama.cpp：`/root/llm/test/llama-server-bl2.service`（副本；q8_0+DFlash2，卡 0,1,2） | `:8082/v1/chat/completions` | 测试实例无 key |
| **BL3** | B5 采用链：同 BL2 启动命令，仅 `LD_LIBRARY_PATH=/root/libdir-t8b` | 同上 | 同上 |

BL2a = 官方二进制（434ddbb，字面口径）；BL2b = `LD_LIBRARY_PATH=/root/libdir-pristine`（b11053 原版，**同源 A/B，B5 增幅不用换算**）。md5 自证：两侧 `libggml-cuda/libggml-base/libllama` md5 必须不同，相同则 A/B 无意义。

## 运行

```
python scripts/stress_client.py --base-url http://192.168.50.235:8000 \
  --api-key <BL1 时给; BL2/3 留空> --model <服务的 alias> \
  --prompt-tokens 255000 --gen 128 --reps 2 --tag BL1
```

- `reps >= 2` 报离散；服务**不重启**（热口径，记录时标注 NODROP/热）。
- 每 rep 输出一行：`STRESS tag=... rep=... ttft_s=... pp_tps=... tg_tps=... prompt_tokens=... completion_tokens=...`。
- **AL/ms 轮（llama）**：请求后从服务日志 grep `draft acceptance|prompt eval time|eval time`（nohup 日志直接 grep；BL1 用 `journalctl -u vllm-1cat --no-pager -n 200` 只读查）。vLLM 不报 AL 则记 `-`。

## 纪律（红线 + 实机互斥，必须遵守）

1. 动服务前**三查**：`job_list`、在跑子代理、`nvidia-smi`（GPU 必须空）。
2. 服务启停仅限用户授权范围；BL1 测完**恢复原状态**（默认停机）。
3. `/root/llm/systemd/`、`/root/llm/llama.cpp` 只读；副本与日志落 `/root/llm/test/`。
4. 模型只从 `/mnt/3.84t/**`（BL1 的 FP8 是用户标准服务自带，属明示授权例外）。
5. 报数带口径（冷/热、引擎、KV 类型、卡数、DFlash2 on/off、AL）；跨引擎比较注明模型格式差异（FP8 vs Q8_0，V100 上同为 f16 张量核路径）。

## 脚本

- `scripts/stress_client.py` — **执行它，不要手写新客户端**。python3 标准库（urllib）实现，SSE 流式计时，两引擎通用；`--save-prompt/--prompt-file` 控制 prompt 缓存以保证字节级一致。
