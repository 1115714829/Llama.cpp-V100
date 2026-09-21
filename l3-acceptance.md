# l3-acceptance.md — L3 验收：生产模型 Q4_K_M、3 卡同 NUMA、256K（2026-09-20）

**配置**：`Qwen3.8-27B-TurboFCFusion-...-Q4_K_M.gguf`（17.23 GiB）；`CUDA_VISIBLE_DEVICES=0,1,2`（**同一 NUMA node**）；
`--split-mode tensor --tensor-split 1,1,1`；`--ctx-size 262144 --parallel 1`；`--flash-attn on`；KV `q8_0`；
采样按抱抱脸推荐（temp 0.7 / top_p 0.8 / top_k 20 / repeat 1.05）。单元 `llama-server-l3-{pristine,c4}`（8083/8084）。
显存：每卡 ~11.4-11.7 GiB（3 卡合计 ~34.6 GiB = 17.23 权重 + KV/scratch）→ **2 卡太紧，3 卡合适**。

## 1. Prefill / TTFT（prompt = 248901 tokens）
| 变体 | prompt eval | 速率 | 总耗时 |
|---|---|---|---|
| pristine | 672204.71 ms | **370.28 t/s** | 672.20 s |
| c4 | 677586.02 ms | **367.33 t/s** | 677.59 s |

- 差 **−0.80%** → **prefill 无差异**（符合预期：C4 只改 decode 的 GEMV；256K prefill 走 FP16/MMF 路径）。
- **TTFT ≈ 11.2 分钟**，占整个 256K 交互耗时的 ~99% → **谈 256K 体验，TTFT 才是主角**。

## 2. Decode，**关 MTP**（唯一可归因 kernel 的配置；`seed 42`，各 3 次）
| 变体 | rep1 | rep2 | rep3 | 均值 |
|---|---|---|---|---|
| pristine | 33.77 | 34.07 | 34.07 | **33.97 t/s** |
| c4 | 33.89 | 34.21 | 34.20 | **34.10 t/s** |

- 离散度 **<1%**（关 MTP 后极其可重复）→ **Δ = +0.4%**。
- 对照：单卡 llama-bench tg128 上 C4 = **+2.8%**。
- ⇒ **多卡 tensor 切分把 C4 的单卡收益稀释到几乎为零** —— 每卡只拿 1/N 权重、GEMV 被切开，
  这正是先前对"6 卡会掩盖单卡收益"的预测的**首个直接证据**。

## 3. Decode，**开 MTP**（生产配置）：数值由内容主导，**不能用于归因**
| 变体 | eval t/s | 接受率 | mean len |
|---|---|---|---|
| pristine | 56.58 | **0.608** | 3.43 |
| c4 | 39.11 | **0.336** | 2.33 |

- c4 看似慢 45%，但它那一次**接受率只有一半** → 差异来自**生成内容**，不是 kernel（C4 只值 +2.8%）。
- 同一实验在**重复文本** prompt 上曾测得接受率 **0.96 → 85.81 t/s**（见 `mtp-sampler-cpu.md`）。
- ⇒ **带 MTP 的 tg 必须连接受率一起报**；要归因 kernel 必须关 MTP 或固定 seed。
- 观察区间：真实散文 **33.7~56.6 t/s**（接受率 0.245~0.608）；重复文本可到 **85.8 t/s**（0.96）。

## 4. 与 1cat-vLLM 对比（最终核心目标）

> 🛑 **2026-09-20 DSH 审计：本节结论已被取代，勿沿用。**
> 本节结论是「差距的主因不是 kernel，而是投机解码的接受率/草稿质量」。后续实测**推翻**了它：
> ① 我们的 AL（4.11–6.08）**不输** 1cat（v1.5.0 公开 GSM8K 契约 AL request-mean 5.3888 / pooled 4.5644）；
> ② 差距全在**每轮延迟**（见 `premise-check-1cat-vs-llamacpp.md` §3 与 `HANDOFF.md` §6）；
> ③ 本节引用的 1cat「256K = 50.376 t/s」是 **NVFP4 且无 MTP** 的行，与我们「Q4_K_M 带 MTP」不是同一契约。
> 现行判据见 `AUDIT-2026-09-20-dsh.md` 与 `1CAT-PORT-BACKLOG.md`。
参考文档给 27B / 256K decode = **50.376 t/s**（带 DFlash2 投机解码）。
我们同模型同上下文（Q4_K_M、3 卡）**带 MTP** 的实测区间 = **33.7~56.6 t/s**，随接受率摆动。

⇒ **差距的主因不是 kernel，而是投机解码的"接受率 / 草稿质量"**：
- 我们的 MTP 在真实散文上只有 0.245~0.608 的接受率；
- 1cat 用的是 **DFlash2**（专门设计的 draft + 稀疏拒绝 + SM70 tail kernel，并为长上下文衰减、
  M=N 分组、GDN metadata 等做了大量工程，见 `dflash2-llama-cpp-research.md`）。
- ⇒ **继续死磕 GEMV/MMQ 参数的边际收益，远小于把投机解码做好。** 这条判断有数据支撑，
  应作为后续优先级依据。

## 5. 方法与教训（都踩过）
1. **给 instruct 模型用 raw `/completion` 必须自带 chat 模板**：直接喂重复纯文本会让模型**第一个
   token 就 EOS**（`stop_type: "eos"`、`tokens_predicted: 1`、`content: ""`）→ 必须走
   `/v1/chat/completions`。第一版 L3 就是这样废掉的（**prefill 有效、decode 全废**）。
2. **等服务就绪要等 HTTP `/health`**，不要 grep 日志里的 `model loaded`（本环境匹配不上，白等 420 s/轮）。
3. **长任务必须 `nohup` 到后台**：前台跑 >10 min 的 ssh 会被工具超时杀掉（SIGHUP），
   c4 那半轮就是这样被腰斩的。
4. 关 MTP 时 tg 的**离散度 <1%**；开 MTP 时会因内容差到 **45%** —— 同一台机器、同一个模型。
