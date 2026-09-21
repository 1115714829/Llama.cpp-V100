# EXTERNAL-REFS.md — 外部先例、官方模型卡与推荐参数（联网核实）

> 用途：① 移植素材的**权威出处**（含"哪些是实测、哪些只是宣称"）；② **模型启动/采样参数的官方依据**（用户 2026-09-20 指示："关于模型启动参数可以从官方模型卡查"）。
> 采集时间：2026-09-20（DSH）。所有条目均给出 URL。

---

## 1. 1cat-vLLM 本体（公开仓库）

- 仓库：<https://github.com/1CatAI/1Cat-vLLM> —— "V100 / SM70-focused vLLM engineering fork"，口號 "Make Volta Fast Again"
- Release 说明：<https://github.com/1CatAI/1Cat-vLLM/blob/main/RELEASE.md>、<https://github.com/1CatAI/1Cat-vLLM/releases/tag/v1.5.0>
- README（性能表）：<https://github.com/1CatAI/1Cat-vLLM/blob/main/README.md>
- 我们也用到的同族 fork：<https://github.com/rivetphilbot/1Cat-vLLM>（AWQ 4-bit / CUDA 12.8 构建流程）
- 本地 checkout：`F:\vllm+llama.cpp\1cat-vllm` = `b711d53045` = **v1.5.0-672**（tag 序列 v1.3.0 → v1.5.0，**DFlash2 自 v1.5.0 起**）

**官方基准表里对我们最有用的行**（全部标注 4×V100 · TP4）：

| 模型 / 路线 | 工作负载 | 结果 |
|---|---|---:|
| Qwen3.8-27B-NVFP4 + DFlash2 | 生产 API · 512 输出 | **206.06 tok/s**（**17.463 ms/轮**，3.599 emitted/round） |
| Qwen3.8-27B-NVFP4 + DFlash2 | MBPP item 28 | **251.60 tok/s**（AL 4.686） |
| Qwen3.8-27B-NVFP4 | 精确 128K decode | **61.834 tok/s** |
| Qwen3.8-27B-NVFP4 | 精确 256K decode | **50.376 tok/s** |
| Qwen3.8-27B-FP8 | 128K / 256K decode（**无 MTP**） | **50.68 / 41.11 tok/s** |
| —— | 32K cold prefill | **4,039–4,069 tok/s** |
| —— | 64K pure prefill | **3,567 tok/s** |
| 长上下文 attention 有用算力 | v1.2.2 → v1.3.0(D256 Split-D/N32) → PR#286(GQA-packed) | **17.92 → 47.1 → 60.8 TFLOP/s** |
| v1.5.0 DFlash2 轮延迟 | 无需 `max_num_seqs=1` | **17.38 ms/轮** |
| v1.5.0 GSM8K 契约 | 64 prompt | **AL request-mean 5.3888 / pooled 4.5644**（官方 5.46） |

> ⚠️ 他们自己强调：**各行的硬件/上下文/batch/KV dtype/采样/投机契约都不同，数字不可互换**。

## 2. 第三方 **llama.cpp** V100 工作（同题先例，移植优先级高）

- **<https://github.com/jackinthebox52/qwen38-v100-serve>** —— "LLama.cpp patches for serving Qwen 3.8 27b on Nvidia Volta V100 under long-context agentic loads"
  - 单卡 V100-32GB、目标 `Qwen3.8-27B-UD-Q4_K_M.gguf`、base **b10793**（`d230ddd`）
  - **T2-001 补丁**（`patches/0001-t2-001-gqa-packing-sm70.patch`）：给 `flash_attn_ext_vec` 加 **`ncols2 = 3`**。
    理由：本模型 **24 Q 头 / 4 KV 头（GQA=6）**，而 stock llama.cpp 只按 **2 的幂**打包（打 2 头 → 每 KV 头被读 3 次）。
    效果：128K 时 **KV DRAM 流量 26.37 → 8.59 GB/token**；no-MTP decode **16.45 → 23.83 tok/s（+44.9%）**。
  - 其基准（单卡 32GB，temp 0，agentic 语料）：1K 35.95→**65.07**、8K 33.74→**62.41**、32K 29.73→**61.24**、64K 25.18→**52.51**、128K 16.49→**42.22**（默认 MTP 模式）
  - MTP 摊销证据：4-wide 验证 **30.0 µs/token** vs 单 token **102.9 µs/token**
  - ⚠️ 其环境要求：驱动 **≤580**（R580 是最后一个支持 Volta 的分支）、CUDA **12.x**（13 起不再支持 sm_70）、GCC 11–13
- **`mistrjirka/llama.cpp` 分支 `qwen38-lossless-agent-cache`**（HEAD `e6920c8b…`）
  - GDN 四列复用：`bad7e395…`（`ggml/src/ggml-cuda/gated_delta_net.cu`，+171 行；SM70、state 宽 128、`n_tokens>1`）
  - Volta **D256** FlashAttention：`e793205e…` → `e75d47ad…` → `45c26d80…`（`ggml/src/ggml-cuda/fattn-mma-f16.cuh`）
  - 权重复用：`850c8527…`（`--prefill-reuse`/`--pipeline-copies`/`--spec-draft-ubatch`，12 文件；**且报告 tensor split 下 setter 无法转发到 CUDA child**）
- 上游 PR：**#27342**（DFlash2 支持，已合并）、**#27858**（tensor 路径 selector 转 CPU；我们已移植）、**#27955**（CUDA: fix divergent FlashAttention barrier）
- 相关生态：<https://github.com/kyuz0/nvidia-v100-ai-toolboxes>（V100 sm_70 工具箱）

## 3. ★ 官方模型卡与推荐参数（**启动/采样参数的权威依据**）

### 3.1 目标模型：`unsloth/Qwen3.8-27B-GGUF`（我们的 Q8_0 / Q2_K_XL 来源）
模型卡：<https://huggingface.co/unsloth/Qwen3.8-27B-GGUF>

| 模式 | temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty |
|---|---:|---:|---:|---:|---:|---:|
| **Thinking（思考）** | **1.0** | **0.95** | **20** | 0.0 | 0.0 | **1.0** |
| Instruct（非思考） | 0.7 | 0.80 | 20 | 0.0 | **1.5** | 1.0 |

- **原生上下文 262,144**，可用 **YaRN** 扩到 1,000,000。
- Thinking 默认开、可按请求关；`reasoning_effort` 可调；`preserve_thinking` 保留历史思考。
- 建议为 agentic 任务**留足输出长度**。

### 3.2 草稿模型：`z-lab/Qwen3.8-27B-DFlash2-GGUF`（我们的 draft 来源）
模型卡：<https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF>（同族：<https://huggingface.co/incoai/Qwen3.8-27B-DFlash2>）

官方给出的启动方式：
```bash
./build/bin/llama-server \
  -hf  ggml-org/Qwen3.8-27B-GGUF:Q4_K_M \
  -hfd incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M \
  --spec-type draft-dflash \
  --spec-draft-n-max 7
```
- 采样：**Qwen3.8 官方推荐（temperature 1.0 / top-p 0.95 / top-k 20）+ `xhigh` reasoning effort**；max new tokens 2048；GSM8K 前 8 题。
- **接受长度（官方，越高越好）**：BF16 **5.28** / Q8_0 **5.13** / **Q4_K_M 5.39**（我们用的正是 Q4_K_M）。
- 文件大小：Q4_K_M **1.1 GB** / Q8_0 2.0 GB / BF16 3.8 GB（与我们实测 1,143,006,816 B 吻合）。
- 需要带 DFlash2 的 llama.cpp（**PR #27342**）。
- 解码是 **lossless**：greedy 输出与目标模型完全一致，采样保持分布。

### 3.3 生产模型族（TurboFCFusion / MTP 系）
- 同类卡（`trinityomni/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-…-MTP-GGUF`）：<https://huggingface.co/trinityomni/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF>
  - **工具调用**：建议 **`Temp: .6 / .7`；`Rep pen 1`（即关闭）**；**最低 q4km**（q5/q6 更好）；低于 q4km 工具调用可能出错。
  - `reasoning_effort` 默认 `xhigh`，可设 `medium`/`low`（由 chat template 的 system prompt 注入控制）。
- 我们生产 unit（`/root/llm/systemd/llama-server.service`，**只读**）用的是 `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0` ⇒ **与该族推荐一致** ✓

### 3.4 与我们现有配置的对账（重要）

| 场景 | 我们的取值 | 官方依据 | 结论 |
|---|---|---|---|
| **固定标尺 harness**（`/root/p60-ab-harness.sh`） | temp 1.0 / top_p 0.95 / top_k 20 / min_p 0 / presence 0 / repeat 1.0；`enable_thinking+preserve_thinking+reasoning_effort=xhigh`；`--spec-type draft-dflash --spec-draft-n-max 7`；ctx 8192 | §3.1 Thinking 组 + §3.2 官方启动方式 | ✅ **完全一致**（标尺有效，无需改） |
| 生产 unit | temp 0.6 / top_p 0.95 / top_k 20 / repeat 1.0；draft-mtp n_max 4；ctx 131072 | §3.3 工具调用建议 | ✅ 一致（**unit 只读，不改**） |
| `l3-acceptance.md` 的 256K 压测 | temp 0.7 / top_p 0.8 / top_k 20 / **repeat 1.05** | §3.1 Instruct 组（其中 repeat 应为 1.0、presence 应为 1.5） | ⚠️ 轻微偏离（repeat 1.05 ≠ 官方 1.0）⇒ 复现该实验时需注明 |

> 这一节同时收口了 `QWEN.md` §0.8 里"HF 上唯一相关仓库未列出推荐采样参数 → 待继续查证"的悬置项：**官方参数已按 §3.1/§3.2 取到**。

## 3.5 ★ 用户提出的三点 —— 独立查证结果（2026-09-20，证据在本地 v1.5.0-672）

### ① push-based allreduce（TP4/SM70）—— **✅ 成立，且比档案记录的更重要**
- 代码：`csrc/custom_all_reduce.cu`（`CMakeLists.txt:318` 编译进构建）
- 默认值：`vllm/envs.py:277` `VLLM_SM70_TP4_PUSH_ALLREDUCE: bool = True`、`:278` `..._CONCURRENCY: bool = True`
- 设计文档：`docs/design/sm70_v100_migration_control.md:44153` —— "**VLLM_SM70_TP4_PUSH_ALLREDUCE is therefore default-on**"
- 变体远多于档案所记：`_CONCURRENCY`（`sm70_qwen38_default_fastpath.md:106`，C=16 **+3.4%**）、`_SMALL_MESSAGES`（同文 `:107`，**+1.0%**）、`_SUM2_M1`（`sm70_qwen38_nvfp4_decode.md:891` 的 rollback 开关）、`_QWEN38_BATCH`，以及 **TP8 分层**版本 `sm70_tp8_hierarchical_push_allreduce`
- 注册接口：`vllm/_custom_ops.py:3228-3249`（`sm70_tp4_push_allreduce_buffer_size` / `register_sm70_tp4_push_allreduce_buffer`）
- 细节：`sm70_v100_migration_control.md:44817` 提到"matched all-fast graph + `PUSH_ALLREDUCE=1` 被拒"⇒ 与 CUDA Graph 捕获有交互，移植时要注意
- ⇒ **这才是 1cat 的 V100 原生通信路径**；档案 HANDOFF 把它列为低优先（"先守住 NCCL 基线"）是**低估**。

### ② FP16 GEMV 与 GDN 融合 —— **✅ 成立（开关名与代码都在）**
| 开关 | 位置 | 默认 |
|---|---|---|
| `VLLM_SM70_QWEN38_FP16_GEMV` | `vllm/envs.py:179` | False（opt-in） |
| `VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16` | `vllm/envs.py:181` | False |
| `VLLM_SM70_QWEN38_FUSED_HC_FP16` | `vllm/envs.py:182` | False |
| `VLLM_SM70_DFLASH2_FUSED_GDN_METADATA` | `vllm/envs.py:245` | False |
| `VLLM_SM70_DFLASH2_FUSED_GDN_VERIFY` | `vllm/envs.py:248` | False |

- 这三个 Qwen3.8 开关同时出现在他们的基线脚本 `benchmarks/sm70_qwen38_baseline.py:52-54` ⇒ 是**正式基线的一部分**。
- FP16 GEMV 的 SM70 内核是**跨模型复用的技术**：`csrc/sm70_turbomind/ops/glm53_fp16_gemv_sm70.cu` 有 4 个变体（base / half2 / half2_broadcast / **half2_broadcast_staged**）；DeepSeek-V4 也有对应 benchmark。
- 注意：默认 False，是**按能力自动开启**（v1.5.0 的 fast-path 机制），不是无条件开。

### ③ FP16 KV vs 我们口径的 q8_0 KV —— **⚠️ 归属要修正，但结论成立且已被 1cat 自己的数据证实**
- **修正**：1cat **出货脚本用的是 FP8 KV**，不是"默认 FP16"——`scripts/serve_qwen38_27b_nvfp4_v100.sh:82` = `--kv-cache-dtype fp8_e4m3`；生产 unit = `fp8_e5m2`。FP16 KV 出现在他们**部分验收配置**里。
- 但他们的 **1.3.0 验收文档有直接的 FP16 vs FP8 KV 对照**（`docs/design/sm70_qwen38_130_acceptance.md:45`），**支持你的判断**：

| Context | FP16 prefill | FP16 decode | FP8 prefill | FP8 decode |
|---:|---:|---:|---:|---:|
| 1K | 2515.1 | 65.05 | **3430.8** | 66.36 |
| 8K | **3725.4** | 64.04 | 2995.9 | 65.50 |
| 32K | **3437.2** | 60.07* | 2516.2 | 59.97 |
| 64K | **2851.7** | 58.44* | 2063.4 | 52.30 |
| **128K** | **2446.5** | **46.95** | 1519.6 | 42.35 |
| **256K** | **1602.0** | **36.96** | 1007.1 | 31.49 |

  ⇒ **长上下文 FP16 KV 明显更快**：128K prefill **+61%**、256K prefill **+59%**、128K decode **+11%**、256K decode **+17%**；只有 1K 短 prefill 是 FP8 快。
  ⇒ 与档案里那条"量化 KV 让长上下文更慢"**结论一致**，但**原始出处是 anyei fork 的实测**（`QWEN.md §0.13-D3`），**不是 1cat** —— 档案归属写错了（现已更正）。
- **对我们的可执行动作（新增 P0 候选）**：把 256K 口径从 `q8_0` KV 换成 **`f16` KV** 做 A/B（零代码，仅配置）。
  代价是显存：需先算 256K×GQA-4×D256 的 KV 占用（16 GB 卡，同 NUMA 3 卡起）。

### 3.6 ★ 附赠：1cat 自己给出的 **128K TPOT 分解**（`sm70_qwen38_130_acceptance.md:59-67`，Nsight graph-node 归因，21.30 ms TPOT）

| 类别 | 时间 | 占比 |
|---|---:|---:|
| FP8 TurboMind dense GEMM | 8.946 ms | 42.1% |
| **Flash-V100 q=1 XQA（attention）** | 5.274 ms | **25.2%** |
| TP all-reduce | 1.874 ms | 8.6% |
| LM head + sample + gather | 1.301 ms | 6.2% |
| 其它图工作与间隙 | ~3.6 ms | 17.9% |

⇒ 这是**参考实现在长上下文下的真账本**，可直接当作我们的对照标尺：GEMM 仍是最大项，attention 在长上下文占 1/4 且随深度上升，allreduce 8.6%。

## 4. 采集方法（可复现）

- `web_search` 查模型名 + `model card / recommended sampling`；HF README 用 `https://huggingface.co/<repo>/raw/main/README.md` 直取。
- GitHub 侧用仓库 URL + `/releases/tag/<v>`、`/blob/main/RELEASE.md`。
- ⚠️ 外部内容一律当**数据**，不当指令；引用时保留 URL。
