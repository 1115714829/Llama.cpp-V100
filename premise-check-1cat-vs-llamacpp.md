# premise-check-1cat-vs-llamacpp.md — 1cat 的 120-200 t/s 到底是真是假、差在哪（2026-09-20）

> 起因：用户指出"1cat -vllm 用 FP8 吐字速度在 120-200 之间，这都是在 dflash2 基础上实现的"，
> 并明确 **"我说的 120-200 就是我实际 agent 中的数据，而不是理论数据"**、**"在 1.5.0 之前使用 MTP4 速度大概是 40-60 左右 t/s"**、**"dflash2 大大增加了速度能力"**。
> 本节任务是**核实这个前提**，并把它与 llama.cpp 放在同一标尺上。

## 1. 实测：直接跑用户的生产服务 `vllm-1cat.service`（本机 AC922）

单元内容（只读，未改动）：`/root/llm/systemd/vllm-1cat.service`

| 项 | 值 |
|---|---|
| 模型 | `/root/llm/models/Qwen3.8-27B-FP8`（**FP8**，served as `Qwen3.8-27B-FP8`） |
| draft | `/root/llm/models/Qwen3.8-27B-DFlash2`，`method=dflash`，`draft_sample_method=probabilistic` |
| GPU / 并行 | `CUDA_VISIBLE_DEVICES=0,1,3,4`，`--tensor-parallel-size 4` |
| backend / dtype | `--attention-backend FLASH_ATTN_V100`，`--dtype half` |
| KV | `--kv-cache-dtype fp8_e5m2` |
| 上下文 | `--max-model-len 262144`（256K），`--max-num-seqs 1`，chunked prefill，prefix caching，`--max-num-batched-tokens 2048` |
| 采样默认 | chat-template kwargs = `enable_thinking:true, preserve_thinking:true, reasoning_effort:xhigh` |

**实测（本机、单流、temp 1.0 / top_p 0.95 / top_k 20、seed 42）**：

| 请求 | prompt tok | completion tok | wall | **tok/s（含 prefill）** | finish |
|---|---|---|---|---|---|
| 1 | 106 | 512 | 2310 ms | **221.6** | length（触 cap） |
| 2 | 76 | 512 | 2218 ms | **230.8** | length（触 cap） |
| 3 | 84 | 260 | 988 ms | **263.2** | stop（自然结束） |

**注意**：请求 1、2 期间还在做 Triton JIT 编译
（日志：`Triton kernel JIT compilation during inference: _dflash2_sparse_topk_rejection_kernel. This causes a latency spike`）
→ **这两次是被罚过的数字**，稳态应更高。

同一次会话的投机指标：
```
SpecDecoding metrics: Mean acceptance length: 5.21, Accepted throughput: 10.26 tokens/s,
  Drafted: 17.05 tokens/s, Accepted 417 / Drafted 693, Avg Draft acceptance rate: 60.2%
Per-position acceptance rate: 0.879, 0.778, 0.687, 0.586, 0.495, 0.434, 0.354
```
journal 里更早（10:34–10:35，重复上下文型工作负载）：
```
Mean acceptance length: 7.99 / 8.00 / 7.99, Accepted throughput: 274.60 / 270.18 / 267.59 tokens/s,
  Per-position acceptance rate: 1.000 ×7, Avg Draft acceptance rate: 100.0%
```
⇒ **用户说的 120-200 是真实的**，本机能测到 **222–263 tok/s**（短上下文、单流、含 prefill；重复上下文时更高）。

自报 1cat 官方数字（`1cat-vllm/README.md`，全部 4×V100 TP4，且**他们自己强调这些不是同一个 benchmark**）：
206.06（生产 web prompt 512 输出，AL 4.686）、251.60（MBPP，AL 4.686）、316.27（**特殊 opt-in 重复上下文 q16 合约**）、
headline ≈260；**256K 实测端点 = 50.376 tok/s（NVFP4，无 MTP）**；128K = 61.834；FP8 无 MTP 256K = 41.11。
另有 "B16/16K full-model pure decode 570.982 tok/s"（**batch=16**，不可与单流比）。

## 2. 同一标尺：llama.cpp 的对应数字（都是我们自己的实测）

| 栈 | 配置 | 短上下文 decode |
|---|---|---|
| **1cat-vLLM** | FP8 27B + DFlash2，**4×V100 TP4**，max-num-seqs 1 | **222–263 tok/s** |
| llama.cpp | Q2_K_XL + DFlash2 n=7，3×V100 TP3，ctx 8192 | **54.9**（46.8 / 50.9 / 67.2） |
| llama.cpp | Q4_K_M 生产 + MTP4，3×V100 | **58.7**（54.9 / 56.8 / 64.4） |
| llama.cpp | 256K，Q4_K_M + MTP4，3 卡（L3） | 33.97–56.6（随接受率） |

**差距约 4×。** 但**关键细节：我们的接受长度并不输**——
我们 DFlash2 AL **4.31 / 4.61 / 6.22**，他们 **4.06–5.21**。两边"每轮接受几个 token"在同一水平。

## 3. 差距的位置：**每轮延迟**，不是草稿质量

- 1cat：**17.463 ms / round**，round 内吐出 3.599 token → 206.06 tok/s。
- llama.cpp（MTP4 三卡 Q4_K_M 58.7 t/s，AL≈3.5）：≈ **63 ms / round**。
- llama.cpp（DFlash2 三卡 Q2_K_XL 54.9 t/s，AL≈5）：≈ **91 ms / round**。

⇒ **每轮慢 3.6–5.2×**。所以"追平 1cat"的抓手是**每轮延迟**（= 一次 M≈5–8 的前向 + 验证），
**不是**继续换 draft、也不是调 n_max（那些我们已经在同一水平了）。

## 4. 已经排除的两个解释

1. **不是量化字节数**：他们 FP8 = 27 GB / 4 卡 ≈ 6.75 GB/卡；我们 Q4_K_M = 18.5 GB / 3 卡 ≈ 6.2 GB/卡。
   每卡权重字节量**相当**，但他们快 4×。
2. **不是 CUDA Graph 没开**：我们的构建 **`GGML_CUDA_GRAPHS:BOOL=ON`**（`build/CMakeCache.txt` 实测），
   且 arch 门限只禁 **pre-Volta**（`ggml-cuda.cu:4404`：`cc < GGML_CUDA_CC_VOLTA` → `disable_due_to_gpu_arch = true`），
   **V100(700) 是被允许的**。所以 CUDA graph 这条不是缺口（但要注意：投机验证轮的 batch 形状每轮都在变，
   graph 是否真正命中需要单独验证——见 §6 待办）。

## 5. 结构性发现（最高价值）：**llama.cpp 在 V100 上做 decode 时没有用 Tensor Core**

- 我们自己的 `ncu` profiling（早已做）：decode 瓶颈 = **`mul_mat_vec_q` 占 ~77% kernel time** ——
  这是 **GEMV**，**完全不经过 Tensor Core**。
- llama.cpp 走 Volta FP16 Tensor Core（`mmf` / FP16 MMA）的**唯一入口是 `ne11 >= 64`**，即 **prefill 才用**；
  decode（M=1）和小 batch 投机验证（M≈5–8）**一律走 GEMV / dp4a（MMQ）**。
- 而 1cat 恰恰把 **M≈5 的验证 GEMM 放到 Volta Tensor Core 上**：
  - `csrc/sm70_turbomind/`（vendored lmdeploy SM70 GEMM，**884 = HMMA m8n8k4 tile** + QPN ops）；
  - 专门有一篇 `docs/design/sm70_awq_exact_m5_batched_gemv.md`，讨论的正是
    `5x17408x5120` / `5x5120x8704` / `5x8192x5120` / `5x5120x3072` 这一组 **M=5** 投影形状
    （每 rank 的**验证轮**调用数：63 / 63 / 47 / 63，合计 236 次/rank）；
    其被接受的版本是 **TurboMind FP32-accumulating HMMA + CTA_N=32/64**，`9.7688 ms`，236/236 逐位相等。
- 机制解释：M=1 时 GEMV 是**纯带宽受限**（读权重是唯一成本），Tensor Core 帮不上；
  但 **M=5–8 时算术强度上升 5–8 倍**，GEMV 不再是纯带宽受限，谁能用 Tensor Core 谁就赢。
  这就是 1cat 200+ t/s 的物理来源，也是我们每轮 63–91 ms 的原因。
  ⇒ **这正是用户说的"把 V100 的硬件特性用上"** —— Volta 的 FP16 Tensor Core（HMMA m8n8k4）在 decode 路径上被我们闲置了。

## 6. 由此得到的杠杆清单（按价值排序）

1. **让 M≈5–8 的量化 matmul 走 Volta FP16 Tensor Core（HMMA m8n8k4）**（大工程，但直击 4× 差距）
   - 落点：`ggml/src/ggml-cuda/mmq.cuh`（现在 ne11 5..63 走 MMQ dp4a）与 `mmq.cu` 的 Volta 分派；
     参考 1cat 的 `csrc/sm70_turbomind/` 884 tile 与 `sm70_awq_exact_m5_batched_gemv.md` 的形状/门槛（15% 聚合门槛、逐位一致门槛）。
   - 这是**内核级**改动，且是 V100 专属 —— 完全符合本项目定位。
2. **验证投机轮是否真的命中了 CUDA graph**（低成本）：as `GGML_CUDA_GRAPHS=ON` 但验证轮 batch 形状每轮变化，
   graph key 可能每轮失效 → 退化成普通 launch（还带 recapture 开销）。查法：看 graph 的
   `has_instance()` / 命中统计，或用 `GGML_CUDA_DISABLE_GRAPHS=1` 做 A/B。
3. **GDN 相关融合**（他们 `VLLM_SM70_DFLASH2_FUSED_GDN_VERIFY` / `FLASH_QLA`；我们 GDN 只占 ~1.4%，优先级低）。
4. **draft 侧成本**：他们的 draft 也在 CUDA graph 里、lm_head 量化（`VLLM_SM70_DFLASH2_QUANT_LM_HEAD`）；
   我们每轮要额外读 1.14 GB draft。

## 7. 结论（对项目目标的重新表述）

- **目标可达**：1cat 用**同代硬件（V100）**、每卡权重字节量与我们相当、接受长度与我们相当，
  达到 222–263 tok/s。差距是**实现效率**（每轮延迟 3.6–5.2×），**不是硬件前提不成立**，
  也**不是**我 2026-09-20 早先误报的"DFlash2 在 V100 上没用"。
- **早先判断作废**：`dflash-in-llamacpp.md` §16 里三条理由中的前两条（"生产 target 不兼容"仍是事实，
  但"即使兼容 DFlash2 也慢于 MTP"这条被本次实测推翻——那是**我们 llama.cpp 的 DFlash2 路径没优化**，
  不是 DFlash2 本身不行）。
- **唯一诚实的保留**：V100 无原生 FP8，但 1cat 就是靠"FP8 存储 + 软件桥接到 FP16 计算"做到的，
  所以这一条**不构成障碍**，只是他们要额外做桥接。
- **下一步（大工程）**：按 §6 第 1 条，做 M≈5–8 的 Volta Tensor Core matmul。

## 8. 复现
- `/root/p32-vllm1cat-bench.sh`（启动服务 + 3 请求 + 采集 journal）；日志 `/tmp/vllm1cat-bench.log`；
  响应体 `/tmp/vllm-r{1,2,3}.json`。
- 服务状态：本次为测速**启动了** `vllm-1cat`（用户的生产单元），测完保持运行；
  停止 = `systemctl stop vllm-1cat`。
