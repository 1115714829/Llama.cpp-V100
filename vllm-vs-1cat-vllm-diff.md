# vLLM vs 1cat-vLLM 差异分析（V100/SM70 专项）

> 目的：从 1cat-vLLM（官方 vllm 的 V100 优化分支）提取核心改动，学习其实现方法，
> 迁移到 llama.cpp 的 V100（SM70）优化。本文件是差异总览；核心改动细节见 `core-changes.md`（待深挖），
> 参考文档见 `FlashAttention-V100-reference.md`。

## 0. 工作区拓扑与基线（2026-09-20 重建并复核）

四个目录并列在 `F:\vllm+llama.cpp\`：

| 目录 | 是什么 | HEAD | 状态 |
|---|---|---|---|
| `llama.cpp/` | **本项目**（V100 专项优化目标） | `1af554f8f` = tag `b11053` | 工作树**仅** `ggml/src/ggml-cuda/mmvq.cu` 被改（= C4）；其余 3604 个文件与 b11053 一致；无未跟踪文件；`ls-files -v` 全为 `H`（无 skip-worktree / assume-unchanged） |
| `vllm/` | 官方 vllm **最新 main**（参考用，非基线） | `751f6807d9` | 洁净；比分叉点前进 **4543** 个 commit |
| `vllm-forkpoint/` | **1cat 分叉点版本的 vllm**（`vllm` 仓库的 git worktree，detached HEAD） | `4ff865c38`（describe `v0.21.1rc0-438-g4ff865c38e`） | 洁净；已核对**不含**任何 1cat 独有目录（`flash-attention-v100` / `csrc/sm70_turbomind` / `flashinfer-sm70` / `flash_qla`） |
| `1cat-vllm/` | 1cat-vLLM（学习对象） | `b711d53045`（describe `v1.5.0-672-gb711d53045`） | 洁净 |

**注意事项**
- `vllm/` 与 `vllm-forkpoint/` 是同一个 git 仓库的两个 worktree（共享对象库）。`vllm-forkpoint/` 只用于
  对照 / 查"上游原本长什么样"，不要在它上面改代码。
- 分叉点 commit 在 `1cat-vllm` 仓库内**也存在**，所以"1cat 改了什么"直接：
  `git -C 1cat-vllm diff 4ff865c38eac3c3602c064fd43b00836805cd0a4..HEAD [-- <path>]`，
  不需要第三个仓库参与；`vllm-forkpoint/` 是给你肉眼对照用的。
- 想撤销 `vllm-forkpoint/`：`git -C vllm worktree remove ../vllm-forkpoint`。

### 本次重建的产物（均在本目录）
- `1cat-changed-files-vs-forkpoint.txt` — `--name-status` 全清单（1443 行，A/M/D）
- `1cat-diffstat-vs-forkpoint.txt` — 逐文件增删行数
- `1cat-changes-by-dir.txt` — 按顶层目录聚合（见 §2）
- `c4-mmvq-volta.patch` — llama.cpp 工作树当前**唯一**改动（C4）的补丁，随时可回退 / 重打

## 1. 分叉点（2026-09-20 独立复核通过）
- 1cat-vLLM 从官方 vllm 分叉于 commit **`4ff865c38eac3c3602c064fd43b00836805cd0a4`**。
- 该 commit：`[Bugfix] Disable allreduce_rms_fusion when pipeline_parallel_size > 1 (#43616)`，AuthorDate 2026-05-29。
- 最近的 vllm tag：**`v0.21.1rc0` + 438 个 commit**（`git describe --tags 4ff865c38` = `v0.21.1rc0-438-g4ff865c38e`）。
- 1cat 在其上共 **1331 个提交**，几乎全是 SM70/V100 专项工作。
- **复核方式（两个独立证据，不靠记忆）**
  1. 在 1cat 仓库内：`git merge-base 751f6807d9 HEAD`（vllm 真实 main tip 的对象在 1cat 仓库内也存在）
     -> 返回 `4ff865c38...`；即真实共同祖先就是它，分叉后 1cat 未再并入上游。
  2. 计数复核：`vllm` 从分叉点到 main 走了 **4543** 个 commit；1cat 从分叉点到 HEAD 走了 **1331** 个 commit。

## 2. 差异规模（2026-09-20 复核，数字与上轮一致）
- `git diff --shortstat 4ff865c38..HEAD`：**1443 文件，+534174 / -4020**。
- 状态分布：**新增 1078 / 修改 365 / 删除 0 / 改名 0** —— 1cat **没有删除任何上游文件**。
- **以新增为主**：534K 行新增 vs 仅 4K 行删除（4K 全在 365 个被改文件内部）—— 1cat 基本不动 vllm
  原有代码，而是大量新增 SM70 kernel、新模型架构、量化算子、DFlash2 投机解码等。
- 按顶层目录聚合（完整表见 `1cat-changes-by-dir.txt`）：

| 目录 | 文件数 | 新增行 | 删除行 |
|---|---|---|---|
| `vllm/`（Python 侧：模型 / 量化路由 / 调度） | 437 | +136564 | -3348 |
| `tests/` | 290 | +63163 | -136 |
| `benchmarks/`（SM70 micro-benchmark 轨迹） | 261 | +128510 | -19 |
| `csrc/`（CUDA kernel：sm70_v37 / sm70_79t / marlin / moe） | 221 | +83801 | -78 |
| `docs/` | 127 | +84739 | -10 |
| `flash-attention-v100/` | 30 | +19771 | 0 |
| `flash_qla/`（GDN gated delta rule） | 23 | +7398 | 0 |
| `requirements/` | 13 | +206 | -239 |
| 顶层文件（README / AGENTS / 审计 md 等） | 12 | +4216 | -172 |
| `tools/` `cmake/` `flashinfer-sm70/` `docker/` `scripts/` `.buildkite/` `assets/` | 29 | +5806 | -18 |
| **合计** | **1443** | **+534174** | **-4020** |

- 完整清单：`1cat-changed-files-vs-forkpoint.txt`（A/M/D）、`1cat-diffstat-vs-forkpoint.txt`（逐文件行数）。
  旧文件 `changed-files.txt`（`--name-only`）保留，内容与新清单一致。

**读法提示**：学习重点在 `csrc/` + `flash-attention-v100/` + `flash_qla/` + `flashinfer-sm70/`
（合计 278 文件 / ~111K 行）；`benchmarks/` 是 1cat 的调参轨迹（参考价值高但量最大，按需摘读）；
`vllm/` 是 Python 侧编排（模型 / 量化 / DFlash2 / CUDA Graph）。

## 3. 核心 V100/SM70 代码区（按目录分类）

### 3.1 FlashAttention V100（对应参考文档 Layer 1/2/3）
- `flash-attention-v100/kernel/`
  - `fused_mha_forward.cu` / `fused_mha_forward_paged*.cu` / `fused_mha_forward_paged_full.cu`
  - `flash_decode_paged.cu` / `flash_decode_turboquant.cu`
  - `fp8_kv_bridge.cu` / `fp8_kv_utils.cuh` / `paged_kv_utils.cuh`（KV 宽加载 / FP8 KV）
  - `contiguous_to_paged.cu` / `paged_to_contiguous*.cu`（KV 布局转换）
  - `h3/forward.cu` / `h3/forward_sparse.cu`（稀疏 attention，QSA）
  - `flash_v100_traits.cuh`（Volta HMMA fragment 归属 / traits）
- `csrc/attention/sm70_v37/`（`bridge.cu`/`prefill.cu`/`tail.cu`）— D=256 重写（v37）
- `csrc/attention/sm70_79t/`（`prefill.cu`/`prefill_q8192.cu`/`stable_rows.cuh`）— 75T/79T prefill
- `csrc/attention/sm70_grouped_long/`（`grouped-attention.cu`/`scalar-attention.cu`/`fp8_kv_utils.cuh`）

### 3.2 量化 GEMM / MoE（NVFP4 / MXFP4 / U4 / U8）
- `csrc/quantization/marlin/sm70_*`（`sm70_marlin_{nvfp4,mxfp4,u4,u8,u4b8,u8b128,fp8}_gemm.cu`、
  `sm70_marlin_{mma,splitk,common,dispatch,iterator_utils}*`）
- `csrc/moe/marlin_moe_wna16/sm70_*`（grouped MoE 各量化 GEMM + dispatch）
- `csrc/moe/moe_permute_unpermute_op.cu` / `moe/permute_unpermute_kernels/`

### 3.3 TurboMind SM70 GEMM（vendored lmdeploy）
- `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/`
  - `kernel/sm70_884_{4,8,16}.cu` —— **884 = HMMA m8n8k4**（Volta 基本 MMA tile）
  - `scheduler_sm70.cuh`、`gemm.cu`、`kernel.cu`、`unpack.cu`、tuner/
- `csrc/sm70_turbomind/ops/`（`awq_qpn_m1_sm70.cu`、`nvfp4_qpn{2,4}_sm70.cu`、`mxfp4_qpn_m1_sm70.cu`、
  `glm53_fp16_gemv_sm70.cu`、`qwen38_prefill_cutlass.cu`、`sm70_top20_philox_sampler.cu`）

### 3.4 Volta MMA 基础设施
- `flashinfer-sm70/include/flashinfer/attention/sm70/volta_mma.cuh`
- `flashinfer-sm70/csrc/h3_noncausal_sm70.cu`
- `csrc/sm70_tile_runtime_signal.cuh`

### 3.5 GDN（gated delta rule / Mamba 类）
- `flash_qla/ops/gated_delta_rule/chunk/sm70/csrc/gdn_forward.cu`

### 3.6 研究用 micro-benchmark（HMMA/WMMA 调参轨迹，数量极多）
- `benchmarks/csrc/sm70_*`：`sm70_hmma884_schedule_micro.cu`、`sm70_raw_hmma_probe.cu`、
  `sm70_wmma_fragment_probe.cu`、`sm70_native_bm32_*`（qk_panel / pv_panel / allp / splitkv /
  dualwarp_pipeline / full_pipeline_d128 ...）、`sm70_instruction_latency_probe.cu`
- `benchmarks/kernels/sm70_*`：`sm70_hc_batch_screen.cu`、`sm70_qsa_topk_sidecar.cu`、`sm70_w13_exact_sidecar.cu`

> 命名解码（对照参考文档）：`qk_panel`/`pv_panel`=QK/PV 分块；`allp`=all-partial；`bm32/bm64`=block
> M 32/64；`d128`=D=128；`884`=m8n8k4；`h3`=某稀疏/变体；`dualwarp_pipeline`=双 warp 流水线。

## 4. 非 CUDA 部分（待深究，Python 侧）
- 新模型架构：Qwen3.8（Flash Next）、DeepSeek-V4、GLM-5.3（在 `vllm/model_executor/models/`）。
- DFlash2 投机解码编排（draft/verify/selector/CUDA Graph）。
- 量化 config / 路由（NVFP4、MXFP4、QPN 的 Python 侧）。
- CI / bench / docs（SM70 验证记录）。
> 下一步：`git diff --name-only 4ff865c38..HEAD` 过滤 `*.py` 与 `models/`，归类到 `core-changes.md`。

## 5. 与 llama.cpp 的映射（初步判断，待验证）
- **FlashAttention V100**（`flash-attention-v100/` + `csrc/attention/sm70_*`）→ llama.cpp `ggml-cuda/fattn.cu`（Volta FA 路径）。
- **量化 GEMV/GEMM for SM70**（marlin / turbomind）→ llama.cpp `ggml-cuda/mmq.cu`/`mmq.cuh`（decode 量化 GEMV，正是 §0.2 确认的"缺 Volta 配置"热点）。
- **Volta MMA 基础设施**（volta_mma.cuh / flash_v100_traits.cuh）→ 指导 llama.cpp 在 Volta 上如何组织 HMMA m8n8k4（双缓冲 / swizzle / fragment 映射）。
- **KV 宽加载**（Layer 1，paged_kv_utils / fp8_kv_utils）→ 指导 llama.cpp KV 访存（合并 / 向量化 / 减少地址运算）。
- 注：1cat 的量化格式（NVFP4/MXFP4）V100 无原生 FP8/FP4，靠软件 dequant；llama.cpp 用 GGUF 量化（Q4_K/Q8_0 等），思路不同但"喂满 HMMA + 减少 HBM 往返"的哲学一致。

## 6. 状态
- [x] 分叉点 + 差异规模 + CUDA 代码区定位
- [ ] `core-changes.md`：逐块深挖技术细节（对照参考文档 + 读实际 kernel 代码）
- [ ] Python 侧（模型 / DFlash2 / 量化路由）归类
- [ ] llama.cpp 落地清单（哪些改动可直接借鉴到 `ggml-cuda`）
