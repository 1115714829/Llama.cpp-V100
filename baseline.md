# baseline.md — llama.cpp b11053 V100 baseline（Phase 0）

> 首个优化前的基准。**所有后续优化都对比此 baseline**（同模型 / 同 batch / 同单卡 GPU2 / 同构建配置）。
> 存档时间：2026-09-20。

## 基准数据
| test | t/s |
|---|---|
| **pp512** | **741.16 ± 13.74** |
| **tg128** | **36.23 ± 0.09** |

- **模型**：`Qwen3.8-27B-UD-Q2_K_XL.gguf`（arch 检出 `qwen35`，27.32B 参数，9.14 GiB，Q2_K 量化，9,828,981,664 字节完整）
- **GPU**：单卡 **GPU2**（Tesla V100-SXM2-16GB，CC 7.0，15360 MiB VRAM），`CUDA_VISIBLE_DEVICES=2`
- **backend**：CUDA，`ngl -1`（全部层上 GPU）
- **命令**：`./build/bin/llama-bench -m /root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf -p 512 -n 128`

## 构建 / 环境
- 源码：b11053（0.4.1-dev，ggml 0.24.0，commit 1af554f8f），从 Windows 工作树 tar 同步到 AC922 `/root/llm/test/v100-opt/llama.cpp`。
- 编译：gcc-toolset-12 + CUDA 12.4（nvcc 12.4.131），`-DCMAKE_BUILD_TYPE=Release -DLLAMA_CUDA=ON`，`CMAKE_CUDA_ARCHITECTURES=70-real`，`-j16`。
- **x86（WSL）+ ppc64le（AC922）双编译通过**（均 0 error）。
- AC922：ppc64le POWER9，176 核，6×V100，driver 550.54.15。**GPU 0/1/3/4（vLLM）全程未动**（编译/基准期间 nvidia-smi 均稳定 ~15.1G）。

## 模型 config（已核实，`src/models/qwen35.cpp` + GGUF）
- **Qwen3.5-27B（arch `qwen35`），n_layer=64（`case 64 → LLM_TYPE_27B`），n_embd=5120**。
- **混合（Flash Next）架构**：`full_attn_interval=4` → **48 层线性/递归注意力（GDN）+ 16 层全注意力（FA/fattn.cu）**。
- ~~全注意力层：head_dim=128（∉{40,72}→TC 可用）、n_head=40、n_kv_head=8（gqa_ratio=5）。~~
  - decode→VEC kernel（SIMT），prefill→MMA_F16（TC）。详见 `core-changes.md` §1/§4。
- 🛑 **2026-09-20 DSH 审计更正（重要，此前整条结论都建立在错误几何上）**：
  实测 1cat 侧 `/root/llm/models/Qwen3.8-27B-FP8/config.json`：
  ```
  hidden_size: 5120            num_hidden_layers: 64
  head_dim: 256                 num_attention_heads: 24
  full_attention_interval: 4    num_key_value_heads: 4
  layer_types: linear_attention ×3 + full_attention ×1（循环）
  ```
  ⇒ 全注意力层是 **head_dim=256 / 24 Q 头 / 4 KV 头（GQA=6）**，**不是** 128/40/8（GQA=5），**也不是** `core-changes.md` §8 反推的 40/4。

  **★★ 硬证据（2026-09-20 DSH 审计，直接读 GGUF 元数据，最强口径）** —— 用 1cat venv 的 python + `gguf` 包读**我们实际加载的那份** `Qwen3.8-27B-Q8_0.gguf`：
  ```
  qwen35.attention.head_count    = 24      (不是 40)
  qwen35.attention.head_count_kv = 4
  qwen35.attention.key_length    = 256     (不是 128)
  qwen35.attention.value_length  = 256
  qwen35.embedding_length        = 5120
  qwen35.block_count             = 65      (64 主体 + 1 MTP)
  qwen35.full_attention_interval = 4
  qwen35.ssm = { conv_kernel 4, group_count 16, inner_size 6144, state_size 128, time_step_rank 48 }
  ```
  ⇒ 与 1cat 侧 `config.json`（24 / 4 / head_dim 256）**完全一致**。
  ⇒ 后果：① 所有按 128/40/8 做的 FA 形状判断失效（这也是"FA 只占 3–6%"结论不可信的原因之一）；
     ② 第三方 llama.cpp V100 补丁（`jackinthebox52/qwen38-v100-serve`）正是针对 **24/4 → GQA=6** 做的 `ncols2=3` 打包，
        128K 时 KV DRAM 流量 26.37 → 8.59 GB/token、no-MTP decode +44.9%（见 `EXTERNAL-REFS.md`）；
     ③ 这解释了 1cat 为何专门写 `csrc/attention/sm70_v37`（**D=256 重写**）。
  注：GGUF 用非标准 key（`qwen35.attention.head_count` 等），落地前应以 GGUF 元数据再核对一次。

## 备注
- pp（prefill）通常 attention/计算 bound；tg（decode）在 V100 上通常 **weight GEMV 内存带宽 bound**。
- 首个优化（FA-V100 → fattn.cu）预期主要影响 **pp**；若 tg 是 weight-bound，tg 提速应转向 MMQ（Phase 3）。
