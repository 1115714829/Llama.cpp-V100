# CODEX-ARCHIVE-INDEX.md — 早期 Codex 研究资料索引（工作区之外）

> 用户 2026-09-20 提到："我之前应该是让他专门做了一个文件夹，其内容就是筛选过后的针对 1cat-vllm 的专项改动代码。"
> 已定位：**不在本工作区**，而在 `C:\Users\a1115\Documents\Codex\` 下的两个日期目录。
> 本文件**只做索引**（不复制其 20 MB+ 的 JSON 证据），并标注**哪些结论不能直接沿用**。

---

## 1. 位置与规模

| 路径 | 内容 |
|---|---|
| `C:\Users\a1115\Documents\Codex\2026-09-05\https-github-com-1catai-1cat-vllm\` | **1cat-vLLM 本体工程**：60+ 篇报告 + 大量实验 JSON |
| ↳ `outputs\llama-v100-implementation-prep-20260906.md` | ★ **llama.cpp V100 支线实施准备**（本轮已读，见 §3） |
| ↳ `outputs\llama-v100-impl\` | **空目录**（原计划的实现落点，未产出内容） |
| ↳ `outputs\`（其余） | `benchmark-report-200k/256k-*.md`、`dflash2-pp-bottleneck-investigation-*.md`、`dflash-draft-kv-acceptance-research-*.md`、`dflash-draft-weight-fp8-feasibility-*.md`、`moe/tp4/pp6-*.md`、`qiankun-source-screening-v3c-*.md`、`six-agent-acceptance-20260906\`（大 JSON 验收证据）等 |
| ↳ `work\` | **1cat 源码副本**：`offloadreview_awq_sm70_gemm.cu`（482 KB）、`offloadreview_gpu_model_runner_v1.py`（566 KB）等 |
| `C:\Users\a1115\Documents\Codex\2026-09-06\llamacpp-dflash2-v100-research\` | AC922 环境包/离线构建（`AC922-环境包交付.md`、`assemble_local.py`、`dependency_audit_notes.md`、rpm 闭包审计、`ac922_known_hosts` 等） |

## 2. ⚠️ 为什么不能直接沿用它的结论

该批资料基于**与现在不同的前提**：

| 维度 | Codex 期（2026-09-05/06） | 现在（2026-09-20，DSH） |
|---|---|---|
| llama.cpp base | `662a0b0121a53c2…` / `b10793` | **b11053**（`1af554f8f`）+ C4/C5/PR#27858 |
| 卡数 | 六卡（tensor split） | **1–6 卡自由**（用户明确，跨岛非瓶颈） |
| 投机 | MTP4（内置 NextN） | **DFlash2 n=7**（外部 draft） |
| 首要目标 | **cold TTFT ≤ 210 s** + 64K/128K ≥80、256K ≥45 tok/s | **追平 1cat（DFlash2 同级效果）**，短上下文迭代 + 长上下文验收 |
| 量化 | Q8_0 / Q8_0 KV | Q8_0（标尺）/ Q4_K_M（生产） |

⇒ **只取方法、候选补丁与证据，不沿用其结论与目标值。**

## 3. 该批资料里最有价值的两条（已并入 `1CAT-PORT-BACKLOG.md` §3）

1. **第三方 llama.cpp CUDA 候选**（源码锚 `mistrjirka/llama.cpp` 分支 `qwen38-lossless-agent-cache`，HEAD `e6920c8b64b…`）：
   - **GDN 四列复用**：`bad7e3952650e0abf86fe6aa2282748730b982cb`，`ggml/src/ggml-cuda/gated_delta_net.cu`，+171 行；SM70、state 宽 128、scalar gate、`n_tokens>1`，每 warp 同更 4 列，复用 q/k/g/beta 读取；**仍按 token 串行，不是 chunked GDN**。
   - **Volta D256 FlashAttention**：`e793205e…` → `e75d47ad…` → `45c26d80…`，`ggml/src/ggml-cuda/fattn-mma-f16.cuh`；D256/ncols64 的 Q 入 shared、K/V scratch 拆分、combine 窗口压缩、有条件 2CTA。
   - 该文档还给了起始文件的 SHA256 与"上游基点逐字相同"的比对结论（有利于低冲突提取），并**明确说尚未 `git apply --check`、未编译**。
2. **权重复用路线的静态缺口**：`src/llama-context.cpp` 只通过 backend registry 查 `ggml_backend_cuda_set_prefill_reuse`，而 tensor split 下 `ggml-backend-meta.cpp` 建 meta device 时 `reg=nullptr` ⇒ **tensor 下该参数到不了真实 CUDA context**（代码级推断）。
   ⇒ 与前缀/权重预取类优化相关，**与我们 Phase P2/P5 的编排改动同源**，值得重测。

## 4. 使用建议

- 需要细节时**按路径直接读**（本索引给出文件名即可定位）；**不要把整棵目录复制进工作区**（体积大、且含旧结论）。
- 引用其数字时，必须标注"**Codex 期、六卡、MTP4、旧 base**"。
- 若要复现其候选补丁，先按 `1CAT-PORT-BACKLOG.md` §5 的模板做**同源 A/B + md5 自检**。
