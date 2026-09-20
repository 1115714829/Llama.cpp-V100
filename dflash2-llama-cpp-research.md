# DFlash2 调研（1cat-vLLM → llama.cpp 映射）

> 来源：1cat-vllm 仓库 grep + `vllm/v1/worker/gpu/spec_decode/dflash2/` 源码（speculator.py 1268 行等）。
> 结论先行：DFlash2 是 1cat-vLLM 的**投机解码（speculative decoding）**方法，不是单点 V100 kernel 优化；落到 llama.cpp 是**大特性**（投机解码子系统），非本轮 V100 kernel 优化范围。

## 1. DFlash2 是什么
- **投机解码**：draft（草稿模型 / n-gram 快速出多个候选 token）+ verify（target 模型一次并行验证 + rejection sampling 接受）。加速 decode（一次前向出多 token）。
- 1cat 里是 `DFlash2Speculator`（继承 `DFlashSpeculator`），SM70/V100 专项调优（3447 处 `dflash` 匹配，含 `sm70_dflash2_*` 基准 + kernel）。
- 子组件（`vllm/v1/worker/gpu/spec_decode/dflash2/`）：
  - `speculator.py` — `DFlash2Speculator.propose`（draft + verify 主逻辑）；`_proposal_nucleus_logits`（top-p 采样 draft）；`_selector_walk_kernel`（选择候选）。
  - `ngram_assist.py` — `DFlash2NgramAssist`（n-gram 草稿辅助，从历史 token 复用）。
  - `lookup.py` — `suffix_lookup` / `fuse_draft` / `_point_mass_draft_logits_kernel`（draft 融合）。
  - `sparse_rejection.py` — top-k/top-p **rejection sampling**（验证 + 接受/拒绝）。
  - **SM70 专项**：`_requires_sm70_tail(device, num_steps)` — 仅 CC (7,0)（V100）时，最终依赖 selector slot 需要**独立 kernel**（Volta 的依赖/同步特性处理）。

## 2. 与 V100 优化的关系
- DFlash2 是 **decode 加速**（投机解码，减少 decode 步数），与 V100 kernel 优化（mmvq GEMV / FA / GDN）**正交**——DFlash2 让每步前向更快被"摊薄"，kernel 优化让每次前向更快。两者可**叠加**。
- DFlash2 的 SM70 专项（`_requires_sm70_tail`）印证 1cat 对 V100 的依赖/同步做了专门处理（Volta 无异步依赖原语，需独立 kernel）。

## 3. 映射到 llama.cpp
- llama.cpp 有**投机解码**基础设施（`llama-cli`/`llama-server` 的 `-md <draft-model>` + 部分 n-gram/lookahead 投机），但**无 DFlash2 这套**（n-gram assist + suffix_lookup + top-k/top-p sparse rejection + SM70 tail）。
- 移植 DFlash2 需要：投机解码 draft+verify 循环、n-gram draft、rejection sampling、SM70 tail kernel —— 是**投机解码子系统级**的工作，非单 kernel。
- **评估：大特性 / 高工作量**，且依赖 llama.cpp 现有投机解码框架的成熟度。不是 V100 kernel 优化的一部分。

## 4. 结论 / 建议
- 本轮 V100 专项优化的重点应是 **kernel 级**（mmvq GEMV nwarps[已落地 +3.4%]、FA、GDN、overhead/CUDA-graph）。
- DFlash2 作为**独立的 decode 加速方向**单独跟踪：若未来要做，先评估 llama.cpp 投机解码框架 + 移植成本（建议单独立项，不并入 V100 kernel 优化）。
- 1cat 的 `_requires_sm70_tail`（Volta 依赖处理）可作为 llama.cpp 在 V100 上做投机解码/多 kernel 依赖时的参考。
