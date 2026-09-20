# 本期工作计划（V100/SM70：学习 1cat-vLLM → 优化 llama.cpp）

> 缩圈结构：Phase(大) → Milestone(中) → Step(小) → 验收。
> 哲学：**原理上通过 → 实测 → 一点点改进**（没有"最优/更佳"，固定任务就是这样）。
> 权威约定：`QWEN.md` + `llama.cpp/AGENTS.md`（冲突以 AGENTS.md 为准）。
> 状态更新：本文件随执行推进，每完成一个 Milestone 更新 `[ ]`→`[x]` 并补数据。

## 已定决策（用户 2026-09-20 授权）
- 计划/规则文件：本文件 `1cat-vllm-v100-study/WORKPLAN.md`。
- **首个优化**：借鉴 **FA-V100 → llama.cpp `ggml-cuda/fattn.cu`**。
- **baseline**：单卡 **GPU2**，`pp512 / tg128`，模型 `Qwen3.8-27B-UD-Q2_K_XL.gguf`（9.2 GiB，已下载+校验）。
- **DFlash2 调研**：串行（不用子代理，主线间隙我做），存档 `dflash2-llama-cpp-research.md`。
- **执行方式**：授权后自主推进；非关键决策我直接定并迭代；**仅"大范围重写/新子系统"停下等用户确认**（AGENTS.md 红线）。

## 贯穿规则（每个 Step 都遵守）
- **先存档再测试**：有成果/结果先存档（数据/文档/变更）再上机测试；测后更新文档。
- **编译双通过**：x86(WSL) + ppc64le(AC922) 均通过才算编译成功。
- **显存红线**：编译先 `free -g`+`nvidia-smi`，`-j` 控 ~16，编中抽查；**永不碰 GPU 0/1/3/4（vLLM）与 llmscope**；GPU 2/5 可自主处置。
- **不凭记忆**：Volta/SM70 细节、head-dim 支持、确切版本等，上网 / 读实际代码查证。
- **不用子代理**：全程主线程串行。
- **正确性优先**：相关 `test-*` 通过、输出一致，不为速度牺牲正确性。

## Phase 0 — 基线（先有基准再谈优化）
- [ ] **M0.1 编译 b11053**（工作目录 `/root/llm/test/v100-opt/`）
  - [ ] M0.1a **WSL x86_64 编译通过（硬前置）**：先查 WSL 环境（cmake/gcc），WSL 内 `cmake -B build && cmake --build build -j`。
  - [ ] M0.1b **AC922 ppc64le 编译**：gcc-toolset-12 + CUDA12.4（配方见 QWEN.md §0.6）；显存红线同上。
- [ ] **M0.2 baseline**：`CUDA_VISIBLE_DEVICES=2 build/bin/llama-bench -m /root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf -p 512 -n 128` → 记 pp512/tg128 t/s；抽查输出非乱码。
- [ ] **M0.3 存档**：写 `study/baseline.md`（数据+配方+环境快照）；更新 QWEN.md §0.5。
- **验收**：x86+ppc64le 双编译通过；baseline 数字落地并存档；`nvidia-smi` 确认 0/1/3/4 未动。

## Phase 1 — 学习 FA-V100（只读 deep-dive）
- [ ] **M1.1** 读 1cat `flash-attention-v100/kernel/`：`flash_v100_traits.cuh`（16-warp/512-thread 块、D→BLOCK_M/N、`kGmemElemsPerLoad=8` 128-bit）、`fused_mha_forward*.cu`（online softmax / QK·PV / D-split）、`paged_kv_utils.cuh`+`fp8_kv_utils.cuh`（Layer-1 宽 paged KV 加载）。
- [ ] **M1.2** 读 `csrc/attention/sm70_v37/`（D=256 重写）、`sm70_79t/prefill*.cu`（75T/79T prefill）。
- [ ] **M1.3** 读 llama.cpp `fattn.cu` Volta 现状：line 199/200（Volta GQA 特判）、381（Volta 分支）、**644**（`volta_mma_available && ne0∉{40,72}` 才用 TC）、**649**（tile：矩阵够大才用 TC）→ 对比 1cat 找差距。
- [ ] **M1.4 产出 `study/core-changes.md`**：FA-V100 核心技巧 + 与 `fattn.cu` 逐项映射 + 差距/可迁移点 + **第一个最小改动建议（我定，标注给用户看）**。
- **验收**：`core-changes.md` 完整，明确第一个改动建议。

## Phase 2 — 首个优化落地（FA-V100 → fattn.cu，最小可行改动）
- [ ] **M2.1** 选"原理可行+改动最小+风险最低"的点（候选：放宽 line 644 head-dim 40/72 排除 / 调 line 649 tile 选择 / 引入 128-bit KV 加载），**先存档设计**（改哪、为何、预期）。
- [ ] **M2.2** WSL 写码 → x86 编译通过（硬前置）。
- [ ] **M2.3** AC922 ppc64le 编译（记录问题 → 回 WSL 修）。
- [ ] **M2.4** 实测：单卡 GPU2 `llama-bench -p 512 -n 128` → 对比 baseline。
- [ ] **M2.5** 正确性：相关 `test-*` + 输出对比（不为速度牺牲正确性）。
- [ ] **M2.6** 存档前后 t/s + 迭代（回 M2.2）。若收益为负 → 回退+记录原因+换下一最小改动（不硬凑）。
- **验收**：t/s 提升或更通用 + 正确性通过 + 存档。

## Phase 3 — 迭代 + 其它优化（按收益逐个走"原理→实现→实测→存档"小循环）
- [ ] **M3.1** MMQ Volta 配置（decode GEMV，§0.2 热点，新增 `mmq-config-volta.cuh`）。
- [ ] **M3.2** FA-V100 其它借鉴（KV 宽加载 / D-split / sparse）。
- 每项独立走 Phase 2 循环，逐项存档。

## Phase 4 — DFlash2 调研（串行，主线间隙，纯调研不改码）
- [ ] 互联网调研 DFlash2 + llama.cpp（投机解码实现/可行性）→ 存档 `study/dflash2-llama-cpp-research.md`。

## Phase 5 — 收尾
- [ ] 汇总 baseline→各优化前后 t/s 总表。
- [ ] 更新 QWEN.md §0.5 + 研究档案索引。

## 留给用户的关键决策点（其余自主）
1. **M2.1 第一个改动方向**：`core-changes.md` 给建议+理由，默认按建议走；用户可否决。
2. 某改动升级为**大范围重写/新子系统** → 停下等用户确认。
3. 其它（`-j` 值、是否补双卡 baseline）自主定。

## 验证方式（端到端）
每个优化 = WSL+AC922 双编译通过 + 单卡 `llama-bench -p512 -n128` 对比 baseline + 相关 `test-*`/输出一致；全程 `nvidia-smi` 确认 GPU 0/1/3/4 未动、vLLM 存活。
