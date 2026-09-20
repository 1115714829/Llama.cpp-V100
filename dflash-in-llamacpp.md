# dflash-in-llamacpp.md — 在 llama.cpp b11053 里跑 1cat 的 DFlash2 draft（2026-09-20）

## 1. 发现

1. 机器上**早已有** 1cat 的 DFlash2 draft GGUF：
   `/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`（1.14 GB，Sep 10 下载）。
2. **llama.cpp b11053 支持 `--spec-type draft-dflash`**（`--help` 类型列表里就有），且能识别这个 draft：
   日志 `common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dflash'`、
   `- n_max=3, n_min=0, p_min=0.00`、`- block_size=8, mask_token_id=248070, n_extract=5, sample_from_anchor=true`。
3. **但它只在非 tensor 切分下能跑**。`--split-mode tensor` 会在 warmup 阶段**硬崩**：
   ```
   ggml-backend-meta.cpp:543: GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) failed
   ```
   栈：`server_context_impl::load_model` -> `common_context_can_seq_rm` -> `llama_decode` -> `process_ubatch`
   -> `ggml_backend_sched_alloc_graph` -> `ggml_backend_buffer_init_tensor`。
   `--split-mode none` 与 `--split-mode layer` **都正常，且两者数字完全一致**。

## 2. 实测（ctx 4096，seed 42，真实散文 prompt，256 tokens，单卡，`libdir-final` = C4+C5）

| spec | tg (eval t/s) | 相对 `none` | 接受率 | mean len |
|---|---|---|---|---|
| `none` | 34.42 | — | — | — |
| `draft-mtp` n=4 | 39.18 | **+13.8%** | 0.330 | 2.32 |
| **`draft-dflash`**（1cat 的 draft） | **39.61** | **+15.1%** | **0.403** | 2.20 |
| `draft-mtp` n=8 | 21.94 | **−36%** | 0.158 | 2.25 |
| `ngram-simple` | 33.74 | −2% | 0.107 | 4.00 |
| `ngram-map-k` | 34.42 | ~0 | 无草稿命中 | — |
| `ngram-mod` | 34.39 | ~0 | 无草稿命中 | — |

## 3. 结论

1. **1cat 的 DFlash2 draft 在 llama.cpp 里能用，而且接受率最高**（0.403 vs MTP 0.330，**+22%**），
   tg 也是最高（**+15.1%**）。=> 这正是"追平 1cat"最直接的抓手：**我们不需要重新发明 draft，机器上就有**。
2. **`n_max` 不是越大越好**：MTP n=8 掉到 **−36%**（接受率从 0.33 腰斩到 0.158）→ **4 附近最优**
   （与生产配置 `--spec-draft-n-max 4` 一致）。
3. **n-gram 类投机在散文上无效**（无可匹配重复）→ 不要在这上面花力气。
4. **明确的阻塞点**：`draft-dflash` 与 `--split-mode tensor` 冲突。而生产要多卡 tensor
   （实测 tensor 比 layer 快约 24%，见 `mtp-sampler-cpu.md`）。
   => **工程目标：让 DFlash draft 图在 tensor 切分下可用**（即 `ggml-backend-meta.cpp:543` 那条断言背后的
   放置/切分逻辑）。这是**中等规模的内核外改动**，按 AGENTS.md 属于"大改动先与用户确认"的范畴。
5. **单卡不受影响**：单卡时 split-mode 无关，直接 `--draft-dflash` 就能拿到 **+15%**。

## 4. 多 prompt 复测：**推翻第 3 节第 1 条**（2026-09-20）

同一设置（ctx 4096、seed 42、`--split-mode none`、256 tokens），3 个不同散文 prompt：

| prompt | `dflash` 接受率 / tg | `mtp4` 接受率 / tg |
|---|---|---|
| p1 GPU 显存层级 | 0.403 / 39.76 | 0.330 / **40.76** |
| p2 灯塔与漂流瓶 | 0.367 / 38.32 | **0.397** / **45.29** |
| p3 数据并行 vs 张量并行 | 0.313 / 35.57 | 0.257 / **35.94** |
| **均值** | **0.361** / 37.88 | **0.328** / **40.66** |

**修正后的结论**：
1. **两者的接受率其实在同一水平**（dflash 0.361 vs mtp 0.328；单 prompt 的 0.403 vs 0.330 是**偶然**，
   p2 上 mtp 反而更高）。单 prompt 就下"dflash 接受率最高"的结论是**过度解读**，已作废。
2. **吞吐上 `draft-mtp` 反而更快（40.66 vs 37.88 t/s，三个 prompt 上都快）**：
   MTP 的 draft 就是模型自带的 NextN 头（**零额外权重读取**），而 DFlash 每步都要读一个 **1.14 GB 的
   独立 draft 模型** —— 在带宽受限的 decode 里，这点开销**大于**它多换来的接受率。
3. ⇒ **生产继续用 `--spec-type draft-mtp --spec-draft-n-max 4` 是对的**（也印证了 n_max=4 的选择）。
4. ⇒ **原本打算修的"dflash × tensor 冲突"（t14）价值大幅下降** —— 因为即便修好，
   dflash 在吞吐上也不赢 MTP。**降级为"已知但不动"**。

## 5. 仍然成立的结论
- `n_max=4` 附近最优，**加大到 8 反而 −36%**（接受率腰斩）。
- **n-gram 类投机在散文上无效**。
- 单卡场景 `draft-dflash` 可用（`none`/`layer` 均可，数字一致），但与 `--split-mode tensor` 冲突
  （`ggml-backend-meta.cpp:543`）。只是如上一条，**它不赢 MTP，所以不值得为它动代码**。

## 6. 教训
**单 prompt 的投机解码结论不可信**：接受率随 prompt 摆动 0.25~0.40，足以让排序翻转。
任何投机解码的对比必须 ≥3 个不同 prompt（固定 seed）+ 报接受率。

## 8. MTP 旋钮扫描（3 prompt 均值，ctx 4096，seed 42，`--split-mode none`，单卡）

| 变体 | tg 均值 (t/s) | 接受率均值 |
|---|---|---|
| `--spec-draft-n-max 3` | 39.27 | 0.351 |
| **`--spec-draft-n-max 4`（现状 / 生产）** | **40.66** | 0.328 |
| `--spec-draft-n-max 5` | 37.05（**−8.9%**） | 0.281 |
| `--spec-draft-n-max 6` | 33.85（**−16.8%**） | 0.237 |
| `n-max 4 --spec-draft-p-min 0.30` | 39.17（−3.7%） | **0.365**（接受率↑ 但吞吐↓） |
| `n-max 4 --spec-draft-n-min 2` | 40.66（**与 n_max=4 逐位相同**） | 0.328 |

**结论**
1. **`n_max=4` 就是最优点**：3 略低（−3.4%）、5 与 6 明显更差（−8.9% / −16.8%），且接受率同步下降。
   ⇒ **生产的 `--spec-draft-n-max 4` 不要改**。
2. **`p_min=0.30` 把接受率抬到 0.365（相对 +11%）但吞吐反而降 3.7%** —— 剪枝带来的收益抵不过
   少投机造成的损失 ⇒ **不用**。（也说明"接受率更高"不等于"吞吐更高"。）
3. **`n_min=2` 是彻底的 no-op**（数字与 `n_max=4` 逐位相同）⇒ 在 n_max=4 下不起作用。
4. ⇒ **参数/开关这条路已经榨干了**：没有任何旋钮是净收益。要再往前走，只能靠
   **"更省或更准的 draft"**（工程活：改 draft 结构或降低 draft 的每步读取开销），而不是调参数。

## 10. 更正与重大发现（2026-09-20 晚，按官方条件复现）—— **第 4/5 节的结论作废**

### 10.1 互联网查证结果（`z-lab/Qwen3.8-27B-DFlash2-GGUF` model card）
- 我们的 `/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`
  **就是官方 DFlash2 draft**（`incoai/Qwen3.8-27B-DFlash2-GGUF` 的镜像）。
- DFlash2 = **block-diffusion drafter**："predicts a whole block of tokens in a single pass and keeps the
  top candidates at every position. A lightweight **selector** then traces one coherent path through them."
- 官方要求 **llama.cpp PR #27342**；我们的 b11053 **已包含**（`src/models/dflash.cpp` 的
  `build_dflash2_conv`/`build_dflash2_selector`/`dflash2_lattice`）。
- **官方启动参数**：`--spec-type draft-dflash --spec-draft-n-max 7`
- **官方验收长度（Acceptance Length）**：BF16 **5.28** / Q8_0 **5.13** / **Q4_K_M 5.39**；
  条件 = **temp 1.0 / top-p 0.95 / top-k 20 + xhigh reasoning**、GSM8K prompt、**2048** 新 token。

### 10.2 按官方条件复现（单卡 GPU0，Q2_K_XL target，3 个 reasoning 型 prompt，n_predict 1024，seed 42）
| 变体 | tg（3 次） | 平均 | **平均接受长度** | 接受率 |
|---|---|---|---|---|
| `--spec-draft-n-max 3` | 57.62 / 63.78 / 66.43 | **62.6 t/s** | 3.35 | 0.69–0.86 |
| **`--spec-draft-n-max 7`（官方）** | 67.78 / 64.91 / 67.32 | **66.7 t/s** | **4.92** | 0.54–0.57 |

**结论（推翻第 4/5 节）**
1. **同一 draft、同一 n_max=3，仅把内容换成 reasoning 型 + 采样换成官方值：39.8 → 62.6 t/s（+57%）**，
   平均接受长度 2.2 → 3.35。⇒ 之前"dflash 不如 MTP"的结论**是短散文 + temp 0.7 的工况造成的假象**。
2. **`n_max=7` 在官方条件下更好**（66.7 vs 62.6 t/s），且**平均接受长度 4.92 已达到官方 Q4_K_M 的 5.39 水平**
   ⇒ **我们的 DFlash2 draft 表现正常，没有"喂不满"的问题**（上一节"n_max=7 更差"作废）。
3. **66.7 t/s 是 MTP 最好成绩（40.66）的 1.6 倍**，方向与用户给的 1cat FP8 目标（120–200 t/s）一致。
4. 采样参数会显著改变接受率（temp 0.7/短散文 → 0.33；temp 1.0/reasoning → 0.55）。
   ⇒ 报接受率必须连**采样参数与 prompt 类型**一起报，否则数字不可比。

### 10.3 下一步（已按此推进）
- **Phase 1**：解决 `draft-dflash` × `--split-mode tensor` 的崩溃（`ggml-backend-meta.cpp:542-544`），
  让 DFlash2 能用在多卡生产配置上 —— 现在这一条的价值被上面的 +57% 大幅抬高了。
- **先行低成本对比**：`layer + dflash n_max=7` vs `tensor + mtp n_max=4`（3 卡、生产模型、同 prompt/seed），
  看"更好的 draft"能否补回 layer 相对 tensor 的约 24% 损失，从而判断 Phase 1 值不值得。

## 12. 多卡矩阵（2026-09-20 晚，3 卡同 NUMA 0/1/2，**生产模型 Q4_K_M**，ctx 8192，官方采样 temp1.0/top_p0.95，reasoning prompt，seed 42，n_predict 512）

| 配置 | tg（3 次） | 均值 | 接受率 | mean len |
|---|---|---|---|---|
| **`tensor + draft-mtp n_max=4`**（当前可用） | 54.81 / 56.74 / 64.36 | **58.6 t/s** | 0.56–0.69 | 3.25–3.78 |
| `layer + draft-dflash n_max=7` | 14.52 / 14.39 / 15.03 | **14.6 t/s** | **0.041** | 1.27–1.33 |
| `tensor + draft-dflash n_max=7` | **崩** | — | — | — |

**结论**
1. **DFlash2 的多卡支持不成熟，两种切分各坏一种**：
   - `tensor` -> 硬崩（`ggml-backend-meta.cpp:543` 的 axis-0 断言）；
   - `layer` -> **静默失效**（接受率掉到 **4%**、mean len 1.27；不是"layer 慢 24%"能解释的）。
2. **但生产模型（17.23 GiB）必须多卡** ⇒ 目前多卡只能用 MTP（**58.6 t/s**），
   而单卡 DFlash2 已经是 **66.7 t/s**。⇒ **把 DFlash2 修到多卡可用 = 当前最高价值的工程**。
3. 顺带印证：同一 MTP 配置，接受率随内容从 0.33（短散文）升到 0.56–0.69（reasoning），
   tg 从未 40 升到 58.6 ⇒ **报 tg 必须连接受率+prompt 类型一起报**。

## 14. Phase 1a 根因确认 + 上游修复（2026-09-20 晚，互联网查证）

### 14.1 根因（我们的诊断 + 上游 diff 互相印证）
- 我们的诊断输出：
  ```
  E meta: per-row op does not support an axis-0 split src0: tensor 'node_650' op 70, src0 ne = [248320 2 1 1]
  ```
  `op 70` 对应 **`GGML_OP_TOP_K`**（`ggml.h` 的 op 表里 TOP_K 就在 ARGSORT 之后）；`[248320, 2]` 是
  selector 的"候选 × 成对 lattice"形状。
- 上游 PR #27858 的 diff 说明了**为什么**：
  > Under tensor parallelism the `lm_head` is split along the vocabulary, which the in-graph
  > selector cannot consume, so DFlash2 ranks on the CPU and reads these tables straight from the GGUF file

  ⇒ **`--split-mode tensor` 把 lm_head 沿词表切分，而 DFlash2 的 selector 要在词表维做 TOP_K + 成对打分，
  切分后无法本地完成** -> `handle_per_row` 断言拒绝 axis-0 切分的 src0。**这是上游已知、仍开放的缺陷**
  （issue #28777「DFlash2 + --split-mode tensor 崩溃（layer 正常）」、ROCm 版 #27829）。

### 14.2 上游修法（PR #27858，**Draft 未合并**；我们 b11053 里没有）
关键改动（`src/models/dflash.cpp`）：
```cpp
const bool use_cpu_selector = params.split_mode == LLAMA_SPLIT_MODE_TENSOR;
dflash_selector_prev = create_tensor(..., use_cpu_selector ? TENSOR_SKIP : 0);
dflash_selector_next = create_tensor(..., use_cpu_selector ? TENSOR_SKIP : 0);
...
if (model.dflash_selector_hidden && model.split_mode() != LLAMA_SPLIT_MODE_TENSOR) {
    build_dflash2_selector(*this, model, inp_tokens);   // 图内 selector 只在非 tensor 下建
}
```
配套：`common/speculative.{cpp,h}` 新增 CPU 侧 selector（`load_dflash2_selector()` /
`build_dflash2_selector_cpu()`，+246 行）与 `common_speculative_is_block_draft` /
`common_speculative_block_draft_n_ubatch`；`src/llama-model.{cpp,h}` + `llama-ext.h` 新增
`llama_model_get_split_mode()`；另有"限制 draft ubatch 省 compute buffer"与"修 prefill 掉速"两个提交。

**已核实：`use_cpu_selector` / `load_dflash2_selector` / `build_dflash2_selector_cpu` /
`common_speculative_is_block_draft` 在我们 b11053 中全部不存在**（`TENSOR_SKIP` 有，是别的模型在用）
⇒ **Phase 1a = 移植 PR #27858**。

### 14.3 校准数据（上游讨论里拿到的、很有用）
- `treo` 在**单张 RTX 3090** 上：MTP overall **57.79 t/s**（接受率 0.63）vs DFlash2 n=4 **62.21 t/s**（0.615）
  ⇒ **单卡上 DFlash2 仅比 MTP 快约 8%**（他的原话 "not significantly better than MTP"）。
- `Nathanw1014`（gfx1151，Gutenberg 散文）：decode t/s @ depth 0/8k/32k —— base 11.81/11.44/10.54；
  **DFlash2 n=4: 26.39/21.58/21.11**；DFlash2 n=7: 25.18/21.46/16.32
  ⇒ **深度越大 n=4 越优于 n=7（32k 时 n=4 领先 29%）**；DFlash v1 到 32k 已无收益。
- 官方 GGUF 有坑：ngxson（维护者）说 **Aug 27 前生成的 GGUF 必须重转**，`incoai` 那批已知坏，
  建议用官方 `z-lab`。（我们的是 Sep 10 的，理论上没这个问题；但若出现"接受率极低"要先怀疑它。）
- 正确性 issue **#27407**：batched verification 下 greedy 输出与无投机基线发散（DFlash2 会放大）——
  引用数字时要注意"lossless"只在特定条件下成立。

### 14.4 对 V100 的特别提示
该修法把 selector 搬到 **主机（CPU）** 上跑。本机是 **POWER9 176 核**，
所以主机侧 top-k/lattice 的绝对开销可以接受；但**仍需实测**它与"tensor 切分省下的时间"谁更划算。
另外按上面的深度规律，**256K 场景应优先试 n_max=4 而不是 7**。

- [ ] **Phase 1a**：修 `tensor + dflash` 的 `handle_per_row` axis-0 断言。
- [ ] **Phase 1b**：查 `layer + dflash` 接受率仅 4% 的原因（draft 的 hidden-state 注入 / 卷积 / selector 在层切分下的放置）。
- [ ] Phase 2：draft 放置与量化（`-ngld`/`-devd`/`--spec-draft-override-tensor`、`llama-quantize` 更小量化）。
- [ ] 验收：单卡 66.7 t/s 已达标方向；多卡修好后在 256K 上做 L3 抽查，对齐 1cat 的 120-200 t/s。

- 生产投机配置（`draft-mtp --spec-draft-n-max 4`）**已经是本仓库可调范围内的最优**。
- 与 1cat（DFlash2）的差距来自**他们的 draft 是专门训练/工程化的**，
  而不是我们少调了某个开关。要追平需要**大工程**（新 draft 路径），不是小改动。
- 因此本项目在"投机解码"这条线上的现实产出是：**把配置固化并文档化 + 证明这不是参数问题**。

## 16. Phase 1a 完成：PR #27858 移植成功 —— 但结论反转（2026-09-20 晚）

### 16.1 移植内容与构建
- 移植上游 PR #27858 的全部 7 个文件：`common/speculative.h`（`common_speculative_is_block_draft` +
  `common_speculative_block_draft_n_ubatch` + `<algorithm>`）、`common/speculative.cpp`（`is_dflash2_cpu` +
  `sel_next/sel_prev/sel_hidden` + `load_dflash2_selector()` + `build_dflash2_selector_cpu()` + 块走查分支 +
  ubatch 上限）、`common/common.cpp`（fit 估算同步）、`src/llama-ext.h` + `src/llama-model.cpp`
  （`llama_model_get_split_mode`）、`src/llama-model.h`（注释）、`src/models/dflash.cpp`
  （`use_cpu_selector` = `params.split_mode == LLAMA_SPLIT_MODE_TENSOR` → `TENSOR_SKIP`；
  `res->t_h_nextn = cur`；图内 selector 只在 `!= LLAMA_SPLIT_MODE_TENSOR` 时构建）。
  同时**撤销**了 `ggml/src/ggml-backend-meta.cpp` `handle_per_row` 里的临时诊断（已恢复为 b11053 原样）。
- 构建：AC922 gcc-toolset-12 + CUDA 12.4，`cmake --build build --config Release -j8 --target llama-server llama-bench`，
  **无 error / FAILED**，`libllama.so`、`libggml-base.so`、两个可执行文件均重新链接。
  （用户已指示：本机空载时编译并行度从 **`-j82`** 起步。）
- 二进制校验（**有效性自检**）：快照到 `/root/libdir-27858`。
  `libggml-cuda.so.0.24.0` md5 = `b4b467736ca4a58ac1c747c7ff14fb00` —— **与 `libdir-final` 逐位相同**
  ⇒ 本次改动**没有动任何 CUDA kernel**，唯一变量就是新增的 CPU 侧 selector 代码；
  `libllama.so.0.4.1` md5 = `c2bb2b3062a07779b7d334230ce5ac06` ≠ `libdir-final` 的 `811b90a9252744bc853d9cfeb5f9f4f0`
  ⇒ 新代码确实进了被加载的库。

### 16.2 崩溃已修复（1 卡，Q2_K_XL target，`--split-mode tensor` + `draft-dflash`）
| | 旧 build | 新 build（libdir-27858） |
|---|---|---|
| 结果 | load 阶段 SIGABRT（`ggml-backend-meta.cpp:543` axis-0 断言） | **health ok=1，仅 5 s**，无任何断言 |

**CPU 路径生效的两条直接证据**（新 build 日志）：
```
I common_speculative_init_result: capping draft context ubatch from 512 to 64 (block draft)
W model has unused tensor selector_predecessor.weight (size = 35758080 bytes) -- ignoring
W model has unused tensor selector_successor.weight (size = 35758080 bytes) -- ignoring
```
第二条即 `TENSOR_SKIP` 生效 —— selector 表不再进显存，改由主机从 GGUF 直接读取。

### 16.3 三卡矩阵：生产模型 Q4_K_M（0/1/2 同 NUMA，ctx 8192，官方采样 temp 1.0 / top_p 0.95 / top_k 20，
seed 42，3 个 reasoning prompt，n_predict 512，**四个分支全部用同一个 libdir-27858**）

| 配置 | tg（3 prompt，t/s） | 均值 | 接受率 | mean len |
|---|---|---|---|---|
| **`tensor + draft-mtp n=4`**（生产基线） | 54.93 / 56.77 / 64.37 | **58.7** | 0.565 / 0.591 / 0.694 | 3.25 / 3.36 / 3.78 |
| `tensor + draft-dflash n=7`（原本必崩） | 13.04 / 13.60 / 14.90 | **13.8** | **0.031 / 0.037 / 0.053** | 1.22 / 1.26 / 1.37 |
| `tensor + draft-dflash n=4` | 15.56 / 16.08 / 17.87 | **16.5** | 0.049 / 0.058 / 0.091 | 1.20 / 1.23 / 1.36 |
| `layer + draft-dflash n=7` | 15.35 / 14.51 / 15.02 | **15.0** | 0.041 / 0.040 / 0.048 | 1.28 / 1.27 / 1.33 |

⇒ **崩溃修好了**（`tensor + dflash` 不再 abort），但**生产模型上的接受率仍只有 ~4%**，
且**与 split mode 无关**：`layer`（图内 selector）与 `tensor`（CPU selector）落到同一水平。

### 16.4 隔离实验：把 target 换成 Q2_K_XL（同 draft、同 libdir、同 3 卡 tensor、同采样/seed/prompt，n_predict 1024）

| 配置 | tg（3 prompt，t/s） | 均值 | 接受率 | mean len |
|---|---|---|---|---|
| **`tensor + draft-dflash n=7`** | 46.75 / 50.91 / 67.23 | **54.9** | **0.473 / 0.516 / 0.746** | **4.31 / 4.61 / 6.22** |
| `tensor + draft-mtp n=4` | 66.25 / 68.53 / 81.11 | **72.0** | 0.599 / 0.600 / 0.787 | 3.40 / 3.40 / 4.15 |

**结论一：移植是正确的。** 同一条 CPU selector 代码路径、同一个 draft、同样 tensor 切分 →
mean len **4.31–6.22**，与官方 Q4_K_M 的 **5.39** 同水平（此前单卡官方条件 4.92）。
⇒ **排除了"移植有 bug"**（否则同一条路径不可能在换 target 后突然变好）。

**结论二：~4% 的真因是 target 模型不兼容。** Q2_K_XL target → mean len 4.3–6.2；
`TurboFCFusion-...-Q4_K_M` target → mean len 1.2–1.4。**同一个 draft、同一份代码**。
- 机制推断：DFlash2 draft 训练在**基座 Qwen3.8-27B** 上，它消费 target 的**指定层输入 embedding**
  （`n_extract=5`）与 target 的 **lm_head**。TurboFCFusion 是合并/微调变体（自带 MTP NextN 头、
  "NEO-CODER" 输出头），表示与输出分布已漂移 → selector 的候选排序基本失效。
- ⇒ **DFlash2 无法用于生产 TurboFCFusion 模型**（除非重训 draft）。

**结论三：即使在兼容 target 上，DFlash2 也不比 MTP 快。** mean len 更高（4.3–6.2 vs 3.4–4.2）
但 tg 更低（46.8/50.9/67.2 vs **66.3/68.5/81.1**）—— 因为 DFlash2 每步要额外读 **1.14 GB** 独立 draft，
在带宽受限的 decode 里这点开销**大于**多换来的接受率。与 §4 的散文三 prompt 结论一致（37.88 vs 40.66）。

### 16.5 更正：§10.2 的 "dflash 66.7 > mtp 40.66" 作废
那一次的两侧用的是**不同 prompt**（dflash 用 reasoning 型、mtp 用散文型），**不可比**。
本次同 prompt 对照（16.4）显示 **MTP 反而快约 31%**。
⇒ 结论改为：**在 V100 上，MTP 是更快的投机解码方案。**

### 16.6 对项目最终目标的影响
"追平 1cat（DFlash2 + FP8，120–200 t/s）" 这条路在**本机不成立**，三条独立理由：
1. **生产 target 不兼容 DFlash2**（16.4，接受率 4%）；
2. **兼容 target 上 DFlash2 仍慢于 MTP**（16.4，带宽受限）；
3. **V100 无 FP8**（§0.1）—— 1cat 的 120–200 t/s 是 FP8 数字，硬件前提不同。
⇒ 投机解码这条线的现实产出 = **把 MTP 配置固化并文档化**（`draft-mtp --spec-draft-n-max 4`）
+ **用数据证明 DFlash2 不是本机的答案**。Phase 1a 的工程价值是**修好上游崩溃、并验证 CPU selector 的正确性**，
不是提速。

### 16.7 复现命令与日志
- `/root/p27-build-27858.sh`（构建）→ `/root/p28-verify27858.sh`（崩溃复现 + 二进制校验）
  → `/root/p29-mg27858.sh`（三卡矩阵）→ `/root/p30-iso27858.sh`（target 隔离实验）。
- 日志：`/tmp/verify27858.log`、`/tmp/verify27858-server.log`、`/tmp/mg27858.log`、`/tmp/mg27858-*.log`、
  `/tmp/iso27858.log`、`/tmp/iso27858-*.log`。
- 注意：`--split-mode tensor` 下会有一条**无害**警告
  `common_fit_params: failed to fit params to free device memory: llama_params_fit is not implemented for
  SPLIT_MODE_TENSOR, abort`（是 fit 功能未实现，不是崩溃；grep "abort" 时别误判）。

### 16.8 遗留
- [x] **Phase 1a**：`tensor + dflash` 的 `handle_per_row` axis-0 断言 —— **已修复并验证**。
- [~] **Phase 1b**：`layer + dflash` 的 4% —— **已定性**：不是 split mode 问题，
      是**生产 target 与基座 draft 的不兼容**。若要继续，需要"为 TurboFCFusion 重训/适配 draft"（大工程，非 kernel 活）。
- [ ] Phase 2：256K 场景下 `n_max=4` vs `7` 的深度规律（上游社区：32k 深度时 n=4 领先 29%）—— 但前提是先有兼容 target。

