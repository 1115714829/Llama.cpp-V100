# mtp-sampler-cpu.md — 为什么 MTP draft 的采样器跑到 CPU 上（根因已定位）

> 状态：**根因已定位**（源码级），代价待实测。日期 2026-09-20，base b11053。

## 1. 现象

用 `llama-server --spec-type draft-mtp --spec-draft-n-max 4` 启动时，日志里出现：

```
W spec common_specu: backend offload failed for seq_id=0; using CPU sampler
```

即 MTP draft 的采样链**没有落到 GPU**，退回 CPU 采样。

## 2. 根因：`--split-mode tensor` 下后端采样**未实现**，直接返回 false

调用链：
- `common/speculative.cpp:1411`（MTP 实现 `common_speculative_impl_draft_mtp`，类定义在 `:1330`）：
  ```cpp
  if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
      SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
      llama_sampler_free(chain);
      chain = nullptr;
  }
  ```
  （同样的模式也在 `:504` 的 eagle3 与 `:1037` 的 dflash 路径里。）
- `include/llama.h:1339` → `src/llama-context.cpp:3959` → `llama_context::set_sampler`。
- **`src/llama-context.cpp:1233`**：
  ```cpp
  if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
      static bool warned = false;
      if (!warned) {
          LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
          warned = true;
      }
      ...
      return false;
  }
  ```

**结论**：只要 `--split-mode tensor`，`set_sampler` 就无条件返回 false。
- 这不是配置错误，是**上游未实现的功能**（backend sampling 只支持非 tensor 切分）。
- 外层那句 `backend offload failed ... using CPU sampler` 是**误导性信息**，真正的理由是
  "backend sampling not supported with SPLIT_MODE_TENSOR"（那条 warn 只打一次，且被 SPC_WRN 抢了视线）。

## 3. 为什么这不是"改个参数"就能解决

生产配置用 tensor 是因为它明显更快（历史实测：tensor 比 layer/pipeline 的 decode 快约 19%，prefill 约 2x）。
而 GPU 采样只在**非 tensor** 切分下才开启。所以这是一个**真实的取舍**：

| 方案 | GPU 采样 | 代价 |
|---|---|---|
| `--split-mode tensor` + MTP | 否（CPU 采样） | 每个 draft step 要 logits 同步回 host |
| `--split-mode layer` + MTP | 是 | 失去 tensor 的并行加速（prefill/decode 都慢） |
| 改 llama.cpp：为 TENSOR 实现 backend sampling | 是 | 上游级功能改动（新子系统级别，按 AGENTS.md 要先与用户确认） |

**先量取舍，再决定要不要动代码。** 见下面第 4 节。

## 4. 实测结果（2026-09-20，GPU 3,4 同 NUMA，Q2_K_XL，ctx 8192，同一 ~1700-token prompt，gen 128）

| 变体 | split-mode | spec | `offload failed` | **eval t/s** | 接受率 / mean len |
|---|---|---|---|---|---|
| A | tensor | none | 0 | **41.33** | — |
| B | tensor | draft-mtp n=4 | **1（CPU 采样）** | **85.81** | 0.962 / 4.85 |
| C | layer | none | 0 | **33.44** | — |
| D | layer | draft-mtp n=4 | **0（GPU 采样真的生效）** | **77.06** | 0.971 / 4.88 |

### 结论（含一处重要纠错）
1. **MTP 是巨大收益，不是负担**：B（tensor+MTP）**85.81** vs A（tensor 无 MTP）**41.33** = **+107%（约 2.08x）**。
   -> **此前文档与记忆里"MTP 让 tg 更慢（37 vs 43）"的说法是错的**：那是**跨上下文比较**
   （256K 带 MTP vs 短上下文无 MTP）叠加接受率混淆造成的假象。**同上下文、同 prompt 下，MTP 直接翻倍。**
2. **`--split-mode tensor` 是对的**：无 MTP 时 tensor **41.33** > layer **33.44**（+24%）。
3. **GPU 采样确实能打开**（D 的 `offload failed` = 0，走 layer 模式即可），
   但 **D 77.06 < B 85.81** —— 即 **"CPU 采样"的损失小于 "layer 相对 tensor" 的损失**。
   -> **当前最优就是 `--split-mode tensor` + CPU 采样**。为 tensor 模式实现 backend sampling 是**大特性**
      （`src/llama-context.cpp:1233` 直接 `return false`），收益上限约 85.81 -> ~92（~7%），
      **性价比不足 → 这条线降级为"已知但不动"**（不再列为首选瓶颈）。
4. **⚠️ 本实验的接受率被 prompt 内容抬高**：prompt 是重复的 `The quick brown fox ...` 文本，
   投机解码对重复文本异常好猜 -> **0.96 是上限**；真实文本更低（此前见过 0.55~0.82）。
   所以 **+107% 是上限而非典型值**；但即便按 0.55 的接受率，MTP 仍是明确的大收益。
5. **对 L3 的含义**：L3 **必须带 MTP**（与生产一致）跑，并且**必须同时报告接受率** ——
   tg 由接受率主导，只报 tg 无法解释。评估任何 kernel 改动对 tg 的影响时，
   要么关 MTP，要么固定 seed 让两侧生成同样内容。

## 5. 相关坑（避免重复踩）
- 别拿"短上下文无 MTP 的 43.20 t/s"去比"256K 带 MTP 的 37.34 t/s" —— 上下文/MTP/生成内容全都不同，**不可比**。
- MTP 开启后 `eval time = X ms / N tokens` 的 N 是**含投机接受**的最终 token 数 → tg 会随**接受率**变化，和 kernel 速度混在一起。
