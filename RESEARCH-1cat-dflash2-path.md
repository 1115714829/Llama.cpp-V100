# RESEARCH: 1cat-vLLM v1.5.0 DFlash2 推理路径精确定位 + llama.cpp b11053 对照表

- 生成日期: 2026-09-20
- 参考树(只读): `F:\vllm+llama.cpp\1cat-vllm` = `v1.5.0-672-gb711d53045`, HEAD `b711d5304525dfc0cca6bc8a0bb005f33fe1bbf8` (2026-09-17)
- 对照树: `F:\vllm+llama.cpp\llama.cpp` HEAD `79504e72baf77396a2aa621dca64fd4db8699fc0` (base b11053)
- 本文件所有 `file:line` 均在上述两个 HEAD 上人工核对过。
- 未在本机(AC922)运行任何东西; 下文凡标注「文档实测」的数字都是 **1cat 自己文档里的 Nsight/trace 数字**, 不是本地复现。

## 证据分级(全文通用)

| 标记 | 含义 |
|---|---|
| `[代码]` | 直接读源码得到的行号与逻辑, 可信 |
| `[文档实测]` | 1cat 设计文档里记录的 Nsight/trace 毫秒或 tok/s, 未经本地复现 |
| `[文档宣称]` | 文档的结论性描述, 无原始数字支撑 |
| `[未验证]` | 我无法从代码或文档确认的事情 |

---

# 1. DFlash2 完整算子链 (每步 file:line)

## 1.0 总览: 一轮 = 5 段

1cat 自己的阶段划分 (`docs/design/sm70_dflash2_target_graph_20ms.md:21-27` 表头):

```
Draft graph -> Draft to target -> Target graph -> Target to draft -> (下一轮)
```

对应代码:

| 段 | 入口 |
|---|---|
| Draft graph | `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:523` `propose()` -> `:771` `query_cudagraph_manager.run_fullgraph()` |
| Draft -> Target | `vllm/v1/worker/gpu/model_runner.py:1485` `dispatch_cg_and_sync_dp` + `:1503` `prepare_inputs` |
| Target graph | `vllm/v1/worker/gpu/model_runner.py:1651` `select_attention_graph` -> `:1655` `run_fullgraph` |
| Target -> Draft | `vllm/v1/worker/gpu/spec_decode/dflash2/sparse_rejection.py:224` `try_dflash2_sparse_target_rejection()` |

## 1.1 每轮入口 propose()

`vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:352` `class DFlash2Speculator(DFlashSpeculator)`
继承基类 `DFlashSpeculator` (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:46`), 只覆盖 draft 生成与几处 SM70 专用图。

`propose()` 主体在基类 `dflash/speculator.py:523-797`, 顺序:

| # | 动作 | 位置 |
|---|---|---|
| 1 | `_prepare_proposal_runtime()` 刷新 lookup 资格与控制器 | `dflash/speculator.py:561`; dflash2 覆盖 `dflash2/speculator.py:639` |
| 2 | 取 target hidden: `_get_prepared_context_hidden` 或 `torch.cat(aux_hidden_states,dim=-1)` + `combine_hidden_states` | `dflash/speculator.py:568-602` |
| 3 | 把 hidden 拷进持久缓冲 `self.hidden_states[:num_target_tokens]` | `dflash/speculator.py:604` |
| 4 | `prepare_dflash_inputs()` (Triton 单核) | 调用 `dflash/speculator.py:652`; 核 `dflash/speculator.py:817` |
| 5 | `_precompute_context_kv()` 把 target hidden 投影后写 draft KV | `dflash/speculator.py:698` -> `vllm/model_executor/models/qwen3_dflash.py:707,725-726` |
| 6 | ngram/lookup 命中则早退 | `dflash/speculator.py:720-728` |
| 7 | `dispatch_cg_and_sync_dp()` 选图 | `dflash/speculator.py:734-742` |
| 8 | 图内元数据刷新 or 重建 | `dflash/speculator.py:749-766`; dflash2 覆盖 `dflash2/speculator.py:841` |
| 9 | FULL 图 replay 或 eager | `dflash/speculator.py:769-780` |
| 10 | 返回 `self.draft_tokens[:num_reqs]` | `dflash/speculator.py:797` |

## 1.2 draft 的并行块前向

draft 是 **block-8 一次性并行前向** (1 个 anchor + 7 个 mask, 7 个 draft token), 不是自回归。

- dflash2 的图内实现: `dflash2/speculator.py:1173` `_generate_draft()`
  - `dflash2/speculator.py:1182` `_run_model()` (基类 `dflash/speculator.py:398-424`)
    - `:416` `self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(...)`
    - `:419` `self.model(input_ids=None, positions=..., inputs_embeds=...)`
  - `dflash2/speculator.py:1190-1192` `hidden_states = last_hidden_states[sample_indices].view(num_reqs, draft_block, -1)`
- draft 模型: `vllm/model_executor/models/qwen3_dflash2.py:443` `DFlash2Qwen3ForCausalLM`
  - 主类 `DFlash2Qwen3Model` `qwen3_dflash2.py:333`; 层 `DFlash2Qwen3DecoderLayer` `qwen3_dflash2.py:180`
  - 层 forward `qwen3_dflash2.py:238-258`: attention_conv.prepare -> self_attn -> attention_conv.finish -> post_norm -> mlp_conv.prepare -> mlp -> mlp_conv.finish
  - grouped conv (taps/group_size/block_size 来自 checkpoint): `DFlashGroupedConv` `qwen3_dflash2.py:122`; 纯函数 `_grouped_conv` `:99-119`; 卷积块边界取 **checkpoint 训练时的 block_size** `qwen3_dflash2.py:226-228`
- 非因果注意力: draft 元数据 `causal=self._group_causal` (`dflash/speculator.py:761`), dflash2 里 `self._group_causal = not self.requires_non_causal` (`dflash/speculator.py:372`)
- 训练宽度常量: block_size=8 -> `num_query_per_req = 8` (`dflash2/speculator.py:760` 断言 `num_query_per_req == 8`), 7 个 draft token

## 1.3 hidden-state 捕获: 5 个层边界

**结论: DFlash2 的 target hidden 不是取最后一层, 而是取 5 个「层边界」的输入 hidden 拼接后投影。**

| 环节 | 位置 | 内容 |
|---|---|---|
| 边界 id 来源 | `vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:35-53` | `layer_ids = [i+1 for i in dflash_config["target_layer_ids"]]` (`:47`) |
| 注入 target 模型 | `vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:26-32` `set_eagle3_aux_hidden_state_layers` | 调 `set_aux_hidden_state_layers(aux_layers)` |
| runner 侧调用 | `vllm/v1/worker/gpu/model_runner.py:481` | `set_eagle3_aux_hidden_state_layers(self.model, self.speculative_config)` |
| target 侧落点 | `vllm/model_executor/models/qwen3_5.py:678` (`aux_hidden_state_layers`), `:940-941` (`set_aux_hidden_state_layers`), `:943` (`get_eagle3_aux_hidden_state_layers`) | Qwen3.5 主干在层边界 dump hidden |
| 拼接维度 | `vllm/model_executor/models/qwen3_dflash.py:85-93` `_get_dflash_fc_input_size` | `target_hidden_size * len(aux_layers)` |
| 投影层 | `qwen3_dflash.py:466-472` `self.fc = self._make_context_projection(input_size=..., output_size=hidden_size)` | 25600 -> 5120 |
| DFlash2 专用 TP4 分片版 | `qwen3_dflash2.py:344-385` | 仅当输入输出是 `(25600, 5120)` 或 `(20480, 4096)` 且 TP=4 |
| 拼接+投影 | `qwen3_dflash.py:987-1013` `combine_hidden_states` | `result = self.model.fc(hidden_states)` (`:1010`) |
| 实际调用 | `dflash/speculator.py:597-600` | `torch.cat(aux_hidden_states, dim=-1)` -> `combine_hidden_states` |

**「5」这个数字的证据链**:
- `vllm/model_executor/models/qwen3_dflash2.py:350-354` 把 `(25600, 5120)` 与 `(20480, 4096)` 写成显式集合; 25600 / 5120 = **5**; 20480 / 4096 = **5**。`[代码]`
- `tests/v1/spec_decode/test_dflash_mrv2_config.py:272-278` 的 fixture: `target_layer_ids=[5, 19, 33, 47, 61]` -> aux layers `(6, 20, 34, 48, 62)`。5 个 id, 间隔 14。`[代码]`
- `docs/design/sm70_quasar_dflash2_operator_audit.md:154` 「all five layer weights」, `sm70_quasar_dflash2_resource_audit_20260909.md:658` 「four ranks, five layers, four projections and five input snapshots」。`[文档实测]`
- Qwen3.8-27B: block_count=65 (64 主体 + 1 个 MTP 块), 边界 5/19/33/47/61 落在 64 层主体内。`[代码+已知事实]`

⚠️ 注意 `target_layer_ids` 是 **DFlash checkpoint 的层索引**, 捕获用的是 **层边界 = 索引+1** (`eagle3_utils.py:45-47` 注释明说)。`docs/design/sm70_v100_migration_control.md:29722` 记录过「`+1` 选错了 target aux」的历史坑。`[文档宣称]`

## 1.4 selector (selector top-K 16)

### 1.4.1 配置与常量

| 项 | 位置 |
|---|---|
| `self.selector_top_k = int(draft_config["selector_top_k"])` | `dflash2/speculator.py:364` |
| 候选 id/score 缓冲 `[max_num_reqs, draft_block, top_k]` | `dflash2/speculator.py:386-405` |
| `CandidateSelector` 模块(top_k / rank / predecessor+successor codebook / hidden_projection) | `qwen3_dflash2.py:286-330` |
| 构造时 `top_k=int(draft_config["selector_top_k"])`, `rank=int(draft_config["selector_rank"])` | `qwen3_dflash2.py:424-432` |
| 单独 torch.compile tag `dflash2_candidate_selector` | `qwen3_dflash2.py:424` |
| 测试断言 `speculator.selector_top_k == 16` | `tests/v1/spec_decode/test_dflash2.py:1120` |
| 生产准入要求 `selector_top_k == 16` | `vllm/config/vllm.py:175` |
| `dflash2` 判定 = `selector_top_k > 0` | `vllm/config/speculative.py:88-90`; `vllm/v1/worker/gpu/spec_decode/__init__.py:28` |

### 1.4.2 候选生成 (target LM head 上做 top-k)

`qwen3_dflash2.py:456-509` `compute_candidates()`:

| 步 | 位置 | 说明 |
|---|---|---|
| a | `:459-471` | 要求 LM head 未量化; 否则需 `VLLM_SM70_DFLASH2_QUANT_LM_HEAD=1` |
| b | `:480-484` | `self.lm_head.maybe_get_sm70_dflash2_top20(hidden_states, selector.top_k)` (SM70 快路) |
| c | `:485-493` | 回退: 稠密 logits -> 屏蔽 vocab padding -> `_topk` -> 加 `org_vocab_start_index` |
| d | `:497-499` | TP>1: `tensor_model_parallel_all_gather(values/ids)` |
| e | `:501-503` | 全局再 top-k 收敛到 `selector_top_k` |
| f | `:505-508` | `values.float() * output_multiplier`; 可选 `final_logit_softcapping` (tanh) |

- LM head 入口: `vllm/model_executor/layers/vocab_parallel_embedding.py:1200` `maybe_get_sm70_dflash2_top20()` -> `:698-705` -> QPN8 rerank `:560-695`
- 目标侧 sparse rejection 也复用同一入口: `vllm/model_executor/layers/logits_processor.py:536`

### 1.4.3 格点打分 (bilinear 转移分)

`qwen3_dflash2.py:261-282` `_score_edges()`:

```
successors   = successor_table[candidate_ids]                      # :271
predecessor_ids = [anchor, candidate_ids[:, :-1]]                   # :272-278
predecessors = predecessor_table[predecessor_ids]                   # :279
return unary_logits[:, :, None] + einsum("blpr,blcr->blpc",
                                          predecessors * hidden[:, :, None],
                                          successors)               # :280-282
```

即 score(p,c) = unary(c) + dot(prev_codebook[p] * hidden_proj(h), next_codebook[c])。
`hidden = self.hidden_projection(hidden_states)` (`qwen3_dflash2.py:321`)。
调用点: `dflash2/speculator.py:1201-1206` `self.model.model.candidate_selector(candidate_ids, unary_logits, hidden_states, anchor_token_ids)`。

### 1.4.4 路径行走 (selector walk)

`dflash2/speculator.py:909` `_sample_path()` -> `dflash2/speculator.py:922` 启动 `_selector_walk_kernel` (`dflash2/speculator.py:53-127`):

| 核内步 | 行 | 说明 |
|---|---|---|
| 取 req_state / temperature / seed | `:75-78` | |
| `scores = load(scores_ptr + (flat*top_k + previous)*top_k + offsets)` | `:80-87` | 只读 **上一个选中候选那一行** 的 top_k 个转移分 |
| 概率模式温度标定 | `:88-94` | `scores / (temperature * PROPOSAL_TEMPERATURE_SCALE)`; 可选 nucleus top-p |
| `gumbel_noised_argmax(...)` 抽下一个候选 | `:105-115` | `IS_DRAFTING=True`, `APPLY_TEMPERATURE=False` |
| 写回 realized score | `:117-121` | 供 rejection sampler 用 |
| `token = candidate[index]`; `previous = index` | `:122-124` | 链式 |
| 前 6 步 fuse, 第 7 步单独 | `:126-127` + `_selector_walk_tail_kernel` `dflash2/speculator.py:131-201` | 见下 |

**SM70 专用尾核**: `_requires_sm70_tail()` `dflash2/speculator.py:27-33` (只在 SM70 且 num_steps>1 为真);
原因注释在 `dflash2/speculator.py:149-154`: 「Triton can drop the seventh store of a fully unrolled selector walk during CUDA Graph replay on Volta」。
`walk_steps = draft_block - 1` 当尾核启用 (`dflash2/speculator.py:921`), 尾核从 `:922` 启动。

**命中缓存**: `_cache_draft_logits()` (`dflash2/speculator.py:963-983`) -> `_cache_draft_logits_kernel` (`:205-239`)。
把旧候选位置置 `-inf`, 写新候选的 score, 供精确概率拒绝采样复用; 由 `get_sparse_draft_logits()` (`:985-991`) 取出。

## 1.5 target 验证: FP8 KV 上的 grouped q8 验证

### 1.5.1 门控

`vllm/v1/attention/backends/flash_attn_v100.py:5630` `_dflash2_grouped_verify_allowed()`, 全部条件 (`:5657-5711`):

| 条件 | 行 |
|---|---|
| `num_reqs == 1 and num_query_tokens in (8, 16)` (q8 / q16) | `:5657-5661` |
| 或 batched: `num_reqs in (2,4,8) and max_query_len==8 and num_query_tokens==num_reqs*8` | `:5662-5668` |
| `attention_metadata.is_dflash_selector_target` 为真 | `:5673` |
| `max_model_len >= dflash2_grouped_verify_min_model_len` (默认 32768) | `:5674-5675` |
| causal, 且 window=(-1,-1) | `:5676-5677` |
| `tuple(query.shape) == (num_query_tokens, 6, 256)` | `:5678` |
| query FP16 + contiguous | `:5679-5680` |
| key/value cache ndim==4, shape[1] in (1648, 1728, 3296, 3456), shape[2:]==(1,256) | `:5681-5690` |
| KV dtype `torch.uint8` (打包 FP8) 且 stride(-1)==1 | `:5691-5694` |
| **`self.kv_cache_dtype == "fp8_e5m2"`** | `:5698` |
| block_table / seq_lens 形状与 dtype 契约 | `:5699-5710` |

关键几何: `(num_query_tokens, 6, 256)` 里的 **6 = GQA group size = 24 Q 头 / 4 KV 头**。也就是 **一个 KV 页被 8 个 verifier query 行共享, 6 个 Q 头一次算完**。`[代码]`

op 解析: `flash_attn_v100.py:1281-1317` `_get_flash_grouped_verify_op()` -> `from flash_attn_v100 import flash_attn_grouped_verify_paged`
python 绑定: `flash-attention-v100/flash_attn_v100/flash_attn_interface.py:1164`; 导出 `flash-attention-v100/flash_attn_v100/__init__.py:16,41`
开关: `flash_attn_v100.py:4852-4856` `use_dflash2_grouped_verify`, `:4861-4867` 最小 model_len
调用: `flash_attn_v100.py:5749` `_call_dflash2_grouped_verify()`

### 1.5.2 内核

- `csrc/attention/sm70_grouped_long/kernel/grouped-attention.cu` (5011 行, 227881 B) -- grouped 主核
- `csrc/attention/sm70_grouped_long/kernel/scalar-attention.cu` (56952 B) -- 长上下文 scalar 尾核
- `csrc/attention/sm70_grouped_long/kernel/fp8_kv_utils.cuh:10-11` -- `KV_CACHE_DTYPE_FP8_E4M3 = 1`, `KV_CACHE_DTYPE_FP8_E5M2 = 2`
- E5M2 是 grouped verify 实际走的分支: `flash-attention-v100/kernel/flash_decode_paged.cu:4661-4704` `DISPATCH_GROUPED_VERIFY_PARTIAL` 宏里全部传 `KV_CACHE_DTYPE_FP8_E5M2`

**KV dtype 事实对账**: 出货脚本是 `--kv-cache-dtype fp8_e4m3` (`scripts/serve_qwen38_27b_nvfp4_v100.sh:82`), 但 **grouped verify 这个快路自己要求 E5M2** (`flash_attn_v100.py:5698`), 且宏里硬编码 E5M2。两者不是同一个张量: 生产文档里的 unit 用 E5M2 (`sm70_dflash2_target_graph_20ms.md:13` 「target E5M2 KV, draft FP16/auto KV」), 而出货脚本给的是 E4M3。**这是个未在代码里对齐的不一致点**, 见 §6 未验证。`[代码]`

### 1.5.3 分批验证宽度 q8 / q16

- q8 = 1 + 7 (本 checkpoint 的固定宽度)
- q16 = 由 **lookup 自适应控制器** 决定: `dflash2/speculator.py:706-749` `next_num_draft_tokens()`
  - 控制器状态机 `_advance_lookup_controller()` `dflash2/speculator.py:317-349`
  - 异步标志拷贝 `_queue_lookup_flags()` `:680-704`, 消费 `_consume_lookup_flags()` `:668-678`
  - 目的: 在「强 copy 上下文」时把验证宽度从 8 提到 16, 且**不引入 D2H 同步**
- 目标侧图宽度集合: `vllm/v1/worker/gpu/cudagraph_utils.py:156-167` (`decode_query_lens = (8, 16)`)
- 尾部宽度 q1..q7: `cudagraph_utils.py:168-188`; 选择守卫 `model_runner.py:1462-1476`

## 1.6 精确概率拒绝采样

入口 `vllm/v1/worker/gpu/spec_decode/dflash2/sparse_rejection.py:224` `try_dflash2_sparse_target_rejection()`;

| 步 | 位置 | 说明 |
|---|---|---|
| 开关 | `sparse_rejection.py:233` | `envs.VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION` |
| 采样契约检查 | `sparse_rejection.py:184-221` `_supports_sparse_sampling_contract` | 必须 standard 方法; `top_k == 20` (`:204`); 无 penalty/bias/bad-words/logprobs; min_p==0 |
| 取 draft 支持 | `sparse_rejection.py:248-251` -> `dflash2/speculator.py:985-991` | 复用 selector walk 缓存的候选 |
| 取 target 支持 | `sparse_rejection.py:252-255` | `model.get_topk_tokens_and_logits(sample_hidden_states, 21)` |
| 边界歧义保护 | `sparse_rejection.py:34-60` `_compact_target_requires_reference`, 调用 `:258-266` | 第 21 列探针检测 top-20 边界 tie/top-p 边界; 有歧义则退回全词表参考采样 |
| 采样 | `sparse_rejection.py:278-292` -> `rejection_sampler_utils.py:743` | |
| 核 | `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:543-740` `_dflash2_sparse_topk_rejection_kernel` | |
| 紧凑行归一 | `rejection_sampler_utils.py:499-539` `_dflash2_compact_target_row` | top-k 内做温度与 top-p, 算 `logsumexp` |

**接受判据** (`rejection_sampler_utils.py:656-658`), 原文:

```
accepted &= (target_proposed_logit - target_lse) > (tl.log(uniform) + draft_proposed_logit - draft_lse)
```

即 `log p_target(x) > log u + log q_draft(x)`, 等价于 `u < p_target(x) / q_draft(x)`。**这是精确概率比拒绝采样**, 不是「target argmax 等于 draft」的贪心判据。`[代码]`

**残差重采样** (`:703-719`):

```
ratio = exp(draft_log_probs - target_log_probs)          # :714
residual_logits = where(keep & (ratio < 1.0),
                        target_log_probs + log1p(-ratio),
                        -inf)                            # :715-719
```

即标准的 `relu(p - q)` 归一化残差; 最后一抽用 `gumbel_noised_argmax` (`:722-732`)。
核 docstring `:577-583` 明确论证: 目标分布在 top-k/top-p 之外为 0, DFlash2 proposal 在 selector top-k 之外为 0, **因此接受比与残差分布在这两个紧凑集合上是精确的, 不需要扫全词表**。`[代码]`

---

# 2. 逐个定位开关 / 内核

## 2.1 VLLM_SM70_DFLASH2_QPN8_RERANK 与 _SHADOW

### 2.1.1 声明与默认

| 项 | 位置 | 默认 |
|---|---|---|
| `VLLM_SM70_DFLASH2_QPN8_RERANK` | `vllm/envs.py:239` (类型), `envs.py:2205-2207` (lambda) | `False` |
| `VLLM_SM70_DFLASH2_QPN8_RERANK_SHADOW` | `vllm/envs.py:241`, `envs.py:2216-2218` | `False` |
| 生产 profile 默认打开 RERANK | `vllm/config/vllm.py:94` `_SM70_DFLASH2_VERIFIER_DEFAULTS` | `"1"` |
| 强制 dense 顺序 | `vllm/config/vllm.py:95-96` `QPN8_DENSE_ORDER="1"`, `ALLOW_CANDIDATE_ORDER="0"` | |

### 2.1.2 C++ 侧: 两条 exact-shape 分支

**(a) `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu:1322-1340`** `select_dense_dispatch_policy()`

```
1324  const char* dflash2_rerank = std::getenv("VLLM_SM70_DFLASH2_QPN8_RERANK");
1325  const char* dflash2_shadow = std::getenv("VLLM_SM70_DFLASH2_QPN8_RERANK_SHADOW");
1327  const bool exact_dflash2_rerank = (dflash2_rerank && atoi(...) != 0)
1328                                   || (dflash2_shadow && atoi(...) != 0);
1330  if (exact_dflash2_rerank && m >= 1 && m <= 8 && n == 62080 && k == 5120
1331      && group_size == 0) {
1335    return turbomind::gemm::DispatchPolicy::kDefault;
1336  }
```

作用: **排除自动 tuning**。注释 `:1332-1334` 说 sparse reranker 复现了这个精确 split-K 契约, 不能让并发启动噪声在某个 TP rank 上选出数值不同的 LM-head spec。

**(b) `csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/gemm.cu:231-242`** `GetSm70AwqTp2FastTarget()`

```
231  const char* selector_rerank = std::getenv("VLLM_SM70_DFLASH2_QPN8_RERANK");
232  const char* selector_shadow = std::getenv("VLLM_SM70_DFLASH2_QPN8_RERANK_SHADOW");
234  const bool exact_selector_rerank = ... ;
237  if (exact_selector_rerank && desc.arch == 700 && desc.type_a == kHalf
238      && desc.type_b == kHalf && desc.type_c == kHalf && desc.m >= 1
239      && desc.m <= 8 && desc.n == 62080 && desc.k == 5120 && desc.num == 1) {
240    return Sm70AwqTp2FastTarget{desc.n, desc.k, 8,    256,         64,
241                                10,     1,      true, "s884_1x4x1"};
242  }
```

字段顺序按 `Sm70AwqTp2FastTarget` 的聚合初始化: `n, k, cta_m=8, cta_n=256, cta_k=64, split=10, order=1, active=true, layout="s884_1x4x1"`。
作用: **把 M in [1,8] 的 62080x5120 FP16 LM head 钉死在固定 tile**, 与 rerank 预期完全一致。

### 2.1.3 python 侧

| 项 | 位置 |
|---|---|
| 判定函数 `_sm70_dflash2_qpn8_rerank_enabled()` | `vllm/model_executor/layers/vocab_parallel_embedding.py:58-59` |
| `_sm70_dflash2_qpn8_rerank_requested()` (含 shadow) | `:62-65` |
| packed layout 申请 | `:68-77` |
| dense 顺序保护 | `:80-95` |
| 快路资格 | `:103-124` |
| 缓冲分配 (`_sm70_dflash2_rerank_*`) | `:215-297` |
| 主实现 `_maybe_sm70_dflash2_qpn8_rerank` | `:560-695` |
| 粗支持 QPN8 GEMM | `:583-592` `sm70_ops.fp8_qpn8_gemm_sm70_out(..., SPLIT_K=8, CHAINS=1, False, False)` |
| 粗支持 top-64 (不排序) | `:597-603` |
| 精确 FP16 重排 | `:617-628` `sm70_ops.sm70_f16_indexed_rerank_packed_out(..., CTA_N=128, SPLIT_K=10)` |
| 稠密顺序 top-k | `:634-643` `_sm70_dflash2_dense_order_topk` |
| **SHADOW 分支 (eager-only)** | `:654-687`; 拒绝在图内/编译中运行 `:655-659`; 打 support/set/top-1 一致率 `:672-683`; **永远返回稠密结果** `:685-686` |
| 常量 | `:38-44` |

常量原文:

```
38  _SM70_DFLASH2_QPN8_LM_HEAD_SHAPE   = (62080, 5120)
39  _SM70_DFLASH2_QPN8_MAX_ROWS        = 8
40  _SM70_DFLASH2_QPN8_CANDIDATES      = 64
41  _SM70_DFLASH2_QPN8_SPLIT_K         = 8
42  _SM70_DFLASH2_QPN8_ACCUMULATOR_CHAINS = 1
43  _SM70_DFLASH2_RERANK_CTA_N         = 128
44  _SM70_DFLASH2_RERANK_SPLIT_K       = 10
```

### 2.1.4 **n = 62080 是什么维度?**

**答: TP4 切分后的 target LM head 的本地词表维度 = 全局词表 248320 / 4。**

证据:

| 证据 | 位置 |
|---|---|
| `_SM70_DFLASH2_QPN8_LM_HEAD_SHAPE = (62080, 5120)` | `vocab_parallel_embedding.py:38` |
| 准入条件 `tuple(layer.weight.shape) == (248320 // tp_size, 5120)` | `vocab_parallel_embedding.py:384` |
| 全局词表 248320 出现在 sampler 契约 | `vllm/v1/sample/sampler.py:319,327,333,356,454,473,486`; `vllm/v1/worker/gpu_model_runner.py:7793` |
| Qwen3.5 默认 `vocab_size=248320` | `vllm/transformers_utils/configs/qwen3_5.py:44` |
| 62080 这个字面量在 csrc 里只出现 2 次, 都是 rerank 分支 | `awq_sm70_gemm.cu:1330`, `gemm.cu:239` |
| k = 5120 = hidden size | `vllm/config/vllm.py:169` (`hidden_size == 5120`) |

所以这一对分支只在 **「TP4 + 248320 词表 + 5120 hidden + 未量化 FP16 LM head + M in [1,8]」** 这个精确形状上生效 -- 就是 DFlash2 selector 与 sparse target rejection 共用的那个 LM head。**这也解释了为什么 selector 和 target 采样共用同一份日志/重排逻辑**: 见 `logits_processor.py:536` 与 `qwen3_dflash2.py:481` 调的是同一个 `maybe_get_sm70_dflash2_top20`。

## 2.2 VLLM_SM70_DFLASH2_FUSED_GDN_METADATA / VLLM_SM70_DFLASH2_FUSED_GDN_VERIFY

### 2.2.1 FUSED_GDN_METADATA (envs.py:245)

| 项 | 位置 |
|---|---|
| 声明 | `vllm/envs.py:245` 默认 `False` |
| lambda | `vllm/envs.py:2239-2240` |
| 生产默认打开 | `vllm/config/vllm.py:98` (`_SM70_DFLASH2_VERIFIER_DEFAULTS`), 以及 `:110` (GLM5 TP8 一套) |
| 共享缓冲分配 | `vllm/v1/attention/backends/gdn_attn.py:822-834` |
| 主实现 `prepare_dflash2_gdn_group_metadata()` | `gdn_attn.py:2189`; **门控在 `:2215`** `if not envs.VLLM_SM70_DFLASH2_FUSED_GDN_METADATA: return None` |
| 契约检查 | `:2217-2234` (builders 非空, num_actual_tokens>0, cuda, int32 1D, state_start/req_mapping 成对) |
| model_state 侧总闸 | `vllm/v1/worker/gpu/model_states/mamba_hybrid.py:139-151` |
| 注入到 attn builder kwargs | `mamba_hybrid.py:118-120` `prepared_dflash2_metadata` |

作用 (`gdn_attn.py:2206-2214` docstring): 把 **10 条独立的 gather/copy 管线压成一次 pointer-table 启动**, 一次算完所有 GDN KV group 的图内元数据。它要求 `mamba_cache_mode in ("none", "align")` (`mamba_hybrid.py:148`), 并且是 `_use_dflash2_common_gdn_metadata` (`VERIFY_FASTPATH` + dflash selector engine, `:139-142`) 的下游。

对应实测: `docs/design/sm70_dflash2_target_graph_20ms.md:441-450` -- 「collapses the five full-attention metadata refreshes into one pointer-table kernel」, paired trace 24.424136 ms -> **23.678170 ms**, Draft-to-target 1.857847 -> 1.213916 ms。`[文档实测]`

### 2.2.2 FUSED_GDN_VERIFY (envs.py:248)

| 项 | 位置 |
|---|---|
| 声明 | `vllm/envs.py:248` 默认 `False` |
| lambda | `vllm/envs.py:2257-2258` |
| **唯一生产开启点** | `scripts/serve_qwen38_27b_nvfp4_v100.sh:66` `export VLLM_SM70_DFLASH2_FUSED_GDN_VERIFY=1` |
| 层侧门控 | `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:2456-2460` |
| 被依赖 | `qwen_gdn_linear_attn.py:2461-2465` `TP2_GDN_BV2` 要求它先为真 |
| 准入 `_can_use_dflash2_packed_gdn_verify` | `qwen_gdn_linear_attn.py:5210-5246` |
| 实现 `_forward_dflash2_packed_gdn_verify` | `qwen_gdn_linear_attn.py:5248` |
| 日志 | `qwen_gdn_linear_attn.py:2586-2590` |

准入条件 (`:5221-5246`): 必须 `spec_sequence_masks` 存在, `num_spec_decodes > 0`, **`num_prefills == 0` 且 `num_decodes == 0`** (纯验证步), 无 ddtree, mixed_qkv/a/b 都是 FP16, ssm_state 是 FP16 或 FP32, `mixed_qkv.stride(1) == 1` 且 `stride(0) >= shape(1)`。

**注意**: `FUSED_GDN_VERIFY` **不在** `_SM70_DFLASH2_VERIFIER_DEFAULTS` (`config/vllm.py:87-106`) 里 -- 它只由出货脚本显式打开。而 `FUSED_GDN_METADATA` **在** 默认表里 (`:98`)。这个不对称是真实的。`[代码]`

## 2.3 VLLM_FLASH_V100_DFLASH2_FIXED_INTERLEAVED / _STAGE_PAGE_IDS

| 项 | 位置 | 默认 |
|---|---|---|
| `VLLM_FLASH_V100_DFLASH2_FIXED_INTERLEAVED` | `vllm/envs.py:476`, lambda `envs.py:3139-3141` | `True` |
| `VLLM_FLASH_V100_DFLASH2_STAGE_PAGE_IDS` | `vllm/envs.py:477`, lambda `envs.py:3142-3144` | `True` |
| 谓词定义 (grouped) | `csrc/attention/sm70_grouped_long/kernel/grouped-attention.cu:219-227` | |
| 谓词定义 (scalar) | `csrc/attention/sm70_grouped_long/kernel/scalar-attention.cu:219-227` | |
| 谓词定义 (flash-attention-v100 副本) | `flash-attention-v100/kernel/flash_decode_paged.cu:220-227` | |

谓词原文 (`grouped-attention.cu:219-227`):

```
219  bool dflash2_grouped_fixed_interleaved_enabled() {
220    const char* value = std::getenv("VLLM_FLASH_V100_DFLASH2_FIXED_INTERLEAVED");
221    return value == nullptr || value[0] != '0';
222  }
224  bool dflash2_grouped_stage_page_ids_enabled() {
225    const char* value = std::getenv("VLLM_FLASH_V100_DFLASH2_STAGE_PAGE_IDS");
226    return value == nullptr || value[0] != '0';
227  }
```

**作用点**: 唯一观察到的消费者在 `flash-attention-v100/kernel/flash_decode_paged.cu:4661-4704` 的 `DISPATCH_GROUPED_VERIFY_PARTIAL` 宏:

```
4664  const int64_t page_size = k_cache.size(1);
4665  const int64_t fixed_block_stride = 2 * page_size * kGroupedVerifyHeadDim;
4666  const bool fixed_interleaved_layout =
4667      dflash2_grouped_fixed_interleaved_enabled() &&
4668      (page_size == 1648 || page_size == 3296) &&
4669      k_cache.stride(0) == fixed_block_stride && v_cache.stride(0) == fixed_block_stride &&
4671      k_cache.stride(1) == kGroupedVerifyHeadDim && ... k_cache.stride(2) == kGroupedVerifyHeadDim;
4675  const bool stage_page_ids =
4676      fixed_interleaved_layout && dflash2_grouped_stage_page_ids_enabled();
4677  if (stage_page_ids && page_size == 1648) { LAUNCH(..., 1648, ..., true,  true,  FP8_E5M2); }
4681  else if (stage_page_ids)              { LAUNCH(..., 3296, ..., true,  true,  FP8_E5M2); }
4685  else if (fixed_interleaved_layout && page_size == 1648) { LAUNCH(..., true,  false, ...); }
4689  else if (fixed_interleaved_layout)    { LAUNCH(..., 3296, true,  false, ...); }
4693  else if (page_size == 1648)           { LAUNCH(..., false, false, ...); }
4697  else if (page_size == 3296)           { LAUNCH(..., false, false, ...); }
```

含义:

- `FIXED_INTERLEAVED`: K/V 页按 **固定交错布局** 排布, 判据是 **stride 三元组精确匹配** (`stride(0) == 2*page*256`, `stride(1) == stride(2) == 256`, `stride(-1) == 1`), page_size 只允许 1648 或 3296。这样核可以**丢掉每页的偏移算术**, 按固定步长直接取。
- `STAGE_PAGE_IDS`: 在已固定布局的前提下, **预先把 page id 取到寄存器/共享内存**, 一次 staging 多次复用, 而不是每个 tile 重新解引用 block_table。它需要 fixed 布局成立 (`:4676` 直接把 `fixed_interleaved_layout` 与进去)。
- 两个开关各自可关 (`=0`), 关掉后落到 `(int, bool, bool) = (page, false, false)` 的通用 stride 版。
- 文档侧: `docs/design/sm70_dflash2_long_verify_decay.md:620-621` 明确写「`VLLM_FLASH_V100_DFLASH2_FIXED_INTERLEAVED=0` disables fixed addressing; `VLLM_FLASH_V100_DFLASH2_STAGE_PAGE_IDS=0` disables page-ID staging」。`[文档宣称]`
- 测试: `tests/kernels/attention/test_sm70_flash_v100_grouped_verify.py:447-468`, `tests/test_envs.py:283-284`

⚠️ **观察**: 在 `csrc/attention/sm70_grouped_long/` 这两个 TU 里, 这两个谓词**被定义但没有任何调用点**(我 grep 了整个目录, 只有定义行 + 同名模板参数 `FIXED_INTERLEAVED_HKV1_LAYOUT` 在 `grouped-attention.cu:4366-4367, 4379`, 那是给 e4m3 hkv1 page1568 的另一条路, 由调用者按 **stride 计算** 传入, 见 `flash_decode_paged.cu:5251-5274, 5324`)。**同一份源码的 `flash-attention-v100/` 副本才是实际被编译进 grouped verify 派发的那一份**。这一点在做移植时很关键。`[代码]`

## 2.4 target 与 draft 各自的 CUDA Graph 在哪里建

### 2.4.1 target 图

| 环节 | 位置 |
|---|---|
| manager 构造 | `vllm/v1/worker/gpu/model_runner.py:633` `self.cudagraph_manager = ModelCudaGraphManager(...)` |
| 类定义 | `vllm/v1/worker/gpu/cudagraph_utils.py:422` `class ModelCudaGraphManager(CudaGraphManager)` |
| 入口 | `model_runner.py:897` `capture_model()` -> `:922` `self.cudagraph_manager.capture(...)` |
| 实现 | `cudagraph_utils.py:510` `ModelCudaGraphManager.capture()` |
| **实际 capture** | `cudagraph_utils.py:375` `graph = torch.cuda.CUDAGraph()`; `:379` `with torch.cuda.graph(graph, self.pool): forward_fn(CUDAGraphMode.NONE)` |
| 通用实现 | `cudagraph_utils.py:310` `CudaGraphManager.capture()` (warmup `:343`, capture `:375-386`) |
| replay | `cudagraph_utils.py:406-419` `run_fullgraph()` -> `:419` `self.graphs[desc].replay()` |
| 选择 | `cudagraph_utils.py:391-404` `dispatch()` |
| 长上下文变体图 | `cudagraph_utils.py:438-508` (`_long_attention_graphs`, `select_attention_graph`) |
| 图宽度集合 | `cudagraph_utils.py:154-188`; 尾部 q1..q7 在 `:182-184` |
| 尾部图运行期守卫 | `model_runner.py:1462-1476` |
| 运行期调用 | `model_runner.py:1485` `dispatch_cg_and_sync_dp`; `:1651` `select_attention_graph`; `:1655` `run_fullgraph` |

### 2.4.2 draft 图

| 环节 | 位置 |
|---|---|
| 模式决定 | `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:268-292` `init_cudagraph_manager()`; FULL 不支持则退 `NONE`, **PIECEWISE 明确不支持** (`:281`) |
| manager | `dflash/speculator.py:287-292` `DFlashCudaGraphManager(...)` |
| 类定义 | `vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py:81` |
| 入口 | `vllm/v1/worker/gpu/model_runner.py:935` `self.speculator.capture()`; dflash2 覆盖 `dflash2/speculator.py:751` |
| 基类实现 | `dflash/speculator.py:294-311` -> `dflash/cudagraph.py:85-133` `DFlashCudaGraphManager.capture()` |
| 元数据自建 | `dflash/cudagraph.py:25-78` `_prepare_dflash_inputs_to_capture()` (`build_attn_metadata(..., for_cudagraph_capture=True)` `:62-77`) |
| 真正 capture | 复用 `cudagraph_utils.py:375,379` |
| replay | `dflash/speculator.py:771` `self.query_cudagraph_manager.run_fullgraph(batch_desc)` |

**DFlash2 额外自建 4 张图** (`dflash2/speculator.py:751-839`):

| 图 | 位置 | 条件 | 用途 |
|---|---|---|---|
| context KV 图 | `:770-778` | `CONTEXT_KV_GRAPH or CONTEXT_PIPELINE` 且 SM70 且 `num_query_per_req == 8` (`:753-764`) | 8 行 context 投影+写 KV 一次 replay |
| context compute 图 | `:782-786` | 再加 `CONTEXT_PIPELINE` (`:779`) | 在 target 采样**之前**先把 K/V 算好 |
| context store 图 | `:787-789` | 同上 | 采样后再写 cache |
| draft B1 paged 元数据图 | `:796-839` `_capture_draft_metadata_graph()` | 非因果 + 单卡 + FlashV100 非 triton-prefill + `_is_dflash_draft_model` (`:807-827`) | 只对 `(num_reqs, num_tokens) == (1, 8)` 生效 (`:842`) |

启动条件汇总 (`dflash2/speculator.py:753-764`): `VLLM_SM70_DFLASH2_CONTEXT_KV_GRAPH or VLLM_SM70_DFLASH2_CONTEXT_PIPELINE` (envs.py:261-262), cuda, capability (7,0), `num_query_per_req == 8`, query manager 有图。

配套 replay 路径: `_precompute_context_kv` `:884-902`, `prepare_target_context` `:849-875`, `_get_prepared_context_hidden` `:877-882`。

---

# 3. DFlash2 轮内时间线

以下全部是 **1cat 自己文档里的 Nsight/trace 数字**(`[文档实测]`), 我**没有**在 AC922 复现过。
配置: QUASAR Qwen3.8-27B, TP4, B1, block 8 (7 draft), FULL target + draft CUDA Graph, Flash-V100。

## 3.1 五段分解的演进

来源: `docs/design/sm70_dflash2_target_graph_20ms.md`

| 版本 | 位置 | Draft | Draft->Target | Target | Target->Draft | Complete |
|---|---|---|---|---|---|---|
| 2026-08-23 baseline | `:18-27` | 4.046 | 4.934 | 24.740 | 2.838 | 36.300 |
| sparse rejection 后 | `:342-345` | 4.004 | 1.977 | 19.191 | 1.994 | 27.166 |
| v17 | `:833-839` | 3.756561 | 0.156985 | 15.288068 | 1.543990 | **20.745603** |
| v21 (首次 sub-20) | `:916-922` | 3.332724 | 0.158264 | 15.281917 | 1.163312 | **19.936216** |
| v22 (复现) | `:931-933` | 3.339601 | 0.155429 | 15.330372 | 1.165597 | **19.990999** |
| v25 (indexed-dot) | `:970-973` | 3.324374 | 0.157883 | 15.272063 | 1.139965 | **19.894284** |
| v26 (另一组卡) | `:970-973` | 3.332363 | 0.157819 | 15.313486 | 1.145328 | **19.948997** |
| v52-r2 | `:1139-1142` | 3.374848 | 0.156565 | 15.292271 | 1.073660 | 19.897343 |
| v53 | `:1139-1142` | 3.385334 | 0.157777 | 15.320752 | 1.079131 | 19.942993 |

单请求稳态 decode: v21 180.686 tok/s, v25 182.516, v26 181.216 (`:925, :972-973`)。AL 恒为 3.685714, 逐位置接受计数 `[28,22,17,12,8,5,2]`。

**结论**: draft 前向 **~3.3 ms**, 而 **target 验证 ~15.3 ms 占 77%**; 两个边界合计 ~1.3 ms。要抄的话, **钱在 target**, 不在 draft。

## 3.2 target 图内部分解 (移植排序用)

来源 `sm70_dflash2_target_graph_20ms.md`

**a) 早期 24.740 ms 图, 256 节点级** (`:44-54`), rank 平均 GPU 服务 23.365 ms / 覆盖率 94.7%:

| 类别 | 服务 ms | 启动数/rank |
|---|---:|---:|
| TurboMind FP8 稠密 GEMM | 10.831 | 256 |
| Copy/cast elementwise | 3.387 | 941 |
| Other kernels | 2.869 | 240 |
| TP all-reduce | 2.316 | 128 |
| Torch/Triton elementwise | 2.494 | 739 |
| RMSNorm/residual | 0.616 | 128 |
| LM-head/sample/TP gather | 0.416 | 48 |
| Dense GEMV/GEMM | 0.257 | 65 |
| Fill/mask | 0.179 | 67 |

表格外的关键判词 (`:56-59`): **1680 个 copy/cast/elementwise 节点吃掉 5.881 ms, 平均每节点 8.9 微秒** -- 图「busy but fragmented」。

**b) 1257 节点 / 19.317 ms 版本** (`:305-315`), 18.357 ms 服务:

| 桶 | 节点 | 服务 ms |
|---|---:|---:|
| TurboMind FP8 GEMM | 256 | 10.730 |
| 一阶段 TP4 all-reduce | 128 | 2.313 |
| recurrent GDN verifier | 48 | 1.213 |
| Flash-V100 partition/reduce | 32 | 0.802 |
| fused Gemma residual/RMS | 127 | 0.643 |
| generic elementwise | 144 | 0.541 |

FP8 GEMM 分三档: 64 次 4.394 ms + 128 次 3.924 ms + 64 次 2.411 ms (`:317-318`)。

**c) QPN8 后 1254 节点 / 16.055 ms 版本** (`:452-455`): 4.235 ms gated-pair QPN8 + 3.186 ms QPN8 (16,1) + 2.059 ms QPN8 (16,2) + 1.277 ms TP4 push all-reduce + 1.205 ms recurrent GDN + 0.734 ms Flash-V100 partition attention。

**d) TP4 all-reduce 对标** (`:322-332, :356-361`): vLLM 10 个 512 线程 CTA, 平均 18.07 微秒/次; SGLang-V100 JIT one-shot push 用 80 个 128 线程 CTA, 平均 12.36 微秒; 128 次链 1.854 ms vs 0.846-0.886 ms。自研 NaN-sentinel push 版 128 次链 **0.850-0.856 ms**, 比保留的 1.854 ms pull 链快 ~1.00 ms, 位级等于 rank-order FP32 参考 (`:380-391`)。

## 3.3 边界 (Draft->Target) 的微观账

来源 `sm70_dflash2_target_graph_20ms.md:128-156`, 31 个稳态边界/rank:

- draft 图结束到 target 图开始: 均值 **5.776 ms** (p50 5.582)
- 其中只有 1.304 ms 是真正的 non-graph GPU kernel
- 剩 **~4.47 ms 是 launch / sync / CPU 排队气泡**
- 每个边界提交 **153 个 kernel**: 1 个 `cross_device_reduce_1stage` (0.785 ms) + 20 `DeviceScanKernel` + 20 `DeviceScanInitKernel` + 20 `compute_cuda_kernel` + 21 `indexSelectSmallIndex` (合计 0.276 ms 服务, 但提交串行成本远大于服务) + 其余约 0.243 ms
- 对标 SGLang-V100: 3.398 ms / 30 kernel / 0.115 ms 服务

优化后 (v17) 这一项降到 **0.156985 ms** (`:836`), 即边界问题基本被消除。

## 3.4 grouped verify 的隔离收益

来源 `sm70_dflash2_target_graph_20ms.md:1095-1102`, page size 3296, paired CUDA-Graph 微基准:

| Context | 原生一趟 | 原生两趟 | 旧 XQA |
|---:|---:|---:|---:|
| 1K | 0.068608 ms | 0.097280 ms | 0.230400 ms |
| 32K | 0.356352 ms | 0.544768 ms | 1.024000 ms |
| 128K | 1.214464 ms | 1.864704 ms | 3.021312 ms |
| 256K | 2.396160 ms | 3.690496 ms | 6.011392 ms |

端点 (`:1112-1117`):

| Input | Grouped | Complete round | Target phase | Steady decode |
|---:|:---:|---:|---:|---:|
| 32K | off | 36.310865 ms | 30.248113 ms | 77.537 tok/s |
| 32K | on | **26.361447 ms** | **20.451644 ms** | **88.392 tok/s** |
| 128K | off | 69.854780 ms | 63.708214 ms | 106.428 tok/s |
| 128K | on | **40.778072 ms** | **34.646866 ms** | **182.294 tok/s** |

短 128-token 是中性: 21.117097 vs 21.060181 ms (`:1127-1129`), 所以准入阈值留在 `max_model_len >= 32768` (`vllm/envs.py:475` 默认值一致)。

## 3.5 256K 容量尾

来源 `docs/design/sm70_dflash2_tail_graphs_20260911.md:118-123, :175-180`:

| Input | Seed | Control ms | Tail graphs ms | Saved |
|---:|---:|---:|---:|---:|
| 1024 | 0 | 15.823 | 15.824 | -0.001 |
| 131072 | 0 | 22.573 | 22.577 | -0.004 |
| 261888 | 0 | 35.464 | 31.396 | 4.067 |
| 261888 | 2 | 37.403 | 31.335 | 6.068 |

seed2 保持 4.727273 accepted drafts / 5.818182 emitted tokens per round, 延迟 -16.22% (`:182-187`)。
最终守卫版: 38.338 -> 32.228 ms, 省 6.110 ms (`:209`)。

## 3.6 数据集级 (验收)

来源 `docs/design/sm70_dflash2_acceptance_20260909.md`

| Fixture | BV8/BV2 完整轮中位 | BV2 纯 decode | 接受/发出 每轮 |
|---|---:|---:|---:|
| release1k | 16.454723 / 16.219526 ms | 183.607 tok/s | 1.989011 / 2.989011 |
| MBPP28 | 16.314348 / 16.096967 ms | 302.494 tok/s | 3.876923 / 4.876923 |

(`:110-113`)

| 子集 | 用例 | BV8/BV2 纯 decode | 接受率 | 接受草稿/发出 token 每轮 |
|---|---:|---:|---:|---:|
| GSM8K | 32 | 309.791 / 316.749 | 57.9715% | 4.058008 / 5.058848 |
| MATH500 | 32 | 243.960 / 246.660 | 48.2514% | 3.377595 / 4.377883 |
| HumanEval | 32 | 225.650 / 229.195 | 42.8062% | 2.996437 / 3.996309 |
| MBPP | 32 | 231.882 / 234.678 | 43.3363% | 3.033541 / 4.033895 |
| LiveCodeBench v6 | 16 | 175.055 / 177.259 | 32.9424% | 2.305965 / 3.305909 |

(`:143-150`)

## 3.7 我们自己的账本 (对照)

来自工作区 `AGENTS.md` §1 (本地实测, 已有):

| 成分 | ms/轮 | 占比 |
|---|---:|---:|
| target 整步 (`decode+sync`) | 32.3-34.3 | 55-58% |
| draft 前向 | 13.8-14.4 | 24% |
| CPU selector | 4.3 | 7% |
| 未归因 | ~6-8 | 11% |
| **合计** | **58.9** | AL 5.55 / 94.18 t/s |

**对照结论**: 1cat 的 target ~15.3 ms, 我们 ~32-34 ms; 1cat 的 draft ~3.3 ms, 我们 ~13.8-14.4 ms。
**draft 侧我们有 4 倍差距**(1cat 已经把它压到 3.3 ms 且是单图 replay), target 侧 2 倍。这与 `AGENTS.md` 里「draft 前向一轮几次待定论」的悬案直接相关: 1cat 的结构决定了 **draft 前向一轮只跑一次**(block-8 并行), 不是 8 次。

---

# 4. 与 llama.cpp b11053 的差异清单 (按可移植性排序)

## 4.0 llama.cpp 现状基线 (对齐)

| 环节 | llama.cpp 落点 |
|---|---|
| draft 类型注册 | `common/speculative.cpp:39` `{"draft-dflash", COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH}` |
| DFlash draft 实现 | `common/speculative.cpp:913` `struct common_speculative_impl_draft_dflash` |
| DFlash2 判定 | `common/speculative.cpp:991-994` `selector_top_k = llama_model_dflash_selector_top_k(model_dft); is_dflash2 = selector_top_k > 0` |
| CPU selector (TP 模式) | `common/speculative.cpp:1080-1157` `load_dflash2_selector()`; `:1163-1266` `build_dflash2_selector_cpu()` (8 线程, `:1195`) |
| draft decode + walk | `common/speculative.cpp:1395-1560` `draft()`; CPU walk `:1470-1510`; 图内 lattice walk `:1511-1530` |
| 图内 selector | `src/models/dflash.cpp:467-556` `build_dflash2_selector()`; 调用点 `:824-826` |
| grouped conv | `src/models/dflash.cpp:394-460` `build_dflash2_conv()` |
| draft 前向图 | `src/models/dflash.cpp:562-827` `graph<false>::graph` |
| target 侧 5 边界注入 (本 fork HEAD) | `llama_set_embeddings_layer_inp` (`src/llama-ext.h:111`), `llama_get_embeddings_layer_inp` (`:115`), `cparams.embeddings_layer_inp` (`src/llama-cparams.h:57`), 图内 `src/llama-graph.cpp:1376-1378`, draft 侧读取 `common/speculative.cpp:1360` |
| target CUDA graph | `ggml/src/ggml-cuda/ggml-cuda.cu:4240` `ggml_cuda_graph_evaluate_and_capture`, capture `:4538`, instantiate `:4456`/`:2659`, 复用判定 `:4492-4538` |
| 接受判据 | `common/sampling.cpp:699-727` `common_sampler_sample_and_accept_n` (target 自己抽出的 token **等于** draft 才接受) |
| FA head_dim 256 | `ggml/src/ggml-cuda/fattn.cu:510` `case GGML_TYPE_Q8_0`; 模板实例 `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-{f16,bf16,q8_0}-q8_0.cu` |

**已有、不算差距的**: 5 边界 hidden 提取 (`embeddings_layer_inp` 通路完整) 与 block-8 并行 draft 前向 (`common/speculative.cpp:1417-1424`)。

## 4.1 P1: 高可移植性 (host 侧 / 现有结构内, 无新数值)

| # | 1cat 有 | 1cat 位置 | llama.cpp 现状 | 落点文件 | 备注 |
|---|---|---|---|---|---|
| P1-1 | selector walk 全程在 GPU (Triton) | `dflash2/speculator.py:53-127`, 启动 `:922` | 已落地 CPU 并行版 (PR #27858); 但 **walk 本身仍是单线程标量** `std::max_element` | `common/speculative.cpp:1487-1498` | 我们实测 CPU selector 4.3 ms/轮 = 7% 轮时间; 1cat 是 ~0 (图内)。可先并行化/向量化 walk, 再考虑搬设备 |
| P1-2 | 转移分是 **批量 matmul** | `qwen3_dflash2.py:280-282` `einsum`, 图内 | 图内版已有 `score_run` 批量 matmul `src/models/dflash.cpp:502-532`; **但仅在不 TP 时** | `src/models/dflash.cpp:824-826` | TP 时退回 CPU 走 `speculative.cpp:1470-1510`。让 TP 也能走图内 selector 是纯收益 |
| P1-3 | selector 末位单独一个尾核 (Volta 图 replay 丢第 7 次 store) | `dflash2/speculator.py:131-201`, 启用判据 `:27-33`, `walk_steps` `:921` | 无 | `common/speculative.cpp:1474-1510` / `src/models/dflash.cpp:537-550` | 这是 **Volta + Triton 特有**的坑; llama.cpp 的 selector 走 ggml 图, 未必复现。**仅在实测到丢 store 时才需要** |
| P1-4 | q1..q7 target 尾部图 (q1 非投机步与部分 verifier 也留在图内) | `cudagraph_utils.py:168-188`, 守卫 `model_runner.py:1462-1476` | llama.cpp 的 CUDA graph 按图 key 缓存 (`ggml-cuda.cu:4492-4538`), **没有针对 1..8 token 的显式短宽 capture 策略** | `ggml/src/ggml-cuda/ggml-cuda.cu` (graph key / enable 策略) | 1cat 实测 256K 尾部省 4.07-6.55 ms (`tail_graphs_20260911.md:118-123`)。**我们 256K 场景最大的一块** |
| P1-5 | 提案温度标定 `PROPOSAL_TEMPERATURE_SCALE` / `PROPOSAL_TOP_P` | `vllm/envs.py:273-274`; 用点 `dflash2/speculator.py:88-94` | 无 | `common/speculative.cpp` walk | 纯算术, 但**改的是采样分布**, 属于质量敏感项, 需 A/B |

## 4.2 P2: 中可移植性 (需要新 kernel 或图改动)

| # | 1cat 有 | 1cat 位置 | llama.cpp 现状 | 落点文件 |
|---|---|---|---|---|
| P2-1 | **grouped verifier**: 8 个 verifier query 行共享一个 FP8 K/V 页, 6 个 GQA 头一趟算完 | 核 `csrc/attention/sm70_grouped_long/kernel/grouped-attention.cu`; 门控 `flash_attn_v100.py:5630-5711`; 调用 `:5749` | `fattn.cu` 有 head_dim 256 FA 模板, 但**没有「多 query 行共享一页 + GQA 头合并」的 decode 变体** | `ggml/src/ggml-cuda/fattn-*.cu` + `src/llama-graph.cpp` 里 verify attention 调用 |
| P2-2 | 目标 **FP8 (E4M3/E5M2) KV cache** | `scripts/serve_qwen38_27b_nvfp4_v100.sh:82`; grouped 强制 E5M2 `flash_attn_v100.py:5698`; 宏硬编码 `flash_decode_paged.cu:4680-4700` | ggml-cuda 里 **grep 不到 fp8/e4m3/e5m2**; KV 类型只有普通量化类型 (Q8_0 最接近) | `ggml/src/ggml-common.h`, `ggml/src/ggml-cuda/*`, `src/llama-kv-cache*` |
| P2-3 | **精确概率比拒绝采样** (紧凑支持上的 chain rejection + relu(p-q) 残差) | `rejection_sampler_utils.py:543-740`; 门控 `sparse_rejection.py:184-266` | `common_sampler_sample_and_accept_n` (`common/sampling.cpp:699-727`) 是 **「target 自己抽出的 token == draft 才接受」**, 不是概率比 | `common/sampling.cpp` (+ 需要暴露 target top-k logits) |
| P2-4 | **QPN8 粗支持 top-64 + 精确 FP16 重排** 的 LM head | `vocab_parallel_embedding.py:560-695`; C++ 固定 shape 派发 `awq_sm70_gemm.cu:1322-1340`, `gemm.cu:231-242` | LM head 是普通 `build_lora_mm` (`src/models/dflash.cpp:782`), 无「粗筛 + 精确重算」两段式 | `ggml/src/ggml-cuda/mmq.cuh` / `mmvq.cu` 或新 op |
| P2-5 | **sparse target rejection 的 21 列边界探针** (检测 top-20 / top-p 边界 tie, 有歧义退回全词表) | `sparse_rejection.py:34-60`, 调用 `:258-266` | 无 | `common/sampling.cpp` |
| P2-6 | batched grouped verify (num_reqs in 2/4/8, request-major ABI) | `flash_attn_v100.py:5662-5668`; envs.py:472 | 无 | 同 P2-1 |

## 4.3 P3: 低可移植性 (架构级 / 模型特定)

| # | 1cat 有 | 1cat 位置 | 说明 |
|---|---|---|---|
| P3-1 | **fused GDN metadata** (10 条管线压 1 次 pointer-table) | `gdn_attn.py:2189-2234`, 总闸 `mamba_hybrid.py:139-151` | GDN 是 Qwen3.5 独有; llama.cpp 的 GDN/linear-attn 路径完全不同 |
| P3-2 | **packed GDN target-verification** 路由 | `qwen_gdn_linear_attn.py:2456-2460`, `5210-5246`, `5248` | 同上; 且只由出货脚本开 (`serve_...sh:66`) |
| P3-3 | **context KV CUDA graph + pipeline** (采样前先算好 K/V) | `dflash2/speculator.py:751-794`, `849-902` | 依赖 vLLM 的双流图编排; llama.cpp 无同等抽象 |
| P3-4 | **sharded context FC** (25600->5120 按 TP4 切, gather_output) | `qwen3_dflash2.py:344-385` | llama.cpp 的 `fc` 是单个 `build_lora_mm` (`src/models/dflash.cpp:610`) |
| P3-5 | **自适应 q8/q16 lookup 控制器** | `dflash2/speculator.py:317-349, 639-749` | 无对应; llama.cpp 的 `n_max` 是静态的 |
| P3-6 | **ngram assist + device lookup + lookup-augmented 宽度** | `dflash2/speculator.py:1007-1172`; `dflash2/ngram_assist.py`; `dflash2/lookup.py` | 无对应 |
| P3-7 | **draft-local metadata graph** (B1 (1,8) 专用) | `dflash2/speculator.py:796-847` | 只在 TP1 非因果时生效, 收益窄 |
| P3-8 | **QPN8 rerank SHADOW 审计通道** (eager-only, 永远返回稠密) | `vocab_parallel_embedding.py:654-687` | 这是**审计基建**, 本身不提速; 若做 P2-4 建议照抄这套 shadow 纪律 |

## 4.4 llama.cpp 有 / 1cat 没有 (反向)

| 项 | llama.cpp 位置 | 说明 |
|---|---|---|
| `draft-dspark` 变体 | `common/speculative.cpp:943-944, 1003-1011`; markov head `src/models/dflash.cpp:281-392` | DSpark markov head + 置信度头 + anchor-first block。1cat 侧是另一条线 |
| `d2t` reduced-draft-vocab 散射 | `src/models/dflash.cpp:798-812` | 1cat 无 |
| `p_min` 概率早停 (按 softmax argmax 概率) | `common/speculative.cpp:1499-1508, 1523-1529, 1545-1551` | 1cat 用 lookup 控制器而非 p_min |
| 交叉点 mmvq/mmq 的 Volta 表 | `ggml/src/ggml-cuda/mmvq.cuh` (C4/C5 已落地) | 1cat 是独立的 TurboMind 路线 |
| NCCL 编进 `libggml-cuda.so` | 本 fork 已落地 | 1cat 用 vLLM 自己的 allreduce/quickreduce |

## 4.5 每项在 llama.cpp 的落点汇总

| 优先级 | 落点文件 | 对应 1cat 能力 |
|---|---|---|
| P1 | `common/speculative.cpp` (walk 并行/向量化, 提案标定, 接受契约) | P1-1, P1-2, P1-5, P2-3 |
| P1 | `src/models/dflash.cpp` (TP 也走图内 selector, 末位尾写) | P1-2, P1-3 |
| P1 | `ggml/src/ggml-cuda/ggml-cuda.cu` (短宽 q1..q7 图 capture / key 策略) | P1-4 |
| P2 | `ggml/src/ggml-cuda/fattn-*.cu` + `src/llama-graph.cpp` | P2-1, P2-6 |
| P2 | `ggml/src/ggml-common.h` + `ggml/src/ggml-cuda/*` + `src/llama-kv-cache*` | P2-2 |
| P2 | `common/sampling.cpp` | P2-3, P2-5 |
| P2 | `ggml/src/ggml-cuda/mmq.cuh` / `mmvq.cu` | P2-4 |
| P3 | `src/llama-kv-cache*`, `src/llama-model.cpp` | P3-1, P3-2 (若真要做 GDN 类模型) |
| P3 | `src/models/dflash.cpp` (context FC 分片, draft 元数据图) | P3-3, P3-4, P3-7 |

---

# 5. 一次 DFlash2 轮的「谁在何时干什么」(合并视图)

```
[上一轮结束]
  |
  |-- Target -> Draft 段 (1cat ~1.07-1.54 ms; 优化后 ~1.14 ms)
  |     try_dflash2_sparse_target_rejection()            sparse_rejection.py:224
  |       _supports_sparse_sampling_contract()           sparse_rejection.py:184
  |       model.get_topk_tokens_and_logits(hidden, 21)   sparse_rejection.py:252
  |       _compact_target_requires_reference()           sparse_rejection.py:34
  |       dflash2_sparse_topk_rejection_sample()         rejection_sampler_utils.py:743
  |         _dflash2_sparse_topk_rejection_kernel         rejection_sampler_utils.py:543
  |           accept: log p_t > log u + log q_d          :656-658
  |           residual: p + log1p(-ratio)                :714-719
  |
  |-- Draft 段 (1cat ~3.32-3.76 ms, FULL CUDA Graph 一次 replay)
  |     propose()                                        dflash/speculator.py:523
  |       combine_hidden_states(cat(aux[5]))             dflash/speculator.py:598-600
  |       prepare_dflash_inputs()                        dflash/speculator.py:652
  |       _precompute_context_kv()                       dflash/speculator.py:698
  |       dispatch_cg_and_sync_dp()                      dflash/speculator.py:734
  |       [_refresh_draft_graph_metadata()]              dflash2/speculator.py:841
  |       query_cudagraph_manager.run_fullgraph()        dflash/speculator.py:771
  |         (graph body = _generate_draft)               dflash2/speculator.py:1173
  |           _run_model()  -------------------------->  dflash/speculator.py:398
  |           compute_candidates()  ------------------>  qwen3_dflash2.py:456
  |           candidate_selector()  ------------------>  qwen3_dflash2.py:314
  |           _sample_path()  ------------------------>  dflash2/speculator.py:909
  |             _selector_walk_kernel                    dflash2/speculator.py:53
  |             _selector_walk_tail_kernel (SM70)        dflash2/speculator.py:131
  |           _cache_draft_logits()                      dflash2/speculator.py:963
  |       _apply_lookup()                                dflash2/speculator.py:1107
  |
  |-- Draft -> Target 段 (1cat 优化后 ~0.157 ms; 优化前 4.93-5.78 ms)
  |     model_runner.py:1485 dispatch_cg_and_sync_dp
  |     model_runner.py:1503 prepare_inputs
  |
  |-- Target 段 (1cat ~15.27-15.33 ms, FULL CUDA Graph)
  |     model_runner.py:1651 select_attention_graph
  |     model_runner.py:1655 run_fullgraph
  |       per layer: GDN verifier 或 grouped verify attention
  |         _dflash2_grouped_verify_allowed()           flash_attn_v100.py:5630
  |         _call_dflash2_grouped_verify()              flash_attn_v100.py:5749
  |           flash_attn_grouped_verify_paged()         flash_attn_interface.py:1164
  |             csrc/attention/sm70_grouped_long/kernel/grouped-attention.cu
  |       LM head + selector 支持 (QPN8 rerank)          vocab_parallel_embedding.py:560
  |
  |-- 回到下一轮
```

---

# 6. 未验证 / 存疑

1. **KV dtype 不一致(代码级)**: 出货脚本 `--kv-cache-dtype fp8_e4m3` (`serve_...sh:82`), 但 grouped verify 门控硬要求 `fp8_e5m2` (`flash_attn_v100.py:5698`) 且派发宏硬编码 E5M2。两者要么是不同 unit 的不同配置(文档 `sm70_dflash2_target_graph_20ms.md:13` 写 E5M2), 要么 grouped 快路在 E4M3 下**根本不生效**。**未验证**: 没有跑起来看 route log。
2. **`VLLM_FLASH_V100_DFLASH2_FIXED_INTERLEAVED` / `_STAGE_PAGE_IDS` 在这两个开关打开时的实际收益**: 我没有找到 1cat 文档里 A/B 这两个开关的毫秒数字。`sm70_dflash2_long_verify_decay.md:620-621` 只说怎么关, 不说省多少。**未验证**。
3. **`page_size` 1648 vs 3296 的语义**: 代码只按数值分支, 没有注释说明 1648 = 8 页对齐 206 token 之类。**未验证**具体 KV page 布局常量来源。
4. **`kGroupedVerifyHeadDim` 的值**: 宏里用它算 stride, 我未读到定义处; 由 shape 契约 (256) 推断是 256。**未验证**。
5. **本机(AC922)上没有复现任何 ms 数字**。全部 `[文档实测]` 来自 1cat 自己 4 卡 V100-SXM2-32GB 的报告; 我们的卡是 16GB, 且它们用 TP4 跨岛(物理 GPU 4-7 / 0-3)。
6. **`VLLM_SM70_DFLASH2_FUSED_GDN_VERIFY` 缺少独立的收益数字**: 文档 `sm70_dflash2_target_graph_20ms.md:63-71` 把它列为第一项待 A/B 的实验, 但我没有找到 A/B 结果。**未验证**。
7. **`docs/design/sm70_dflash2_target_graph_20ms.md` 结论时效**: 该文件同时在 `:9` 说目标 sub-20 ms「not an accepted endpoint」(v17 20.7456), 又在 `:986-991` 说「independently replicated below 20 ms」。**两个结论是不同日期的不同版本**, 引用时必须带版本号(v25/v26/v52/v53 才是 sub-20)。
8. **llama.cpp 侧「draft 一轮几次」**: 与 `AGENTS.md` §1 待查项 2 相关。1cat 结构上明确是**一轮一次 block-8 前向**(`dflash2/speculator.py:1173-1224` 一次 `_run_model`)。llama.cpp 的 `draft()` (`common/speculative.cpp:1395-1445`) 也是**一次 `llama_decode` 装所有序列的整块**(`:1440`)。⇒ 我们 13.8-14.4 ms 的 draft 不是「跑了 8 次」, 而是**一次 block-8 前向比对方慢 ~4 倍**。这直接支持 `AGENTS.md` 里「83 GB/s 头寸 8-12x」那一支的怀疑。**(建议主代理把这条当作 Phase B3 的结论输入)**

---

# 7. 引用到的 1cat 文档清单

| 文件 | 用途 |
|---|---|
| `docs/design/sm70_dflash2_target_graph_20ms.md` | 五段分解 / target 图节点账 / 边界账 / QPN8 / grouped verify / sub-20 ms |
| `docs/design/sm70_dflash2_tail_graphs_20260911.md` | q1..q7 尾部图 / 256K 实测 |
| `docs/design/sm70_dflash2_acceptance_20260909.md` | 数据集级 16.2/15.8 ms 与接受率 |
| `docs/design/sm70_dflash2_nvfp4_17ms.md` | 17 ms 战役规划与验收门(状态: planning only) |
| `docs/design/sm70_quasar_dflash2_15ms.md` | QUASAR 15 ms 战役 / whole-round resource audit |
| `docs/design/sm70_quasar_dflash2_operator_audit.md` | 「all five layer weights」等算子级审计 |
| `docs/design/sm70_dflash2_long_verify_decay.md` | 两个 grouped 开关的说明 |
| `docs/design/sm70_v100_migration_control.md` | 历史坑索引(2.7 MB, 只按需 grep) |
