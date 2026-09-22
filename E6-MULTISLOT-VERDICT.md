# E6 判决：候选 A 胜出（R185）—— **多槽重建缓存**，判据已实测

## 实测判据（臂 `e6fp`，端口 8216，源码 `ggml-backend-meta.cpp` 加 `GGML_META_FP_STATS`；`BUILD_RC=0 ERRORS=0 BUILTLINES=115 MARK_FPSTAT=1`；sha256 门 `f3edac19...` 一致）
```
[FPSTAT] calls=640 same_as_prev=0 distinct=20 max_count=207
[FPSTAT] calls=704 same_as_prev=0 distinct=24 max_count=226
[FPSTAT] calls=768 same_as_prev=0 distinct=24 max_count=247
[FPSTAT] calls=832 same_as_prev=0 distinct=24 max_count=266
```
- **`same_as_prev=0`**：832 次 meta 调用里，**没有一次**与前一次的内容指纹相同 ⇒
  N6a（`GGML_META_REBUILD_CACHE`，"只认紧邻上一次"）**在结构上不可能命中** ——
  这是 N6a 实测"机制生效但不省时间"的**追溯解释**（此前只是推测）。
- **`distinct=24`，最高频占 266/832 = 32%** ⇒ **槽位 >=8 即可接近 100% 命中**。
  ⇒ **候选 A（多槽）胜出；候选 B（改 CUDA 图 key）不必付重编 CUDA 模板的代价。**
- 同臂 `MEDIAN_TG=95.79`（对照 e5graph 97.33 / e4slot 97.02）＝ **全图指纹哈希的探针税约 1.6%**，在离散内。
- `[META]` 仍稳定在 `total=10.865 dev=7.183 ar=2.212 ms/call, sub/call=47.6, ar/call=46.6`（与 E5 稳态一致 ⇒ 口径更正被二次确认）。

## 7. 第四、五次尝试与 arena 耗尽证据（R187）
| 变体 | 结果 |
|---|---|
| (4) 契约正确版：stc_slot_fp[2]，活跃容器已持有同指纹则完全不 reset 并 _next = _cur；否则走 legacy ^1 | 新 abort：ggml.c:1805 GGML_ASSERT(obj_new)（ggml arena 耗尽），不再是 meta 的节点断言 |
| (5) 对照臂两次 | fpc0a 96.30 / fpc0b 95.95 t/s，sha256 门一致，[FPSTAT] distinct=24 same_as_prev=0 逐位相同 |

为什么 (4) 仍然错（推理与证据一致）：_next 只能在上一次调用末尾设定，而重建使用的是建图阶段提交的 _index。
故 (4) 的保留分支会让下一次（指纹可能不同）的重建也落进同一个容器，新张量被塞进未 reset 的 arena，obj_new 失败。
arena 耗尽的 abort 正是这个推理的直接证据。

### 修正后的设计（下一手，尚未上机）
容器必须在重建时刻选定并 reset，而不是靠 _next 提前一轮：

    i_cur = stc_compute_index;
    i_use = (stc_slot_fp[i_cur] == fp) ? i_cur : (i_cur ^ 1);
    if (i_use != i_cur) stc_compute_index = i_use;      // 就地覆盖，供本次 rebuild 使用
    if (stc_slot_fp[i_use] != fp) { reset(stc_compute[i_use]); stc_slot_fp[i_use] = fp; }
    stc_compute_index_next = i_use;                     // 与下一次提交保持一致

两条铁律：(1) reset 的对象必须是本次要用的那个容器；(2) 持有同指纹的容器永不 reset。
（第 (2) 次尝试曾做过就地覆盖，但它用了 fp %% 8 的 8 槽向量、指纹在槽间碰撞，混淆了变量；本次用固定 2 槽 + 明确的 i_use。）
风险：2 槽只覆盖热指纹的一部分，收益按命中率打折；distinct=24 里最热的一个占 32%%。

## 实现约束（读码发现，必须先解决）
1. 现有 per-backend 图状态是**单份**：`backend_ctx->backend_configs[j].{nodes[i], cgraphs[i]}`、
   `backend_ctx->n_subgraphs`、`n_reduce_steps`、`max_nnodes`、`rebuild_fp/rebuild_done`。
2. 张量容器是**双缓冲**：`buf_ctx->stc_compute_index ^= 1` + `stc.simple_tensors/tensor_images.clear()`
   （`ggml-backend-meta.cpp:2066-2075`）。**只把 cgraphs 做多槽、不扩容器 ⇒ 槽内张量会在两次 rebuild 后被翻转重置**（静默失效）。
   ⇒ 多槽必须**同时**把 `stc_compute` 扩到 >=N 槽，并按槽位索引选择。
3. 最小侵入做法：把"整套 per-backend 图状态"打包成 `slot`（含 `backend_configs` 数组 + n_subgraphs/n_reduce_steps/max_nnodes + stc 容器索引），
   在 `graph_compute` 入口按 `rebuild_fp` 做**状态换入/换出**（LRU，N=8），miss 时才真正 rebuild。
   `backend_ctx` 里的"当前态"字段保持不变 ⇒ 计算主循环与 `allreduce_fallback` 无需改动。

## 预期收益（按 E5 成本模型）
direct 20.6 次/call x 281 us = **5.8 ms/call**；命中后这些子图能 warmup 并 replay（~30 us）⇒ 每次省约 **5.6 ms**，
capture 同步消失再省约 1 ms/call ⇒ **约 6.5 ms/call x 3.27 decode/轮 = 21 ms/轮**
⇒ 53.6 -> **~32.6 ms/轮**（tg 97 -> **~159**）。**单独一笔即可越过 150 线。**

## 上游与外部对照（都已编号入账）
- **#18934（已并）**：`nodes[0]` 作 key 的起源（"cuda graphs get disabled when there are splits"）。
- **#19754（已并）**：warmup 门（"同一 cgraph 至少两次属性一致才捕获"）。
- **#21611（已并）**：因"node 指针一直在变"而加 **LRU 淘汰**（只淘汰，**不改 key**）。
- **#28652（OPEN）**：交替变体共用一把 key 会互相作废 warmup，把投机解码推向 direct —— 就是我们。
- **#28666（已关未并）**：**唯一**试图改 key 的 PR（first_node_ptr -> + ne[]）。
- **#25406（已关未并）**：`split_graph` 每次重发 uid，uid 快路径永不命中。**#24549（OPEN）**：tensor split 下双上下文共享缓冲时干脆禁用复用。
- **结论**：上游知道机制、视其为刻意设计，**没有任何已合并的 PR 修它** ⇒ 这是真空白，我们的多槽是**自研但方向与 vLLM 一致**的解法。

## 本地 vLLM 对照（`6bb777b3` 读码，file:line 已记）
- key = **预先枚举的 capture list** 里的 `BatchDescriptor(num_tokens=PADDED, num_reqs, uniform, lora...)`（`vllm/vllm/forward_context.py:29-57`）⇒ **key 每步都重复**；
- **每个 descriptor 只在捕获前 warmup 一次，之后无条件 replay**（`v1/worker/gpu/cudagraph_utils.py:436-441, 530-543`），捕获在 `capture_model` 后**全局关闭** ⇒ **没有任何属性检查能作废已捕获的图**；
- **固定地址输入缓冲、原地写**，replay 时**断言地址未变**（`v1/worker/gpu/input_batch.py:30-40`、`model_runner.py:1362-1379, 1900-1903`）⇒ **不存在 direct 回退路径**；
- draft（DFlash2/MTP）**有自己的一套捕获图**，绝不与 target 融合（`spec_decode/dflash/speculator.py:122-146, 469-471`）。
⇒ **vLLM 的答案就是"让 key 重复 + 绝不作废已捕获的图"** —— 与候选 A 同源。
