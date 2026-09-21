# SPEC: draft 上下文图复用率 0% 的根因与修法（P-B 改写版）

> 建立于 2026-09-21 Round 97。全部行号已主线核对。
> 现象来自 `SESSION-2026-09-20-measurements.md` §25.7 的两行 `[RT] perf`。

## 1. 现象（NPRED=512 官方口径，`/tmp/p60-tp3a-server.log`）

| 上下文 | rounds | reuse | rebuild | alloc_us |
|---|---:|---:|---:|---:|
| target `Qwen3.8-27B` | 294 | 270 | 24 | 648 186 |
| draft `Qwen3.8-27B-DFlash2` | 556 | **0** | **556** | **1 042 602** |

- draft 5 层 / 1.14 GB，alloc 总量却是 65 层 / 29 GB target 的 1.6 倍。
- draft 每轮 2 次 `process_ubatch`（556 ≈ 2 x 278 轮），target 每轮 1 次（294 ≈ 288 轮）。
- 摊到每轮：draft alloc 约 1.9 ms + build 约 0.13 ms；`draft_decode` 墙钟 13.60 ms/轮。

## 2. 根因（已核对的源码事实）

`src/llama-context.h:371-374`：
```cpp
// Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
std::array<llm_graph_result_ptr, 2> gf_res_prev;
llm_graph_result_ptr gf_res_reserve;

llm_graph_result * gf_res_prev_active = nullptr;   // <-- 只有一个
```

`src/llama-context.cpp:2424-2430`：
```cpp
llm_graph_result * llama_context::get_gf_res_prev() {
    auto & res = gf_res_prev[n_outputs > 0];      // <-- 两个槽，按 n_outputs 选
    if (!res) { res.reset(new llm_graph_result(gf_res_reserve->get_max_nodes())); }
    return res.get();
}
```

`src/llama-context.cpp:1352 / 1372 / 1387 / 1419`：
```cpp
auto * res = get_gf_res_prev();
if (!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)) { n_reused++; }
else { gf_res_prev_active = nullptr; ...reset/sched_reset/build_graph/alloc_graph...; gf_res_prev_active = res; }
```

**结论**：缓存有 2 个槽、`gf_res_prev_active` 只有 1 个指针 => 只要相邻两次 `process_ubatch` 在
「有输出 / 无输出」之间**严格交替**（DFlash2 的 注入[n_outputs==0] -> 块前向[n_outputs>0] 正是如此），
`gf_res_prev_active == res` **永远为假** => 每次调用都走 reset + 重切 + 重分配 => `reuse=0`。
target 每轮只有 1 次调用，同槽连续命中，所以 reuse 92%。

## 3. 候选修法（按风险排序）

### A. 让 `gf_res_prev_active` 也按槽索引（改动最小，3 行）
```cpp
std::array<llm_graph_result *, 2> gf_res_prev_active = { nullptr, nullptr };
// process_ubatch:
const size_t i_res = n_outputs > 0 ? 1 : 0;
auto * res = get_gf_res_prev();
if (!graph_reuse_disable && gf_res_prev_active[i_res] == res && res->can_reuse(gparams)) { ... }
else { gf_res_prev_active[i_res] = nullptr; ...rebuild...; gf_res_prev_active[i_res] = res; }
```
**必须先证明的安全性**：`llm_graph_result` 不持有 arena（`src/llama-graph.h:897-971` 确认），
arena 是 `ggml_backend_sched` 里**唯一**的那份 gallocr。在槽 1 上跑过 `alloc_graph` 之后直接复用槽 0 的图，
槽 0 各 tensor 的 `data` 指针是否仍指向有效且互不重叠的区间 —— 这决定 A 是 3 行改动还是要配套改 arena。
旁证：同文件注释写的是 `distinct CUDA graph cache keys`（指 CUDA 后端按 `ggml_cgraph*` 缓存 CUDA 图），
**不是**“每个槽一份显存 arena” => 不能据注释断定安全。

### A0. ⛔ 路线 A 已被主线否掉（Round 98，读了 sched 源码之后）

`ggml/src/ggml-backend.cpp:2080-2093`：
```cpp
enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    if (!sched->is_reset && !sched->is_alloc) { ggml_backend_sched_reset(sched); }
    if (!sched->is_alloc) { if (!ggml_backend_sched_alloc_graph(sched, graph)) return GGML_STATUS_ALLOC_FAILED; }
    return ggml_backend_sched_compute_splits(sched);   // <-- 只认 sched->splits
}
```

复用路径（`llama-context.cpp:1372`）**跳过** `alloc_graph` => `sched->is_alloc` 仍为 true => `graph_compute` 不会重新分配，
直接 `compute_splits(sched)`，而 `sched->splits` 对应的是**上一次 `alloc_graph` 传进去的那张图**（`sched->graph` 是 sched 内部唯一的一份拷贝，`ggml-backend.cpp:1482-1492`）。

**结论**：`gf_res_prev_active` 这个单指针是**承重的**，不是疏漏 —— 它保证「被复用的 res」就是「当前在 sched 里已分配的那张图」。
若按路线 A 改成按槽索引，就会在「复用了槽 0 的 res、但 sched 里分配的是槽 1 的图」时，
**用槽 1 的 splits 去算槽 0 的 tensor** => 结果错乱。**路线 A 作废，禁止实施。**

### A1. 因此真正的修法必须让**两个槽各自保持已分配**
可选：① 每个槽一份 sched（各自 gallocr/arena）；② 让 sched 支持多张常驻图（`is_alloc` 按图记录）。
两者都要付出**双份 compute arena 显存**的代价 —— 而 target 的权重已占 7.3 GB/卡（16 GB 卡），风险高。
⚠️ 这正是 AGENTS §4.20 记为「按批大小分槽的 arena 实测无效」的方向，**必须先解释清楚为什么上次无效**再投入。

### B. 消除交替本身（不碰复用机制）
让 draft 的注入步也走 `n_outputs > 0` 的槽（注入本来就不需要 logits，属于浪费，但只要能复用就值），
或把注入与块前向合成**一次** `llama_decode`。后者更彻底，但动 `common/speculative.cpp` 的调用结构。

### C. 不动复用，直接省掉 alloc（保底）
若 A/B 都不安全，则至少给 draft 走**静态形状**：`GGML_SCHED_SPLIT_CACHE` 与“按批大小分槽 arena”此前实测无效（AGENTS §4.20），
所以这条路要先解释为什么无效，不投入。

## 4. 第一步必须是**先测假设**，不是先改
在 `process_ubatch` 里加 env 门控（`LLAMA_GRAPH_SLOT_DEBUG`）打印：`n_outputs`、`gtype`、`res` 指针、是否命中复用。
一次 8K 短跑（NPRED=64）即可判定：若 draft 的两个槽严格交替、target 恒为同槽，假设成立；否则本 SPEC 作废。
探针必须 env 门控、默认零影响（沿用 `LLAMA_ROUND_TIMING` 的写法）。

## 5. 验收（沿用项目纪律）
1. 判定假设：draft 的两槽交替率、target 的同槽率。
2. 修后：draft `reuse > 0` 且 `alloc_us` 显著下降；**greedy sha256 逐位不变**（f3edac19...，draft 改动不应影响数值）。
3. 官方口径 A/B（`CARDS=0,1,2 SPLIT=tensor L=libdir-instr P2P=1 NPRED=512`，每臂 >=2 次）：
   `MEDIAN_TG` 与 `ms/轮` 都要改善；门槛：`draft_decode <= 10 ms/轮`，`ms/轮 <= 54`（当前 55.4-56.5）。
4. 构建三查 + 四库 md5 + 二进制标记串；测量独占机器。

## 6. 已知风险
- 交替复用可能让两张图的中转 tensor 在**同一块 arena** 上重叠（见 §3A），若如此则必须同时给每个槽独立 arena ——
  那正是 AGENTS §4.20 记录为“实测无效”的方向，需要先解释清楚再投入。
- 任何改动都**不得**改变数值：draft 图的复用只影响内存与调度，不影响算子顺序，所以 sha256 必须逐位不变；
  一旦 sha256 变了，说明改到了语义，立刻回滚。
