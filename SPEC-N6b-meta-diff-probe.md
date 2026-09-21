# SPEC: N6b 的第一颗钉子 —— 主机侧图节点逐字段差异探针

> 来源：R149 子代理规格（2026-09-21，逐行核实行号）。**这份设计此前只存在于会话上下文里，本文件是它的持久化。**
> 为什么它重要：`PLAN-to-180ts.md` §2.7 证明 **N6b（形状稳定化）是 150 t/s 的硬前提**；而 N6b 的第一步就是这份探针的输出。

## 0. 目标（一句话）

**回答「每轮到底是哪些节点的哪些字段在变、变成什么」** —— 目前我们只知道「形状 ne/nb 是抖动主体，源是 GDN 递归状态视图」，
但**没有 per-field 的系统清单**，所以无法判断哪些可以缓存、哪些必须重切图。

## 1. 插入点（精确）

```
文件: ggml/src/ggml-backend-meta.cpp
函数: ggml_backend_meta_graph_compute          (起始 :1966)
插入: :2298 之后（重建块的收尾），env 门控 GGML_META_NODE_DIFF
无关: 不改任何判定 —— 纯诊断，输出不影响行为
```

## 2. 每轮要快照什么（三层，缺一不可）

| 层 | 快照内容 | 为什么需要 |
|---|---|---|
| **外层图节点** | 对每个 `cgraph->nodes[i]`：`name, op, type, flags, ne[4], nb[4], data, buffer, view_src, view_offs, op_params` | 这是 `ggml_cuda_graph_update_required` 实际比较的东西（`ggml-cuda.cu:3057-3069` 的 memcmp 就是整块 `ggml_tensor`） |
| **每个 src** | `(ptr, data, ne[4], nb[4])` | 上游 PR #25406 的结论是「uid 每次重铸」，但**形状来自 src**；要分清是自身还是 src 在变 |
| **映后节点** | `bcj.nodes[i]` 的 `ne/nb/data`（`backend_configs[j].nodes`，meta:1788） | **这是 CUDA 后端真正看到的那一份**，现有 CUDA 侧探针看不到它 |

另外快照 `n_subgraphs`（:1804）与每个 `cgraphs[i].offset`（:1780）。

## 3. 输出形式

```
下一轮对比，按 (张量名, 字段) 聚合，打印 top-N 与「按根节点统计的全同项」
=> 得到一张『谁在变』的清单，而不是一个总数
```

## 4. 关键的现有机制（改的时候要动/要看的地方）

| 位置 | 作用 |
|---|---|
| `meta:1983` | `needs_rebuild = (cgraph->uid == 0) \|\| (cgraph->uid != backend_ctx->uid)` —— **未来把裸 uid 测试换成指纹比较** |
| `meta:2011` | `buf_ctx->stc_compute_index_next = ... ^ 1` —— **容器轮换**（两个容器交替，这是 `[MKEY]` 探针要验的东西） |
| `meta:2032` | `bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(node, j)` —— 节点映射 |
| `meta:2296` | `cgraph_ij->uid = ggml_graph_next_uid();` —— **杀死 CUDA 快路径的那一行** |
| `meta:2452` / `:2472` | 子图派发 / AR 调用点（AR 6.1 ms **不可去**，是真实 NCCL 流量） |
| `sched:1086` / `:1588` | `split_graph` 无条件重铸 uid（`:1588` 是 split 的） |
| `sched:1996` / `:2019-2043` | 已有 `ggml_backend_sched_graph_fingerprint` 与 `GGML_SCHED_SPLIT_CACHE`（**可以扩宽它，而不是新造**） |
| `cuda:3042` | uid 快路径（`:3042`）—— 被 `meta:2296` 挡掉 |
| `cuda:3057-3069` | 整块 `ggml_tensor` memcmp（本轮探针要复现的判定） |
| `cuda:5007` | `graph->warmup_complete = false;` —— 属性变化后的重置 |
| `cuda:5025-5044` | `direct=` 计数器块（**见 §6 的疑点**） |

## 5. 缓存设计要点（探针之后才做）

- 缓存**重建产物**：`backend_configs[j].nodes`、`cgraphs[i].offset`、`n_subgraphs`，并**保留** `cgraph_main->uid`（不再重铸）。
- 失效键：对节点元组做 FNV-1a（`op, type, flags, ne[4], nb[4], data, buffer, view_src, view_offs` + 每个 src 的同类元组）。
- 已知的抖动源（要覆盖）：`cache_r_l*`/`cache_s_l*`（`src/llama-memory-recurrent.cpp:102-105`）、`cache_ple_r_l*`（:112）、
  `conv_states`/`conv_input`/`conv_state_last`/`conv_state_update`（`src/models/delta-net-base.cpp:463-528`，经 `build_rs` `src/llama-graph.cpp:3480-3509`）、
  `k_conv`/`q_conv`/`v_conv`（`src/models/qwen3next.cpp:484-504`）、`Qcur_full`（:247-266）及其视图。
- 内容变但形状/指针不变的一类（权重、`inp_pos`、token id、KV 行）：元组稳定 ⇒ 图重放、内核读新字节；`res->set_inputs(&ubatch)`（`llama_context.cpp:1444`）在 `graph_compute`（:1453）之前原地写。

## 6. ⚠️ 一处**对我方现有量具的质疑**（子代理自己标注：**未验证的假设**）

`ggml_cuda_graph_get_key`（`ggml-cuda.cu:2739`）返回 `cgraph->nodes[0]`，而 meta 后端有**两个轮换容器**（`meta:419-425`、`get_simple_tensor_container` `:450-455`）。
若每个子图的第 0 个节点落在轮换容器里，则 **key 在两个指针间交替 ⇒ 每个 split 有两个 CUDA 图对象，各自需要单独热身**：
这将是「直接路径一直热」的**第二个独立根因**。

- 子代理的诚实标注：`get_simple_tensor_container` 对已在 `stc_static` 里的张量（权重及其切片）返回静态容器，
  所以「第 0 个节点是否轮换」取决于那个节点是什么 —— **未验证**。
- 已实现的判定手段：**`GGML_META_KEY_DEBUG`**（`ggml-backend-meta.cpp`，提交 `32df47bd5`，14 行）打印前 4 个子图前 12 轮的 `nodes[0]` 与 `uid`。
  判据：**同一个 `i` 的 `n0` 是否在两个值间交替**。
- 附带影响：`ggml_cuda_dp_probe`（`ggml-cuda.cu:2786`，调用点 `:3073`）的计数器**按 graph_key 分桶**；若 key 交替，则我们既有的 **13.97% direct 占比是交错统计**。

## 7. 诚实的收益估计（子代理给的，不是乐观值）

```
缓存单独        : 约 -2 ~ -3 ms/轮（只省 prologue 3.3 里的一部分）
缓存 + 形状稳定化: 约 -7 ~ -9 ms/轮（direct 路径那 11.6 ms 才可能变成 replay）
ar 6.1 ms       : 不可去（真实 NCCL 流量）
=> 缓存是【必要但不充分】的；N6b 才是 150 的那一半
```

## 8. 先在纸上定的验证门

1. 探针**零行为影响**：不设 env 时二进制与基线逐位相同（可用 md5 + 标记串双查）。
2. 探针输出必须能解释「ne/nb 抖动 3-4 倍于指针」这条既有观测 —— 否则说明快照层级选错了。
3. 缓存落地后：greedy sha256 逐位一致（`f3edac19...`）+ AL 不劣化 + 每轮 ms 下降。
