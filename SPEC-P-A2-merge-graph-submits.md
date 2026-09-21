# SPEC: 合并每轮 417 次图提交（P-A 重定向版）

> 建立于 2026-09-21 Round 99。取代 `SPEC-P-A-ar-in-graph.md` 的**动机**部分（那条路线本身仍有效，但收益理由要改）。

## 1. 为什么改动机

原动机是「AR 从 53 us 压到 1cat 的 18 us，省 4.8 ms/轮」。这条已被两件事削弱：
- R1 实测：主机省 4.8 ms/轮但**轮时零改善**；
- §25.6 实测：TP3 的 `enqueue/轮` 比 TP2 **多** 4.4 ms，但总轮时仍**少** 2.2 ms。

真正的理由是 **§25.8 的主机侧账本**：每轮主机侧约 **37 ms**（提交 31.5 + 分配 5.9），GPU 侧约 30 ms，轮时 56 ms。
主机侧由这些构成：

| 项 | 每轮 | 来源 |
|---|---:|---|
| target `llama_context::graph_compute` | 20.7 ms | `[RT]` target `enqueue_us` |
| draft `llama_context::graph_compute` | 10.8 ms | `[RT]` draft `enqueue_us` |
| target + draft `alloc_graph` | 5.9 ms | `[RT]` `alloc_us` |

而 target 那一次 `graph_compute` 内部要跑完 `ggml-backend-meta.cpp:2434-2463` 的循环：
```cpp
for (i = 0; i < n_subgraphs; i++) {            // n_subgraphs = 139
    for (j = 0; j < n_backends; j++) ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);  // 3 次
    if (n_backends > 1 && i < n_subgraphs - 1) comm_allreduce(comm_ctx, nodes.data());                            // 1 次 NCCL
}
```
**每轮 139 个子图 x 3 设备 = 417 次 `graph_compute` + 138 次主机侧 NCCL 调用**，全部在主机线程上串行发生。

## 2. 已量化的下界（说明「不是 CUDA 后端慢」）
从 §24 的探针：`[GRAPH] calls=82688 per_call=3.8us avg_nodes=40` => CUDA 后端自身只占 82688 x 3.8 us = 314 ms
（按约 196 轮算 ≈ **1.6 ms/轮**）；`[AR] calls≈27056 avg_tensors=3.0` => AR 约 138 次/轮。
即：**CUDA 后端与 AR 都解释不了 20.7 ms**，大头在 meta 后端这个循环本身以及每次调用的固定开销。

## 3. 本规格要做的两件事（按顺序）

### 步骤 1（必须先做）：把 meta 循环的主机时间拆开 —— env 门控探针
在 `ggml_backend_meta_graph_compute` 里加 `GGML_META_HOST_TIMING`（默认关、零影响），累计并定期打印：
- `loop_us`：整个 for 循环的墙钟；
- `dev_us`：**只**包住 `ggml_backend_graph_compute_async` 的那三层循环；
- `ar_us`：只包住 `comm_allreduce`；
- `rebuild_us`：`needs_rebuild` 分支（第 1984 行起）的墙钟；`n_rebuild`/`n_calls`。

**判据**：若 `dev_us` 占大头 => 减少设备调用次数（见步骤 2）；若 `rebuild_us` 占大头 => 问题在 `needs_rebuild`（见下方陷阱）。

### 步骤 2：按步骤 1 的结论二选一
**2a. `dev_us` 占大头** => 目标是把 417 次调用变成 1 次图启动，即 1cat 的整轮 fullgraph 路线：
在 meta 后端里，对每个设备**外层做一次 `cudaStreamBeginCapture`**，把 i 循环（含 AR 内核）全部录进同一张图，再一次性 launch。
前提与难点：
- 内层 `ggml_backend_graph_compute_async` 自己会用 CUDA graph，**嵌套捕获非法** => 需要一个「外层捕获中」标志让 CUDA 后端走**直录**（不自己捕获）；
- AR 必须是**跑在同一条被捕获流上的内核**（NCCL 支持 stream capture，但更稳的是复用 `patches/0006` 已验证的设备侧 push 协议）；
- 形状变化时要重新实例化；本轮只要求覆盖 **decode 稳态**（ctx 固定、M=8 + 注入两种形状）。

**2b. `rebuild_us` 占大头** => 陷阱在 `ggml-backend-meta.cpp:1970`：
```cpp
const bool needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);
```
而 `ggml-backend.cpp:1588` 在 `split_graph` 里**每次都给 split 重新分配 uid** => 只要 sched 重切图，meta 就得整体重建 139 个子图。
target `reuse=270/294` 所以重建少；**draft `reuse=0/556` 所以每轮重建 2 次**（这就是 §25.7 那 3.62 ms/轮的去处之一）。
若走这条线，与 `SPEC-P-B-draft-graph-reuse.md` 是同一件事，**不要重复投入**。

## 4. 纪律与验收
- 探针：env 门控、默认零影响、ASCII、`fprintf(stderr)`。
- 任何图结构改动：**greedy sha256 必须逐位不变**（`f3edac19...`）；变了就是改到了语义，立即回滚。
- 官方口径 A/B（`CARDS=0,1,2 SPLIT=tensor L=libdir-instr P2P=1 NPRED=512`，每臂 >=2 次）+ 构建三查 + 四库 md5 + 标记串。
- 门槛：`ms/轮` 从 55.4-56.5 降到 **<= 50**；`enqueue/轮`（target+draft）从 31.5 降到 **<= 22**。
- 测量独占机器；端口守卫（AGENTS §4.26）。

## 5. 风险
- 整轮捕获会让**首次实例化**变慢（一次要实例化上千个内核），必须确认它只发生在形状首次出现时。
- `GGML_CUDA_AR_*` 的 copy-engine 路径（`allreduce.cu` 里的 D2H/H2D 分块）**不能被捕获**，走图内路线时要绕开。
- 这是本项目迄今最大的结构性改动，**先做步骤 1 的探针**，拿到 `dev_us/ar_us/rebuild_us` 的拆分再决定投哪条。
