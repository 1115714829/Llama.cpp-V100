# SPEC: 削减 CUDA 图的 direct 执行占比（P-E）

> 建立于 2026-09-21 Round 105/106。这是目前**唯一有量化依据且量级够大**的抓手。
> 取代 P-A（已证伪，见 `SESSION` §25.10）。

## 1. 实测证据（`/tmp/meta2.txt`，TAG=metadiag2，NPRED=64）
```
[GRAPH] calls=30208 capture=1596 replay=18495 direct=10117 decision_us=138011 per_call=4.6 avg_nodes=39.9
[META]  calls=192 sub/call=49.3 ar/call=48.3 | total=18.543 loop=16.087 dev=13.313 ar=2.770 ms/call | prologue=2.456
```
- 两者的调用数对得上（`3 x 49.3 x 205 ≈ 30208`）=> **是同一批调用**，可直接对账。
- `dev` / 次设备调用 = 13.313 ms / (3 x 49.3) = **约 90 µs**；其中 decision 只占 **4.6 µs（5%）**，其余约 85 µs 在 decision 之后。
- **capture 5.3% / replay 61.2% / direct 33.5%**。direct = 主机逐节点派发（约 40 节点/图）。
  按 `0.335 x 约 200 µs + 0.612 x 约 20 µs + 4.6 ≈ 85 µs` 与实测吻合。

## 2. 预期收益
若 direct 压到接近 0（全部重放），`dev` 约 90 -> 约 20-25 µs/次 => 每个 meta 调用省约 10 ms => **每轮约省 10 ms（约 18%）**。

## 3. 机制（已定位行号）
`ggml/src/ggml-cuda/ggml-cuda.cu:4665` `ggml_backend_cuda_graph_compute`；关键段 `4698-4711`：
```cpp
bool use_cuda_graph = false;
bool cuda_graph_update_required = false;
...
use_cuda_graph = true;      // 4700
// else: properties changed or first call - execute directly (use_cuda_graph stays false)   // 4703
...
use_cuda_graph = true;      // 4711
```
即：**只要缓存 CUDA 图的属性与新图不一致，就退回 direct 执行。**

## 3b. ★ 因果链已闭合（Round 110，`ggml-cuda.cu:2743-2776`）
```cpp
if (cgraph->uid != 0 && cgraph->uid == graph->uid) { return false; }   // uid 快路径：uid 没变才算「无属性变化」
graph->uid = cgraph->uid;
for (i < n_nodes) { memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor)); ...memcmp... }
```
完整链条：
```
process_ubatch 重建图（draft 每次 / target 24 次）
  -> ggml_backend_sched_alloc_graph -> split_graph 重新分配每个 split 的 uid（ggml-backend.cpp:1588）
  -> uid 快路径失效 -> 走 O(n) 全量 memcmp
  -> memcmp 比的是整个 ggml_tensor（含 data 指针），sched 重分配使指针移动 => 必然不等
  -> properties_changed = true -> direct 执行 + warmup_complete = false（下次还需连续 2 次稳定）
```
**结论：P-B（draft 每次重建）与 P-E（14% direct）是同一个根因。** draft 的 `reuse=0` 不只白花 3.6 ms 的 alloc，
还每轮制造大量 uid 变化 => 属性抖动 => direct。**修其一即打两个目标。**
（更正：Round 100/101 曾把 draft `reuse=0` 判为「#28549 的有意设计、上限只有 3.9 ms/轮」—— 那**只算了 alloc，漏掉了它对 direct 的连带影响**。）

⚠️ **上限**：即使照 PR #25406 把 uid 稳定下来，**步骤 5 的指针移动仍会让 memcmp 不等**。
所以还需要**分配稳定**（静态形状 / 不重分配），这就把 P-C 与静态形状路线接了进来。

## 4. 第一步：把「为什么退回 direct」问清楚（必须先做）
现成的 `GGML_CUDA_GRAPH_DEBUG` 属性差异探针**会误报**：Round 104 实测它打印的 `prop diff` 行里
`new_ne == old_ne` **且** `new_data == old_data`，属性其实没变（340 行里大量如此）。**不要用它当依据。**

要做的事：在 `4698-4711` 那几个把 `use_cuda_graph` 置 false 的分支上，加 env 门控（`GGML_CUDA_DIRECT_DEBUG`）打印
**具体是哪一项不符**（哪一个 tensor、哪一项属性、旧值/新值），并且只在 direct 判定发生时打印、每个进程限流（如每 4096 次 1 行）。

判据：拿到 direct 的**属性不符清单按频次排序**。若集中在「mask/KV 视图宽度」「递归状态视图指针」这两类，
则与 P-C（mask/KV 分桶）和 A2（递归状态索引式写入）**是同一个根**，那两项的收益要按 decode 稳态重新估值（可能远大于原文的 prefill-only 估计）。

## 5. 先复测占比（正在跑）
pbdiag 那次 direct = **18%**（15071/82688），metadiag2 是 **33.5%** —— **比例在漂**。
必须先在同一库、官方口径（`CARDS=0,1,2 SPLIT=tensor L=libdir-instr P2P=1 NPRED=512`）下跑 **>=2 臂**确认稳定值，
否则优化的目标本身是浮动的。脚本 `/root/direct-ab.sh`，结果 `/tmp/direct-ab.txt`。

## 6. 验收
1. direct 占比：由约 33% 降到 **<5%**；
2. `dev`/次：由约 90 µs 降到 **<=25 µs**；
3. 官方口径 `ms/轮` 由 55.4-56.5 降到 **<=50**；
4. **greedy sha256 逐位不变**（`f3edac19...`）—— 若变了说明改到了语义，立即回滚；
5. 构建三查 + 四库 md5 + 标记串；测量独占机器；端口守卫。

## 7. 风险
- direct 是**为了正确性**存在的退路：属性确实变了却强行重放，会算错。所以目标是**消除属性变化**，不是绕过判定。
- 若清单显示属性变化来自「每轮 KV 长度增长」这类**本质变化**，则只能靠形状分桶（P-C）或静态形状（整轮单图）解决。
