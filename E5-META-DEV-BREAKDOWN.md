# E5 实测更正 + 根因链（2026-09-21 R182）

## 1. 口径更正（推翻 R174 的 dev=17.4）
臂 `e5graph`（端口 8215，`GGML_CUDA_GRAPH_DEBUG=1 GGML_META_HOST_TIMING=1 GGML_META_KEY_DEBUG=1 LLAMA_ROUND_TIMING=1`，
NPRED=512 NODROP libdir-instr P2P=1 TP3，`MEDIAN_TG=97.33`，sha256 门 `f3edac19...` 一致，服务未变）：

| 量 | R174 账本 | **E5 实测** | 说明 |
|---|---:|---:|---|
| `[META] total` | 21.283 | **10.66**（末值） | 早期 18.13(call64) → 14.20 → 12.16 → 11.35 → ... → 10.66 |
| `[META] dev` | 17.435 | **7.187** | |
| `[META] ar` | 2.304 | 2.219 | 每次 allreduce = 2.219 ms / 46.5 = **47.7 us** |
| `sub/call x ar/call` | 未记 | **47.5 x 46.5** | 印证"48 子图 x 3 卡" |
| `[GRAPH]` | replay 78.5 / direct 18.2 / cap 3.3 % | **83.7 / 14.0 / 2.3 %** | calls=124160，每 call **149 compute** |
| tg | — | 97.33 | 与 e4slot 97.02 同档 |

⇒ **R174 的 21.28 ms/call 是"累计平均 + 臂组成"的假象**：`[META]` 打的是**从臂开始到现在的累计均值**，
前 64 次调用里有大量 prefill（耗时是 decode 的数倍），臂越短污染越重。**不得再把 21.28/17.4 当作稳态值。**
⇒ **稳态 meta 主机 = 10.5 ms/call**（768-832 段在 10.50-10.66 之间振荡，不再下降）。

## 2. 成本模型（与实测总量逐位吻合）
每 meta call 149 次 CUDA graph compute：
```
replay  125  x ~30 us  = 3.75 ms
direct   20.6 x ~281 us = 5.79 ms
capture   3.4 x ~300 us = 1.02 ms
                 合计 = 10.56 ms   vs 实测 10.66 ms/call      <-- 吻合
832 calls x 10.5 ms = 8.74 s       vs 臂生成总时长 14.49 s     <-- meta 主机占墙钟 60%
```
且 **832 次 meta call ≈ 850 次 decode**（draft `[RT] rounds=556` 是 **decode 次数**不是投机轮数：556/260 轮 = 2.14 次/轮，
target 294/260 = 1.13 次/轮；556+294 = 850）⇒ **target 与 draft 共用同一个 meta 后端上下文**。

**direct 的价钱**：20.6 x 281 us = **5.8 ms/call** x 3.27 decode/轮 = **19 ms/轮**（53.6 -> ~34.6 ms/轮，**tg ~150**）。
capture 再加 ~3 ms/轮。**这是目前唯一一笔足以单独达标的款子。**

## 2b. `[GRAPH]` 全臂进程（子代理回报的完整序列，我之前只 grep 到末值）
```
calls=256    capture=0   replay=0     direct=256    per_call=18.6us avg_nodes=39.2
calls=2048   capture=5   replay=0     direct=2043   per_call=16.1us
calls=2560   capture=393 replay=94    direct=2073   per_call=13.7us
calls=123904 capture=2838 replay=103918 direct=17148 per_call=4.3us
calls=124160 capture=2838 replay=103977 direct=17345 per_call=4.3us
props changed >= 17000 次（13.7% of computes）<-> direct 12.6%  一一对应
[GRAPH] prop diff #710000 node=kqv_out-55 (reshaped) op=RESHAPE new_ne=[64,24]  old_ne=[64,96]  new_data=0x6238c48100 old_data=0x6238c48100
[GRAPH] prop diff #712000 node=node_0        op=MUL_MAT new_ne=[5120,1] old_ne=[5120,38] new_data=0x7ffdbacc8180 old_data=0x7ffdbb01a600
```
- **前 ~2000 次 compute 全是 direct**（prefill 段）⇒ `[META]` 早期均值 18.1 ms/call 就是这么来的。
- 稳态：replay 85.4%、**direct 12.6%**、capture 2.0%；`decision per_call` 从 18.6 us 降到 4.3 us。
- **决定性一行**：`node_0 op=MUL_MAT new_ne=[5120,1] old_ne=[5120,38]` —— **同一个 CUDA 图 key（= `nodes[0]`）
  被两种形状轮流使用**，与上游 **#28652**（"alternating variants sharing one graph cache key cause
  warmup/replay interference"，OPEN 2026-09-09）标题逐字对应；关闭的 #28666 提出的修法正是
  "给 CUDA 图 key 加首节点 extents"。
- ⇒ 稳态 direct 的成因**同时**包含两类：(A) 新 key（`old_ne=[0,0] old_data=(nil)`，即该 key 从未见过）；
  (B) 既有 key 上的真实形状变化（`[5120,1]` vs `[5120,38]`）。**两者比例决定修法**，故下一步先数指纹。

## 3. 根因链（新，且与"uid 复用"无关）
```
[MKEY] n0 每轮都变（r=0: 0x7ffdc0ff0ec0 -> r=11: 0x7ffc6d001460，uid 22 -> 2094）
[GRAPH] prop diff #2000 node=node_685 op=GATED_DELTA_NET new_ne=[1920,1026] old_ne=[0,0] new_data=.. old_data=(nil)
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   old_ne=[0,0] / old_data=(nil)  =>  这不是"属性变了"，而是 **这个 CUDA 图 key 从来没出现过**
```
1. `ggml_cuda_graph_get_key(cgraph) = cgraph->nodes[0]`（`ggml-cuda.cu:2739`）⇒ **CUDA 图按 nodes[0] 指针缓存**；
2. meta 每次 rebuild 都把 47.5 个子图**重建到同一批槽位**（`backend_ctx->backend_configs[j].cgraphs[i]`），
   而 **target 与 draft 交替调用同一个 meta 上下文** ⇒ 两边的子图互相覆盖 ⇒ `nodes[0]` 每次都在变；
3. 新 key ⇒ `ggml_cuda_graph` 对象全新 ⇒ `warmup_complete=false`（`:4994-5002`，需要"连续两次属性不变"）
   ⇒ 本次 **direct**；下一次若又换 key ⇒ 又是新对象 ⇒ **warmup 永远走不完，永远 direct**；
4. ⇒ **N6a（`GGML_META_REBUILD_CACHE`，"紧邻一次内容相同才跳过重建"）打不中的原因就在这里**：
   target 与 draft 的图内容**永远不会相同**，所以"紧邻一次"这个判据永远为假 —— 与 N6a 实测"机制生效但不省时间"完全自洽。

## 4. 由此得到的修法（下一步，按"简单优先"）
**给 meta 重建缓存加"多槽"**：把 `rebuild_fp` 从"只记紧邻一次"改成**一张小的 (hash -> 子图槽位) 表（2-4 槽，LRU）**，
让 target 与 draft 各自保住自己的一套子图节点数组 ⇒ `nodes[0]` 对每个图跨轮稳定 ⇒ CUDA 图能 warmup 并 replay。
- 改动落在**我们自己的 N6a 代码**里（`ggml-backend-meta.cpp`，C++ TU，ccache 覆盖 ⇒ 增量编译快）；
- env 门控（`GGML_META_REBUILD_SLOTS=N`），可与 C0 做同源 A/B；
- 预期：direct 20.6 -> ~0-2 次/call ⇒ 省 5-6 ms/call ⇒ **~19 ms/轮**；capture 同步消失再省 ~3 ms/轮。
- 风险：多套子图槽位要各自持有 tensor 容器（`stc_compute` 已被 `stc_compute_index` 双缓冲），需要按槽位隔离缓冲区。


## 6. 两条候选修法与判据（R183 待判）
已知事实：稳态下 **replay 85.4% / direct 12.6% / capture 2.0%**，`props changed` >= 17000 次（13.7%）与 direct 一一对应；
且 `prop diff #712000 node=node_0 op=MUL_MAT new_ne=[5120,1] old_ne=[5120,38]` 证明**同一个 CUDA 图 key 被两种形状轮流使用**。
（[MKEY] 在 r=0..11 看到 n0/uid 每轮都换，但那是 **prefill 段**；稳态若 n0 也每轮换，replay 不可能有 85% ⇒ 稳态下 n0 稳定、**变的是属性**。）

### 候选 A（便宜：C++ TU，ccache 覆盖）—— meta 多槽重建缓存
在 `ggml-backend-meta.cpp` 把 N6a 的"只记紧邻一次" `rebuild_fp` 扩成 **2-4 槽 LRU 表**，让 target 与 draft（以及 draft 的两个相位）
各自保住一套子图节点数组 ⇒ 每套的 `nodes[0]`/uid/属性都稳定 ⇒ CUDA 图能 warmup 并 replay。
- **判据：`[FPSTAT] distinct` 小（约 2-6）** ⇒ 内容指纹确实只有少数几种 ⇒ 多槽必然命中 ⇒ 值得做。
- 若 `distinct` 上百 ⇒ 图内容每次都在变（形状真实变化）⇒ 多槽无效，走候选 B。

### 候选 B（贵：`ggml-cuda.cu` 需重编 CUDA 模板，10-30 min）—— 让 CUDA 图 key 形状敏感
`ggml_cuda_graph_get_key()` 现在只返回 `nodes[0]`（`ggml-cuda.cu:2739`），两种形状共用一张图对象 ⇒ 互相把对方 warmup 打回 direct。
改成 **`(nodes[0], n_nodes, nodes[0]->ne[0..1])` 的稳定内联键**（= 上游 **#28666** 对 **#28652** 提的"给 key 加首节点 extents"，已关未并）。
- 这样每个形状变体各自持有图对象 ⇒ warmup 走完 ⇒ replay。
- **判据：`[FPSTAT] distinct` 大（形状真在变）但仍只有少数几种"形状类别"** ⇒ 值得做。
- 风险：图对象数量增加（显存）；键冲突会把两个图合并（表现为持续 warmup 重置，不会算错）。

### ⚠️ 与"已证伪清单"的边界（防误伤）
已证伪的 **"多形状 decode graph 缓存"** 指的是**调度器侧**为多种 decode 形状缓存 **build+alloc**
（`phase0-round-breakdown.md:69-70`：主机侧三段合计 **2.10 ms/轮**，alloc 1.94 是大头；`HANDOFF.md:478`）。
它与本节的 **CUDA 图 warmup 失效（direct 12.6%，5.8 ms/call = 19 ms/轮）** 是**不同机制、不同量级**：
前者是"每轮重切重分配的代价"，后者是"已捕获的图因属性抖动被作废、退回逐节点启动"。
⇒ **候选 A/B 不属于已证伪项**，但引用时必须写清这个区别，否则会被后来的自己误杀。

### 修正一处我自己的推断（R184 读码）
之前由"832 meta call ≈ 850 decode"推断"target 与 draft **共用**同一个 meta 后端上下文"——**这条要修正**：
target 与 draft 是**两个 llama_model**，各自持有自己的 backend 集合，因此各有**自己的** `ggml_backend_meta_context`
（`backend_ctx->uid` / `rebuild_fp` / `rebuild_done` 都是**每实例**状态）。
而 `[META]` / `[FPSTAT]` 的计数器是函数内 `static` ⇒ **进程级共享**，所以 832 是**两个上下文之和**（≈850 次 decode）——
数字仍然自洽，但**乒乓发生在每个上下文自己的 meta 后端内部**：
- draft ctx：注入(~9 tok) ↔ 块(8 tok) 在自己后端上交替；
- target ctx：verify(8 tok) ↔ AL 注入(k tok) 在自己后端上交替。
⇒ **候选 A 的槽数应按"每个后端的相位种类"算（2-4 槽/后端），而不是"按上下文分两套"** —— 结论方向不变，实现参数要改对。

### 共同点
两条都指向同一个根：**llama.cpp 用"指针身份"当图/复用身份，而我们的负载里同一个指针对应多种形状**（target verify 8 / AL 注入 k / draft 注入 ~9 / draft 块 8）。
这与上游 **#28652**（OPEN）、**#25406**（关闭未并，uid 快路径）、**#24549**（OPEN，tensor split 下双上下文共享缓冲）是同一条线。

## 5. 下一步实验必须带的两件事
1. **`[MKEY]` 要在稳态 decode 段采样**（现在探针只打前 12 轮 = prefill 段，恰好是最会变的段）⇒
   改成"跳过前 N 轮、之后每 100 轮打一轮"才能证明稳态下 `nodes[0]` 是否也变。
2. **A/B 用同源库**（四库 md5 + 二进制标记串），每臂不同 PORT，正确性门 `f3edac19...`。
