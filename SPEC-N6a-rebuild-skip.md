# SPEC: N6a —— 让「内容没变」的图真正走快路径（三处 early-out）

> 来源：R173 的四个探针实测（`/root/fpd-chain.log`，2026-09-21 17:47）。
> 这份文件把「为什么每轮都走 direct」从**推断**变成**实测因果链**，并给出最小实现。

## 0. 实测证据（四条，互相独立）

1. **`[FPD]`（新探针，装在调度器层 `ggml-backend.cpp` 的 `alloc_graph` 入口）**：
   `calls=192 same=152 diff=39 hit8=160 dist1=20 dist2=25 dist3=23 dist4=21 dist5=20 dist6=20 dist7=16 dist8=15`
   => 相邻大图中 **79% 内容逐位相同**；细节行显示大图只有两种：**4951 节点（target）与 649 节点（draft）交替**，
   距离直方图 **dist2 最大** => **每个 key 的图内容跨轮是不变的**（周期 2 复现）。
2. **`[SCHED]`**：`BIG calls=132 splits=132 cached=0 split=257.8 us/call alloc=5437.3 us/call`；
   **把 `GGML_SCHED_SPLIT_CACHE=1` 打开后仍然是 `cached=0`**（arm B: `cache=1`，`split=259.5 alloc=5489.5`）。
   => ① 切图缓存**一次都没命中**（与 AGENTS §4.20 的旧记录一致，但现在知道原因）；② **`alloc` 是 `split` 的 21 倍**（5.4 ms/call）。
3. **`[MKEY]`**：每个子图的 `nodes[0]`（= CUDA 图的 key）**每轮都是新地址**（r=0/1/2/3 四轮给出 12 个互不相同的值，uid 也从 22 跳到 430/475/864）。
4. **`[FAK]`**（顺带）：verify 路径 = `TILE` + `need_f16_K/V=1`（`D=256 n_q=8 n_kv=256 kv_type=8`），prefill = `MMA_F16`，无投机 = `VEC`。
   => R146 的 FA 分析成立；但 KV 线已按 R170 封顶（8K 约 2 ms/轮），**不再投入**。

## 1. 因果链（已实测，不再是指推）

```
每轮 ggml_backend_sched_reset()  把 last_graph_fp 清 0   (ggml-backend.cpp:1954)
  -> 切图缓存永不命中（cached=0 实测）
  -> split_graph 重铸每个 split 图的 uid（:1588）
  -> meta needs_rebuild=true（meta:1983）
  -> 旋转 compute 容器 + 重建 bcj.nodes（新建 simple tensor，地址全变）
  -> ggml_cuda_graph_get_key() = cgraph->nodes[0]（:2739）每轮都是新地址（MKEY 实测）
  -> 每次都拿到一个**全新的** ggml_cuda_graph 对象（warmup_complete=false、node_props 为空）
  -> ggml_cuda_graph_update_required() 必然返回 true（:3052 尺寸就不同）
  -> 不走 replay，逐节点 direct
```
=> 这就是 dev 19.7 ms 里 **11.6 ms（59%）走 direct** 的根因，也是 prologue 3.3 ms 的来源。
**注意一个反直觉点：UID 快路径（:3042）不是必需的，属性 memcmp 才是判据；但 key 变了就一切归零。**

## 2. 三处必须同时成立的跳过（缺一即回落到 direct）

| # | 位置 | 现状 | 需要的改动 |
|---|---|---|---|
| 1 | `ggml-backend.cpp` split 指纹 / `sched_reset` | `sched_reset` 把 `last_graph_fp=0`；实测 `cached=0` | 让指纹**跨 reset 存活**（或改多槽） |
| 2 | `ggml-alloc.c:1078-1095` `alloc_graph` | 每次重走 leaf/node 并 `init_tensor`（**5437 us/call**） | 图未变时跳过，**或**让 `init_tensor` 幂等（见下） |
| 3 | `ggml-backend-meta.cpp:1997-2298` rebuild | uid 变就重建 + 旋转容器 | 主图内容未变时**整块跳过**（不旋转/不重解析/不改 uid） |

## 3. 最小实现（只动 `ggml-backend-meta.cpp`，两处 early-out；env 门控 `GGML_META_REBUILD_CACHE`）

1. `ggml_backend_meta_buffer_init_tensor_impl()`：容器里已有该 tensor 条目 **且缓存的张量结构逐位相同** => 直接 `return GGML_STATUS_SUCCESS`，
   不新建 simple tensor（=> 地址不变）。需要在容器里多存一份「上次的张量结构」用于比较。
2. `ggml_backend_meta_graph_compute()`：对主图算内容指纹（覆盖 `op/type/flags/ne/nb/data/buffer/view_src/view_offs` + 每个 src 的同类字段），
   与本轮上次相同 => **跳过整个 `if (needs_rebuild)` 块**（含 `stc_compute_index_next` 的旋转）。
   => `init_tensor` 与 rebuild 同时跳过时，simple tensor 的地址、`bcj.nodes`、子图 uid **全部保持不变**。
3. **先不做**：`:2` 的 `alloc_graph` 跳过 —— 若 1 生效，alloc 仍会走一遍但**不再改动任何指针**（代价 5.4 ms/call，是下一刀）。

## 4. 验证门（按顺序）

1. `GGML_META_KEY_DEBUG=1`：**`n0` 不再每轮变**（这一条是机制是否成立的判据）。
2. `GGML_CUDA_GRAPH_DEBUG=1` + `GGML_CUDA_DIRECT_DEBUG=1`：`[GRAPH] calls=... direct=` 应接近 0、`replay=` 上来；
   `[DIRECT_PROBE]` 的字段直方图应停增。
3. **正确性门**：greedy sha256 必须仍 = `f3edac19446ef641447f8391c71cb1b25a74055d662e9887890fa4a98602ca34`（逐位一致）。
4. **轮时**：期望 **-10 ~ -15 ms/轮**（prologue 3.3 + direct 11.6），即 55.5 -> 40-45 ms/轮（tg 100 -> 123-139）。
5. 同源 A/B：四库 md5 + 二进制标记串；env 门控，**默认关**，跑通后再议默认。

## 5. 风险与红线

- 改的是本仓库最敏感的文件（meta 后端）。**必须 env 门控、默认关**。
- **生命周期**：跳过容器旋转意味着 compute 容器不回收 => 必须确认「同一图重复调用时条目数不增长」（否则内存泄漏）。
- 用户已授权结构性改动（见 `HANDOFF.md` §5 注 + 本目录的 SPEC 系列）。
- 若 A/B 显示只有部分收益（例如 replay 上来了但轮时没动），先看 `[META]` 的 `dev` 是否下降 —— 那说明瓶颈已换位置。
