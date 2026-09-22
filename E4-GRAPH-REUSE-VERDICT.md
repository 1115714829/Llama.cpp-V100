# E4 定论：draft 图 0% 复用是**结构性**的，不是 bug（2026-09-21 R180）

## 一句话
**llama.cpp 的图复用只对"形状稳定且连续出现的图"成立；DFlash2 draft 每轮两次 `llama_decode` 的形状
（`n_outputs` 槽位 + token/embd 种类 + n_tokens）三重不同且交替出现 ⇒ 复用率为 0 是设计使然，
"把 draft 图变成可复用"这个 ~15 ms 的奖赏**并不存在**。**

## 代码级证据（base b11053 + 本 fork，行号已核）

### 1. 复用门（两道，缺一不可）
`src/llama-context.cpp:1387`
```
if (!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)) { reuse }
```
- 第一道：`gf_res_prev_active == res` —— **`gf_res_prev_active` 是单个裸指针**（`llama-context.h:374`，
  只有一处赋值 `:1434`，两处置空 `:1402`/`:608`/`:831`）。
- 第二道：`llm_graph_result::can_reuse()`（`llama-graph.cpp:1420`）→ 先 `params.allow_reuse(other)`，
  再对**每一个** input 调 `can_reuse`，任一为假即整体为假。

### 2. 槽位是"按需不需要 logits"分的两份（`llama-context.cpp:2439`）
```
llm_graph_result * llama_context::get_gf_res_prev() {
    auto & res = gf_res_prev[n_outputs > 0];   // <-- 槽 0 / 槽 1
    ...
}
```
上游给了**两个槽**（对应 `n_outputs==0` 与 `n_outputs>0` 两种图），却只有**一个** active 指针
⇒ 槽位交替 = 每次调用都换 `res` ⇒ 第一道门必失败。

### 3. `allow_reuse` 的形状判据（`src/llama-graph.h:819-853`）
```
ubatch.equal_seqs()/n_tokens/n_seq_tokens/n_seqs/n_seqs_unq 全等
且 ( (!a.token && !b.token) || (!a.embd && !b.embd) || (a.token && b.token && a.embd && b.embd) )
且 n_outputs 相等、samplers 相等
```
注意中间那一行：**"一边只有 token、另一边只有 embd" 是明确的不兼容**。

### 4. DFlash2 draft 每轮两次 decode 的形状对比（`common/speculative.cpp`）
| | 注入前向（`:1418-1431`） | 块前向（`draft()` `:1482-1486`） |
|---|---|---|
| 输入种类 | 只有 `embd`（`batch_inject.embd`，`n_tokens=n_chunk`） | 只有 `token`（`mask_token_id`/`id_last`） |
| logits | **全部 `false`**（`:1428`）⇒ `n_outputs==0` | `!is_dflash2 || is_dflash2_cpu` = **true**（`:1485`）⇒ `n_outputs==8` |
| n_tokens | `n_chunk`，E3 实测 **tok=8.86-8.93（每轮变）** | 恒 8（`n_draft+1`），E3 实测 **tok=8.00** |
| 槽位 | **槽 0** | **槽 1** |

⇒ **三重不兼容**：① 槽位交替（单 active 指针必失败）；② token/embd 互斥（第 3 条明确为假）；
③ n_tokens 不等（8.86 vs 8.00，且注入侧每轮还在变）。**任何一条单独就足以判 0%。**

### 5. 对照组自洽（把 target 的 92% 也解释了）
target 每轮**只有一次** verify decode、形状恒定（8 token、`n_outputs>0`）
⇒ 槽位不变 + n_tokens 不变 + token/embd 种类不变 ⇒ `reuse=270/294=92%`（E2 实测）。
**同一个机制同时解释 0% 与 92%** ⇒ 结论可信，不是巧合。

## 为什么"按槽位记住 active"也救不了（关键否证）
`ggml_backend_sched` **只有单份 splits/graph/缓冲池**（AGENTS §4.20）。
miss 路径会调 `ggml_backend_sched_reset(sched)`（`llama-context.cpp:1405`）**释放全部图分配**。
⇒ 图 A（槽 0）在第 2 次调用建图 B（槽 1）时，A 的 buffer 已经死了；
   第 3 次就算"按槽位"判定 A 可复用，A 的 tensor 也指向已释放的分配 ⇒ **悬垂**。
⇒ 上游只跟踪"紧邻的上一次"是**有意为之的正确设计**，不是遗漏。
⇒ **因此"让 draft 图可复用"这条路在单 sched 架构下不存在**（除非改成整轮单图 / 静态形状 = 大改动）。

## 这一条改写了什么
- **撤销**本轮收敛出的头号项：「让 draft ctx 的图能复用（潜在 ~15 ms/轮）」——**该奖赏不存在**。
- E2/E3 测到的 draft 每轮 ~7.4 ms（`alloc≈1.9 + enqueue≈5.5`）**不是"复用失败的代价"**，
  而是**"每轮必须重建两张不同图"的代价**。要省只能**减少每轮的建图次数**，不能靠复用。
- 唯一可动的形态：把每轮 **2 次 decode 合成 1 次**（注入 + 块在同一张图里，即注释 `:1396` 所说的
  "fused decode encodes and injects them into the K/V cache at the target positions" 路线）。
  预期收益 = 省掉一次 build+alloc+enqueue ≈ **3.7 ms/轮**（53.6 -> ~50，tg 97 -> ~104），
  **不是 7.4**（合并后那一次仍要建图）。
- ⇒ 通往 150 t/s（ms/轮 <= 37.0）的主战场**仍然是 [META] 的 `dev=17.4 ms/call` 主机逐节点派发**
  （AGENTS §1 的 R174 账本），draft 建图只是它的一个 7.4 ms 的分项。

## ★ 已上机实证闭合（R181，臂 `e4slot`，端口 8214，`LLAMA_GRAPH_SLOT_DEBUG=1 LLAMA_ROUND_TIMING=1`）

**判据完全命中，且比读码预测更干净**（原样摘录 48 行 `[SLOT]` 里的关键序列）：

```
[SLOT] ctx=Qwen3.8-27B         call=5  n_outputs=8 slot=1 res=0x48647cb0 prev_active=0x48647cb0 hit=0
[SLOT] ctx=Qwen3.8-27B         call=6  n_outputs=8 slot=1 res=0x48647cb0 prev_active=0x48647cb0 hit=1   <-- 此后一路 hit=1
[SLOT] ctx=Qwen3.8-27B         call=24 n_outputs=8 slot=1 res=0x48647cb0 prev_active=0x48647cb0 hit=1
[SLOT] ctx=Qwen3.8-27B-DFlash2 call=5  n_outputs=8 slot=1 res=0x3f83e220 prev_active=0x48852fe0 hit=0
[SLOT] ctx=Qwen3.8-27B-DFlash2 call=6  n_outputs=0 slot=0 res=0x48852fe0 prev_active=0x3f83e220 hit=0
[SLOT] ctx=Qwen3.8-27B-DFlash2 call=7  n_outputs=8 slot=1 res=0x3f83e220 prev_active=0x48852fe0 hit=0
[SLOT] ctx=Qwen3.8-27B-DFlash2 call=8  n_outputs=0 slot=0 res=0x48852fe0 prev_active=0x3f83e220 hit=0
   ... 一直到 call=24 都是 `hit=0`，`res` 在 0x3f83e220(槽1) / 0x48852fe0(槽0) 之间完美乒乓
```

- **target ctx：24 次里有 19 次 `hit=1`**（前 5 次是 warmup/prefill，之后 slot/n_outputs/res 三者恒定）
  ⇒ 与 `[RT]` 的 `rounds=294 reuse=270 rebuild=24`（92%）一致。
- **draft ctx：24 次里 `hit=0`，一次都没有**；`[RT]` 全臂 `rounds=556 reuse=0 rebuild=556`。
  `n_outputs` 在 8(槽1, 块前向) 与 0(槽0, 注入前向) 之间交替，`prev_active` 永远指向**另一只** res。
  ⇒ **三道门里第①道（单 active 指针 + 双槽）就在单独起作用**，②③甚至不用出场。
- ⇒ **判决坐实：draft 的 0% 是结构性设计结果，不是可修的 bug。**

### 同一臂的 [RT] 读数（与 E2/E3 逐位吻合，可作分母）
| ctx | 轮数 | reuse | build | alloc | setin | enqueue |
|---|---:|---:|---:|---:|---:|---:|
| DFlash2（全臂） | 556 | **0** | 70826 us (127 us/轮) | 1052359 us (**1892 us/轮**) | 23011 us (41 us/轮) | 3125973 us (**5622 us/轮**) |
| target（全臂） | 294 | 270 | 43583 us | 663614 us (2257 us/轮) | 10768 us | 6075487 us (**20665 us/轮**) |

⇒ 每轮两大块：**target enqueue 20.7 ms + draft (build+alloc+enqueue) 7.6 ms = 28.3 ms/轮**（占 53.6 ms 轮时的 53%）。
⇒ 这就是 `[META] dev` 的落点，见 `E5-META-DEV-BREAKDOWN.md`。

### 正确性门（同一次跑）
`greedy sha256=f3edac19446ef641447f8391c71cb1b25a74055d662e9887890fa4a98602ca34` **一致**；
`MEDIAN_TG=97.02`（AL 5.55 ⇒ **ms/轮 57.4**）、`ARM_RC=0`、`MODEL_PATH_OK`、服务 `inactive/active/active` 未变、GPU 全空。
