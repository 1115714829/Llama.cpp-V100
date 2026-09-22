# B2 实现蓝图 -- 图内 selector TP 化（解 ggml meta 分裂代数）

> R277 后续。目标：把 draft 的 top-k/gate/scoring 全部图内化（= 非 TP 路径的原样），
> 消掉 8 MB logits D2H + CPU topk 0.65 + gate 0.55，预期 -2~3 ms/轮（45.94 -> ~43）。
> 失败回退：所有改动 env 门控（`GGML_SPEC_SELECTOR_INGRAPH`），备份链 = 本地 git + `/root/t8-orig/*.pre-*` + wip patch。

## 1. 已确认的机制事实（全部物证在案）

1. **断言触发点**：`ggml-backend-meta.cpp:1007-1008` `case GGML_OP_TOP_K -> handle_per_row`，
   而 `handle_per_row`（:549-552）`GGML_ASSERT(src_ss[0].axis != SPLIT_AXIS_0)`。
   `t_logits` 的分裂态 = **AXIS_0（vocab = ne0）**，来源 `handle_mul_mat` 规则 :588-594
   （权重 axis-1 + MIRRORED 激活 -> 输出 axis-0）= 「lm_head 沿 vocab 切分」的框架表达。
2. **top_k 的墙是语义性的**：归约轴 == 切分轴；每设备只能算分片 top-k，
   而分裂态一致性要求 `sum_j ne[s*n_bufs+j]*nr[s] == 逻辑尺寸`（:505-523）
   => k/设备无法直接表达成逻辑 [k] 的状态（3k != k）。
3. **现成的 all-gather 物证**：:1540-1556 `n_contributors / partial = has_contributor_mask && ne[j]==0 ? zero : tmp`
   -- PARTIAL 张量取回时按贡献掩码**求和**汇编。mul_mat K 切分（:598-601）产出 PARTIAL + AR 边界自动求和（E12）。
   => **散布（每设备把自己的行写进满尺寸零张量的对应位置）+ PARTIAL 求和 = 精确的 concat-gather**。
4. **scatter 原语已存在**：`handle_set_rows`（:760-765）支持 dst + 值/ids 配对；
   `GGML_OP_GET_ROWS_BACK`（:959）= scatter-add 语义（base[ids] += values）。
5. `get_rows`（:753-758）已支持 axis-0 表 + MIRRORED ids；sel 权重 [rank, n_vocab] 是 axis-1（vocab 切分）
   -- MIRRORED ids 下走 handle_generic 需要全 src 同态 => 要么 gather sel 权重、要么复制放置。
6. 下游 scoring 的跨设备配对问题（cand 行 home(d1) x pred 行 home(d2)）**只有两条出路**：
   sel 权重复制（每卡全量）或 gather 后 MIRRORED 运算。gather 路线统一解决全部。

## 2. 推荐实现（gather-first，改一处框架规则 + 图插三个节点）

### P1 框架规则：scatter-concat -> PARTIAL（`handle_set_rows` 或 `handle_get_rows_back` 扩展）
```
dst = MIRRORED/PARTIAL 满尺寸零张量; values = axis-k 切片; ids = 与 values 同态切片
=> 返回 {PARTIAL, ...}，每设备把 values 切片 scatter 进自己的满尺寸贡献，AR/取回时求和 = 完整张量
```
- ids 张量：vocab arange [n_vocab]，**需要与 values 同边界切片**。载体三选一（按优先序）：
  (a) 图内 `ggml_arange` 常量 + 克隆器按 values 切片同步切 ids（需读克隆代码确认配对语义后定）；
  (b) get_rows_back 的 ids 用 MIRRORED 完整 arange + 规则内声明按 values 切片配对（规则文档化）；
  (c) 不用 ids：`ggml_cpy(values_slice, view(dst, base_d))` -- base_d 每设备不同，需克隆器支持每设备视图偏移（最干净但克隆器改动最大）。
- **验证点**：克隆/建点代码在 `ggml-backend-meta.cpp` :1232-1619（缓冲分配侧）与 :2197+（子图切分侧）；
  动手前先读 `calculate_split_state` 调用者到 subgraph build 的数据流（约 :1100-1300）。

### P2 图侧（dflash.cpp，仅 TP 分支）
```
// build_dflash2_selector 入口处（TP 时）：
logits_full = scatter_gather(res->t_logits)      // P1 规则 -> PARTIAL -> 完整
sel_next_f  = scatter_gather(model.dflash_selector_next)
sel_prev_f  = scatter_gather(model.dflash_selector_prev)
// 其后与非 TP 路径逐字相同（top_k/get_rows/mul_mat 全部 MIRRORED 输入）
```
- sel_hidden [n_embd, rank] 的 gate 是 K 切分 mul_mat（:598-601 PARTIAL+AR）= **本来就能跑，无需 gather**。
- 拷贝侧：块批 `logits=false`（is_dflash2_cpu=false 已自动生效）=> 8 MB 拷贝消失；
  lattice 走 `t_h_nextn` 槽（163 KB，MIRRORED 完整）= 现成通道。

### P3 装载（dflash.cpp :145-148）
- 已改：`TENSOR_SKIP -> 0`（sel_next/prev 正常装载，axis-1 切分）；gather 路线下够用。
- 显存代价：2 x [256, 248320] F16?/F32? = 每卡 1/3（切分）≈ 85-170 MB；零额外复制。

## 3. 回退与门控
- env `GGML_SPEC_SELECTOR_INGRAPH=1` 全程门控（4 处条件已就位）；=0/unset = B5 原样。
- 备份：本地 git（未提交可 checkout）；服务器 `/root/t8-orig/{dflash,speculative}.cpp.pre-ingraph` 与 `.pre-b2gather`；
  diff 归档 `wip-sel-ingraph.patch`（B2 前置）与本文件配套的 `wip-sel-gather.patch`（待生成）。

## 4. 验证阶梯（每级一冒烟，坏了回退该级）
1. 构建 + 启动不 abort（当前卡点）。
2. `GALLOC`/图内路径日志确认 top-k 走图内；块批无 8 MB 拷贝（[RT]/spec timing 的 draft_decode 应从 5.3 降到 ~3）。
3. **AL/接受率 ~= 65% 不掉**（= gather 正确性端到端验证；掉 = 边界错位，查 P1 ids 配对）。
4. 门值 `f3edac19…`（draft 侧不改 target 贪心文本，预期保持；破 = 有 draft 泄漏进输出，必须查）。
5. ABBA 4 臂判 ms/轮 边际，>=2% 则采用为 B6。

## 4.5 机制补充（末轮解剖发现，实现前必读）

- **张量镜像机制**（`ggml_backend_meta_buffer_init_tensor_impl` :1187-1239）：每设备持有一个**切片镜像**（ne/scale 重写 + 偏移），
  CUDA 后端在镜像上跑标准算子 => axis-0 切分张量的镜像 = 每设备 [n_vocab/3, n_tokens] **自带 base_d 偏移**（免费的散布定位!）。
- **PARTIAL 张量 = 每设备满尺寸贡献副本**；取回/AR 时按**贡献者掩码求和**（:1540-1556）=> scatter + PARTIAL = 精确 all-gather（P1 的语义基础）。
- **top_k 归约不可切分的证明**（:1171 断言 `sum(ne[*])==逻辑尺寸`）：每设备 k 个候选无法表达成逻辑 [k] 的状态（3k != k）
  => **必须 gather 后再 top_k**（gather-first 路线）；gather 后全部 MIRRORED，top_k/get_rows/mul_mat **零改动**（含 handle_per_row 的 assert 自然通过）。
- **实证利器**：`buf_ctx->debug > 0` 时 :1156 逐张量打印 `SPLIT_STATE: {srcs} -> dst[op, axis, {ne x nr}]`
  => 阶梯每一级先看它确认 ids/values 镜像配对，再跑 AL 验证。
- **ids 载体首选（i）**：`ggml_arange`/I32 常量叶 [n_vocab]，靠 alloc 的切分启发式拿到与 values 同边界的 axis-0 镜像（先用 SPLIT_STATE 日志实证；
  不行则退回 (b) 规则内声明配对，或 (c) 新算子）。镜像配对语义 = set_rows 的 values[i] <-> ids[i] 逐镜像局部配对。

## 5. 若 P1 受阻的降级预案（残值实测口径，R277）
- 退化版（gate+scoring 图内、top-k 留主机）：仅 -0.6~1 ms/轮。
- A 融合：-1.1~1.3 ms/轮（混合模态批，另一条新模式线）。
- selector 线整体收兵：转 target 提交窗口（18.6 ms/轮 重叠优化）/ GPU 地板。
