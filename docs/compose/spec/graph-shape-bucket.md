---
feature: graph-shape-bucket
status: design
updated: 2026-09-23 (R296 design)
branch: feat/graph-shape-bucket
commits:
---

# P-GRAPHTAX 形状稳定化：n_kv 桶量化 + kq_mask 定尺

## [S0] 依据（R294/R295 双证 + 源码定位）

- R294：256K 服务负载 target 489 调用 **rebuild 468（95.7%）**，alloc 198 + setin 73 + enqueue 322 = **593 ms/调用 ≈ ubatch 墙钟全额** => 预填充主机受限。
- R295：host 调用数 4x↓（ub512->2048）直接兑现预填充 +27% => host 税归因成立；但 ub 双刃（吐字 -20%）=> 需形状稳定化而非粗调 ub。
- 源码打击点：
  1. `llama-graph.cpp:48-62 can_reuse_kq_mask`：`kq_mask->ne[0] == n_kv` 是形状锁唯一判据（mask = [n_kv, n_tokens/n_stream, 1, n_stream]）；
  2. `llama-graph.cpp:29 build_attn_inp_kq_mask`：按当前 n_kv 定形建张量；
  3. `llama-kv-cache.cpp:1555 set_input_kq_mask_impl`：逐元素填 n_kv x n_tokens（236K x 512 = 484 MB/调用）= setin 税主嫌；
  4. `llama-graph.cpp:2646 ggml_flash_attn_ext`：K/V 为 n_kv 视图，mask 语义为加性（-inf 屏蔽）。

## [S1] 设计

**n_kv 桶量化（几何阶梯）+ kq_mask/KV 视图定尺到桶上界。**

1. `bucket(n_kv)`：几何阶梯，比值默认 **1.25**（env `GGML_KV_BUCKET_RATIO`，**判值**；`=0` 关闭 = 逐位等价原路径）。256K 全程仅 ~29 个桶形状（462 次 rebuild -> ~29）；填充/扫描膨胀 <=25%（平均 ~12%）。
2. `build_attn_inp_kq_mask`：ne[0] = bucket(n_kv)；`can_reuse_kq_mask` 同步比较 bucket(n_kv)。**桶内复现 => can_reuse 命中 => T8 槽/build/uid 快路径全线复活。**
3. `set_input_kq_mask`：causal/swa 真实区照旧按真 n_kv 填；**尾部 [n_kv, bucket) 整列 -inf**（swa 变体同规则）。
4. K/V 视图 ne[1] 定尺到 bucket：padding 行指向 cache 物理行（预分配 n_ctx 内，地址合法）；其内容为垃圾/旧数据但被 -inf 完全屏蔽。
5. env 默认关；`LLAMA_KV_BUCKET_RATIO=1.25` 开启实验臂（唯一变量）。

## [S2] 数值安全预论证（门值判据）

padding 行经 add mask (-inf) -> softmax exp(-inf)=0 -> 对 PV 贡献恒 0；行 max/sum 不受影响（-inf 不参与 max，sum +0）。
=> **输出应位级不变 => greedy sha256 门必须原样过**；若门破 = 论证有漏洞（疑点：-inf 与行 max=-inf 的 NaN 保护路径、swa 窗口边界），**停机复盘，不得强行重立门**。

## [S3] 预登记及格线（对照 BL3 基线，stress-256k 同尺双 rep）

| 指标 | BL3 基线 | 及格 | 预期 |
|---|---|---|---|
| 256K 预填充 TTFT | 289.2 s | **>= +30%（<=202 s）** | +35~50% |
| 256K 吐字 tg | 32.7 t/s | **不回退 >2%** | 平或小升（decode 轮 reuse 同受益） |
| 机制自证 `[RT] perf` | rebuild 468/489 | **rebuild <= 30，reuse >= 90%** | ~29 桶 |
| 显存峰值 | 基线 | 不升 >0.5 GB | +~12% mask |
| 门值 | f3edac19... | **原样过** | 原样 |

及格 = 全表过。失败侧写入失败记录（只改灰）。

## [S4] 明确不做 / 风险

- 不动 GDN/mamba 状态路径（hybrid-idx 的 n_kv 引用无关图形状）。
- dsv4/mla/lid 共用 build_attn_inp_kq_mask：桶化随同一开关，默认关不影响其它模型。
- swa 变体尾部 -inf 规则必须与窗口逻辑一致（风险点 #2）。
- -inf 的 NaN 保护（next_max==-inf -> 0.0f 行为）已在 splitd_n32 同款保护逻辑中存在；sm70 kernel 路径亦须核对。

## Tasks

- [ ] T1: bucket() + env 解析 + build_attn_inp_kq_mask / can_reuse_kq_mask 定尺（covers: S1-1/2; 判据: =0 逐位等价）
- [ ] T2: set_input_kq_mask 尾部 -inf（causal/swa 双变体）+ K/V 视图定尺（covers: S1-3/4; 判据: 门值原样）
- [ ] T3: 服务器构建（/root/llm/test/v100-opt/llama.cpp，t15-build 纪律 + 标记串三查）+ 冒烟
- [ ] T4: 同尺 A/B（BL3 vs 开关臂，双 rep，[RT] perf 机制自证）对照 [S3]，达标采用；填 AGENTS §1.0 与账本 R296
