# RESEARCH-gqa-packing.md - T2-001 GQA packing (ncols2=3) 移植可行性研究

> 日期: 2026-09-20 | 研究基线: `llama.cpp` HEAD `79504e72b`, base tag `b11053`
> 外部对象: https://github.com/jackinthebox52/qwen38-v100-serve
> 证据规则: 每条结论带 file:line; 「实测」= 本次会话亲自跑出来的; 「文档宣称」= 第三方 README/补丁 message 的说法, 未独立复现
> 本文件只做研究, 未改动任何 `llama.cpp` 源码 (验证用的 apply 全在 %TEMP% 的副本里做, 已删除)

---

## 0. 结论速览 (先读这段)

1. **原始 diff 拿到了**, 不是拼出来的: `patches/0001-t2-001-gqa-packing-sm70.patch`, 16534 B, SHA256 `7AF65463B25955E70EF8E57C39C544E459B05D39135634190905E9D335C9C6EE`, 16 个 hunk, 只改 1 个文件 (`ggml/src/ggml-cuda/fattn-vec.cuh`, +71/-21)。

2. **补丁能干净落到我们的 b11053 树上** (实测): `git apply --check` 退出码 0; hunk #1-#8 偏移 0, hunk #9-#16 偏移 **-2 行**; 无 reject, 无需手工改。偏移来源是 b10793 -> b11053 之间 `fattn-vec.cuh` 唯一的一处改动 (`__syncwarp()` -> `ggml_cuda_syncwarp()`, b10793 的 320-322 三行合成 b11053 的 320 一行), 正好落在两个 hunk 之间的空隙里。

3. **本地 `fattn-vec.cuh` 与上游 b11053 逐字节一致** (实测, 去掉 CRLF 差异后 `-ceq` 为 True)。我们 fork 的 13 个改动文件里不含它, 所以上游 diff 可以直接当基线。

4. **补丁只碰 `fattn-vec.cuh`, 不需要改 `fattn.cu`**。选核逻辑不加也行, 因为打包判定被放在 `ggml_cuda_flash_attn_ext_vec_case` 内部 (per-type 的 explicit instantiation 里)。

5. **文档宣称的机制描述与代码不符** (已实测核实): README 说 vec kernel 按 2 的幂打包到 `ncols2 = 2`, 因此剩 3x 冗余。**实际 `flash_attn_ext_vec` 里 ncols2 恒等于 1** (`fattn-vec.cuh:541` `launch_fattn<D, cols_per_block, 1>`), 即 **一个 block 只算 1 个 Q 头, 零打包**。真实块数是每序列 24 个 (6 个块共享 1 个 KV 头, **6x 冗余**), 打包后 8 个 (2 个块共享 1 个 KV 头)。README 的 "powers-of-two 阶梯 ncols2 = 1,2,4,8" 描述的是 **MMA 路径** (`fattn.cu:200-222`) 而不是 vec kernel。被消掉的因子确实是 3 (24->8), 所以第三方测到的 3.07x 流量下降与真实块数变化自洽; 但 "只打 2 头" 这个说法是错的。

6. **⚠️ 头号发现: 该补丁在我们当前生产配置下是 no-op**。补丁用 `if constexpr (packing_supported)` 限定 **只对 F16/BF16 KV** 生效; 我们所有口径都是 `-ctk q8_0 -ctv q8_0` (`stress-test-256k.md:24`, `l3-acceptance.md:4`, `phase0-round-breakdown.md:16`), Q8_0 走的是 `ggml_cuda_flash_attn_ext_vec_case<256, Q8_0, Q8_0>`, `packing_supported == false`, 编译期就绕过了。**要吃到这个补丁, 必须先切 f16 KV** —— 而这正好已经是 `1CAT-PORT-BACKLOG.md:82` 的 P0 项 (零代码, 独立收益)。两者是同一个动作的两步。

7. **只对 decode (`Q->ne[1] == 1`) 生效**。spec-decode 的 8-token 验证步 (AL 8) 在 Volta 上 `Q->ne[1]*gqa_ratio_eff = 8*2 = 16 <= 16` -> 走 TILE 核 (`fattn.cu:648`); prefill (`Q->ne[1]` 大) 走 MMA。**对 256K prefill / TTFT 一点帮助都没有** —— 那仍是我们最大的头寸。

8. 风险点集中在两处: (a) 打包后块数少 3 倍, 短上下文 (`ntiles_KV` 小) 会掉并行度, 第三方自报 1K 处 -2.0%; (b) 新增模板实例化 (只对 4 个 F16/BF16 类型对的 TU 生效), 编译时间略增。

---

## 1. 来源与取证过程 (可复现)

| 项 | 值 | 怎么拿到的 |
|---|---|---|
| 补丁 raw | `https://raw.githubusercontent.com/jackinthebox52/qwen38-v100-serve/main/patches/0001-t2-001-gqa-packing-sm70.patch` | `web_fetch`, HTTP 200, 16534 B |
| 仓库树 | `https://api.github.com/repos/jackinthebox52/qwen38-v100-serve/git/trees/main?recursive=1` | 12 个条目, `patches/` 下 2 个文件 |
| patches/README | 同目录 `README.md`, 3578 B | `web_fetch` |
| serve.sh | 9975 B | `web_fetch` (KV dtype 的关键证据在这里, 见 §5) |
| 补丁 SHA256 | `7AF65463B25955E70EF8E57C39C544E459B05D39135634190905E9D335C9C6EE` | `Get-FileHash -Algorithm SHA256` |
| 落盘验证 | `git apply --check --verbose` -> `CHECK_EXIT=0` | 在 `%TEMP%` 内拷贝的 `ggml/src/ggml-cuda/fattn-vec.cuh` 上跑, 未触碰工作树 |
| 真实 apply | `git apply` -> `APPLY_EXIT=0` | 同上, 产物只用于 diff 取行号, 已随临时目录删除 |

补丁头部元信息 (patch 文件第 1-5 行, 属「文档宣称」): 作者写的是 `Antigravity Optimization Team <antigravity@local>`, 日期 `Fri, 5 Sep 2026`, base commit `d230ddd` = tag `b10793` (patch 内 patches/README「Applying the Patch Manually」段明确写 `a clean llama.cpp tree at commit d230ddd (tag b10793)`)。

---

## 2. 机制说明

### 2.1 ncols1 / ncols2 的语义

两个参数是**正交**的:

- `ncols1` = 每个 block 处理多少个 **Q token** (batch 维, 即 `Q->ne[1]` 方向);
- `ncols2` = 每个 block 处理多少个**共享同一个 KV 头的 Q 头** (GQA 打包维);
- 二者相乘才是 kernel 里那个 `ncols`。

证据链 (本地 b11053):

- 宿主侧网格: `fattn-common.cuh:1090-1093`
  ```
  const int ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
  const int gqa_ratio    = Q->ne[2] / K->ne[2];
  const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
  const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];
  ```
- 网格赋值: `fattn-common.cuh:1201-1203` -> `blocks_num.x = ntiles_x; blocks_num.y = parallel_blocks; blocks_num.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];`
- kernel 内: `fattn-vec.cuh:104` `const int ic0 = blockIdx.x * ncols;` (原始版用 `ncols`, 补丁改成 `ncols1`); `fattn-vec.cuh:106-107` 从 `blockIdx.z` 拆出 `sequence` 与 `head`。
- tile/mma 核里的对应写法可作对照: `fattn-common.cuh:736,744,763,827,853`。

注意 `ntiles_z_gqa` 用的是**向上取整**。所以 `gqa_ratio % ncols2 != 0` 时块会被**多派**, kernel 必须自己设防 (tile/mma 核确实设了: `fattn-common.cuh:765`、`:855` 的 `zt_gqa*ncols2 + c >= gqa_ratio`)。vec kernel 没有这个防, 所以补丁必须在**派发处**就保证整除 —— 这解释了 `if (gqa_ratio % 3 == 0 ...)` 为什么是必需的而不是可选的。

### 2.2 ncols2 在三条路径上支持哪些取值

| 路径 | ncols2 取值 | 证据 |
|---|---|---|
| **vec** (`flash_attn_ext_vec`) | **恒为 1**, 编译期写死 | `fattn-vec.cuh:541` `launch_fattn<D, cols_per_block, 1>(...)` |
| **tile** | 运行期由 `use_gqa_opt` 决定, 无 2 的幂阶梯 | `fattn-tile.cuh:1253` `use_gqa_opt = mask && ... && Q->ne[1] <= gqa_limit && K->ne[1] % FATTN_KQ_STRIDE == 0` |
| **mma_f16** | **2 的幂阶梯**, Volta 上 `{8,4,2,1}` | `fattn.cu:200-222` (`gqa_ratio % 8 == 0` -> 8, `%4` -> 4, `%2` -> 2, else 1); 非 Volta 分支在 `:242-255` |

补丁自己的注释 (`fattn-vec.cuh` 补丁后 582-583 行) 也承认了这点: `The ncols2 ladders in ggml_cuda_get_best_fattn_kernel and in the MMA path only take powers of two ... gets no KV reuse at all in this kernel: one block per query head`。**这句话是对的, 而 README 正文是错的** —— 同一个仓库内部不一致。

### 2.3 gqa_ratio_eff 怎么算, 以及它到底管什么

`fattn.cu:638-642`:

```
const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
int gqa_ratio_eff = 1;
while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
    gqa_ratio_eff *= 2;
}
```

- 它是「gqa_ratio 里能提取出的**最大 2 的幂**」, 上限 `ncols2_max`。对 D=256 -> `ncols2_max = 8` (行 638)。
- 我们的几何: `gqa_ratio = 24/4 = 6` -> `6%2==0` 得 2; `6%4!=0` 停 -> **`gqa_ratio_eff = 2`**。
- **它只是一个"等效 batch 放大倍数"的估值, 用于选核, 不参与任何 kernel 的模板参数**。用它的地方只有 5 处: `fattn.cu:645, 648, 656, 659, 662, 669`。

Volta 分支 (`fattn.cu:644-652`, `volta_mma_available(cc)` 定义在 `common.cuh:360-362`, 要求 `ggml_cuda_highest_compiled_arch(cc) == VOLTA`, 而 `common.cuh:175-177` 取的是 **<= cc 的最高已编译 arch**, 所以只要编了 sm_70 在 V100 上就为真):

```
644:  if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
645:      if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
646:          return BEST_FATTN_KERNEL_VEC;
648:      if (Q->ne[1] * gqa_ratio_eff <= 16) {
649:          return BEST_FATTN_KERNEL_TILE;
651:      return BEST_FATTN_KERNEL_MMA_F16;
```

`can_use_vector_kernel` = `Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0` (`fattn.cu:611`), 其中 `FATTN_KQ_STRIDE = 256` (`fattn-common.cuh:9`)。

**对我们要关心的三种形状, 走哪个核 (按上述代码推导):**

| 场景 | Q->ne[1] | gqa_ratio_eff | 乘积 | 选核 | 受本补丁影响? |
|---|---|---|---|---|---|
| 单流 decode (无投机) | 1 | 2 | 2 | **VEC** (行 646) | **是** |
| DFlash2 n=7 验证步 | 8 | 2 | 16 | TILE (行 649) | 否 (tile 有自己的 gqa_opt) |
| prefill (ubatch 512) | 512 | 2 | 1024 | MMA_F16 (行 651) | 否 |

### 2.4 为什么 24/4 (GQA=6) 是「3」这个因子, 而不是「2 头」

把两件事分开:

- **选核层面**: `gqa_ratio_eff = 2` 是 2 的幂截断的产物, 它让 `Q->ne[1]*gqa_ratio_eff` 停在 2, 恰好落进 VEC 档。这是 `fattn.cu` 里唯一出现 "2" 的地方。**它不决定 kernel 打包几头。**
- **kernel 层面**: vec 核 `ncols2 = 1`, 所以对 GQA=6 而言, 24 个 Q 头 = 24 个 block, 每 6 个 block 读同一个 KV 头。**冗余是 6x**。
- 打包 3 头后: `ntiles_z_gqa = ceil(6/3) = 2`, `blocks_num.z = 2 * 4 = 8`, 每 2 个 block 读同一个 KV 头。**冗余降到 2x**。
- 被消掉的因子 = 24/8 = **3**。这就是 README 那个 "3x" 的真实来源 (它把 6x 说成了 3x, 但比值对了)。

### 2.5 补丁在 kernel 内具体怎么腾出寄存器

`ncols` 从 1 变成 3, 所有 per-column 的寄存器数组都 x3: `VKQ[ncols][...]` (`fattn-vec.cuh:125/128`)、`Q_reg[ncols][...]` (`:162/164`)、`KQ_max/KQ_sum/KQ_reg/KQ_max_new[ncols]` (`:152,153,279,281`)。补丁的对策是**把每条 D 行摊到更多线程上**, 使每线程持有量按同样倍数缩小:

```
nthreads_packed = D/(2*cpy_ne) < WARP_SIZE ? D/(2*cpy_ne) : WARP_SIZE;   // 新 95 行
nthreads_f      = ncols2 > 1 ? nthreads_packed : 128 / cpy_nb;           // 新 99 行
```

对我们的 D 实算: `cpy_nb = ggml_cuda_get_max_cpy_bytes() = 16` (`common.cuh:399-409`, Volta 及以上返回 16) => `cpy_ne = cpy_nb/4 = 4` => `D/(2*cpy_ne) = 256/8 = 32`, 与 `WARP_SIZE = 32` 取小仍是 **32**。而原来的 `128/cpy_nb = 8`。所以 `nthreads_KQ` 与 `nthreads_V` 从 8 提到 32, 每线程的 Q/KQ 份额变成 1/4, 而列数变 3 倍 => 净 3/4。

**这条路径合法性已核实**: `vec_dot_fattn_vec_KQ_f16<D, nthreads>` 是完全对 nthreads 泛化的 (`fattn-common.cuh:87-115`, 循环步长 `nthreads*cpy_ne`), 没有 `nthreads==8` 的假设。而且 **D=256 的量化 KV 路径本来就在用 `nthreads_KQ_q = nthreads_V_q = 32`** (`fattn-vec.cuh:82-83`), 所以 "32 线程摊一行" 的这套布局早就在跑了 —— 补丁注释里那句 `This is exactly the thread distribution the quantized-K/V path already uses` 属实。

派生常量 (D=256, F16) 前后对照:

| 常量 | 位置 | 原始 (ncols=1) | 打包 (ncols=3) |
|---|---|---|---|
| `nthreads_KQ` / `nthreads_V` | `fattn-vec.cuh:87,88` | 8 | 32 |
| `V_rows_per_thread` | `:93` | 8 | 8 |
| `V_cols_per_iter` | `:94` | 4 | 1 |
| `ne_KQ` = ncols*D | `:122` | 256 | 768 |
| `ne_combine` = nwarps*V_cols_per_iter*D | `:123` | 4096 | 1024 |
| `VKQ` 尺寸 | `:125/128` | [1][16] half2 | [3][4] half2 |
| `Q_reg` 尺寸 | `:162/164` | [1][16] half2 | [3][4] half2 |
| `KQ_max/KQ_sum/KQ_reg/KQ_max_new` | `:152,153,279,281` | [1] | [3] |
| `KQ_max_shared/KQ_sum_shared` | `:432,433` | [1][32] | [3][32] |

注意 `VKQ` 与 `Q_reg` 的**总**寄存器占用其实**下降** (16 -> 12 个 half2), 只有 KQ_max/sum 那一组是 1 -> 3。所以 F16/BF16 上净增益; 量化 KV 因为另有 `Q_i32[ncols][2] / Q_ds[ncols][2]` (`fattn-vec.cuh:166,167`) 与 q8_1 的额外状态, 加上去就溢出 —— 补丁注释自报 `~252-255 registers at D == 256` 且 `measured: 30 STL + 17 LDL in the hot loop` (**文档宣称, 未独立验证**)。

---

## 3. 逐处改动清单 (file:line 以本地 b11053 = HEAD 79504e72b 为准)

**唯一被改的文件: `llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh`** (610 行 -> 661 行)。共 25 处, 对应补丁的 16 个 hunk。

| # | hunk | 本地行 (b11053) | 原始代码要点 | 改成 |
|---|---|---|---|---|
| 1 | 1 | **19** | `template<int D, int ncols, ggml_type type_K, ...>` | 拆成 `<int D, int ncols1, int ncols2, ...>`, 并加 5 行参数语义注释 |
| 2 | 2 | **85 之后插入** | (无) | 新增 `constexpr int ncols = ncols1*ncols2;` + 5 行注释 + `nthreads_packed` + `nthreads_f` |
| 3 | 2 | **87, 88** | `nthreads_KQ = (F16||BF16) ? 128 / cpy_nb : nthreads_KQ_q` (V 同理) | 三元里的 `128 / cpy_nb` -> `nthreads_f` |
| 4 | 3 | **104** | `const int ic0 = blockIdx.x * ncols;` | `* ncols1` |
| 5 | 3 | **106, 107** | `sequence = blockIdx.z / ne02;` `head = blockIdx.z - sequence*ne02;` | `/= (ne02/ncols2);` 与 `head0 = blockIdx.z*ncols2 - sequence*ne02;` (等价于 `(blockIdx.z % (ne02/ncols2))*ncols2`) |
| 6 | 3 | **109, 110, 111** | `Q/K/V += ... head ...` | 变量改名 `head` -> `head0`, 加一行注释 (K/V 的 `head0/gqa_ratio` 不变) |
| 7 | 4 | **115** | `const float slope = get_alibi_slope(max_bias, head, ...);` | `float slope[ncols2];` + `#pragma unroll` 循环按 `head0 + c` 逐列算 |
| 8 | 5 | **164** | `if (ncols > 1 && ic0 + j >= int(ne01.z))` | `ncols1 > 1 && ic0 + j/ncols2 >= ...` |
| 9 | 6 | **177** | `Q_f = (const float *)(Q + j*nb01);` | `Q + (j/ncols2)*nb01 + (j%ncols2)*nb02` |
| 10 | 7 | **209** | `Q_j = (const float2 *)(Q + j*nb01);` (KQ 非量化路径) | 同上 |
| 11 | 7 | **215** | `if (ncols == 1 || ic0 + j < int(ne01.z))` | `ncols1 == 1 || ic0 + j/ncols2 < ...` |
| 12 | 8 | **232** | `Q_j = (const float2 *)(Q + j*nb01);` (Q_reg 路径) | 同 #9 |
| 13 | 8 | **236** | `if (ncols == 1 || ic0 + j < int(ne01.z))` | 同 #11 |
| 14 | 9 | **280, 281** | `if (mask && (ncols == 1 || ...))` / `sum += slope*__half2float(maskh[j*ne11 + i_KQ]);` | 条件同上; `slope[j % ncols2]`, mask 行号 `(j/ncols2)*ne11` |
| 15 | 10 | **380, 381 (删)** | `const float sink = ((const float *) sinks)[head];` + 其后空行 | 删除 (下移到 j 循环内) |
| 16 | 11 | **389 之后插入** | (无) | 在 `break;` 守卫之后插入 `const float sink = ((const float *) sinks)[head0 + j % ncols2];` |
| 17 | 12 | **434** | `if (ncols > 1 && ic0 + j_VKQ >= int(ne01.z)) break;` | `ncols1 > 1 && ic0 + j_VKQ/ncols2 >= ...` |
| 18 | 13 | **501** | `dst[(((sequence*ne01.z + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*D + i0 + tid]` | `ic0 + j_VKQ/ncols2` 与 `head0 + j_VKQ%ncols2` |
| 19 | 14 | **511, 512** | `if (gridDim.y != 1 && tid < ncols && (ncols == 1 || ic0 + tid < ne01.z)) { dst_meta[... KQ_max[tid], KQ_sum[tid]] }` | 改成带 `#pragma unroll` 的 `for (jc)` + `if (tid != jc || ...) continue;`, 让 `KQ_max/KQ_sum` 用编译期常量索引 (否则寄存器数组会被打到 local memory) |
| 20 | 15 | **531** | `template <int D, int cols_per_block, ...> void ggml_cuda_flash_attn_ext_vec_case_impl` | 参数改名 `cols_per_block` -> `ncols1`, 增补 `ncols2` |
| 21 | 15 | **537** | `flash_attn_ext_vec<D, cols_per_block, type_K, ...>` | `flash_attn_ext_vec<D, ncols1, ncols2, type_K, ...>` |
| 22 | 15 | **541** | `launch_fattn<D, cols_per_block, 1>(...)` | `launch_fattn<D, ncols1, ncols2>(...)` |
| 23 | 16 | **552 之后插入 (35 行)** | (无) | 在 `if (Q->ne[1] == 1) {` 内插入打包派发: `packing_supported` (K 与 V 都是 F16/BF16) -> `if constexpr` -> 读 `dst->src[1]` 算 `gqa_ratio` -> `if (gqa_ratio % 3 == 0 && Q->ne[2] % 3 == 0)` -> `_impl<D, 1, 3, ...>` (softcap 0/1 两分支) -> `return;` |
| 24 | 16 | **556, 559** | `_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>` | 中间补 `1` (`ncols2 = 1`) |
| 25 | 16 | **567, 570** | 同上 (`cols_per_block = 2` 的批次分支) | 同上 |

**旁证: 补丁自洽性已核实**。apply 后全文搜索裸 `head` 与标量 `slope`, 结果为零 (只剩 `head0`、`slope[...]`、注释里的单词 `head`)。17 个 hunk 之外没有漏改的引用点。

**不需要改的地方 (核实过)**:

- `fattn.cu`: 补丁完全不碰。vec case 的选取方式在 b10793 与 b11053 之间被重构过 (§4), 但**调用形态 `ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst)` 两边一致** (`fattn.cu:400`), 打包判定在该函数**内部**, 所以与新老派发机制都兼容。
- 类型实例化: 打包分支是 `if constexpr (type_K/type_V 是 F16/BF16)`, 新 kernel 实例化被**自动带进已有的 TU**。实例化清单在 `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-<k>-<v>.cu` (每个文件第 5-7 行 `DECL_FATTN_VEC_CASE(64/128/256, K, V)`), 由 `generate_cu_files.py:24-26` 生成。**不需要新增文件**, 但 F16/BF16 的 4 个类型对 x 3 个 D x 2 个 softcap = 24 个额外 kernel 实例化, 编译时间会涨一点。
- `nthreads_KQ = 32` 的有效性: `static_assert(WARP_SIZE % nthreads_KQ == 0)` (`fattn-vec.cuh:90`) 满足; `get_vec_dot_KQ<F16, D, 32>` 有对应泛化实现 (`fattn-common.cuh:620-639` -> `:87`)。

---

## 4. 冲突评估: b10793 -> b11053

### 4.1 `fattn-vec.cuh` (补丁唯一改的文件): 只有 1 处, 不构成冲突

用 `raw.githubusercontent.com` 取两个 tag 的同名文件做 LCS diff (**实测**):

- b10793 = 612 行, b11053 = 610 行;
- 差异 **只有 4 行**:

```
b10793:320 -#ifndef GGML_USE_HIP
b10793:321 -        __syncwarp();
b10793:322 -#endif // GGML_USE_HIP
b11053:320 +        ggml_cuda_syncwarp();
```

即 3 行合成 1 行, 行号 **-2**。这处位于补丁 hunk #8 (结束于 b10793 的 284 行) 与 hunk #9 (起始于 b10793 的 379 行) **之间的空隙**, 不被任何 hunk 的上下文覆盖。

### 4.2 落盘实测 (决定性证据)

在 `%TEMP%` 里搭一份 b11053 的 `ggml/src/ggml-cuda/fattn-vec.cuh` 副本 (与本地 HEAD 逐字节同源, 见 4.3), 跑:

```
git apply --check --verbose t2-001.patch
```

输出 (**实测**):

```
Checking patch ggml/src/ggml-cuda/fattn-vec.cuh...
Hunk #9 succeeded at 397 (offset -2 lines).
Hunk #10 succeeded at 405 (offset -2 lines).
...
Hunk #16 succeeded at 628 (offset -2 lines).
CHECK_EXIT=0
```

=> **hunk #1-#8 偏移 0, hunk #9-#16 偏移 -2; 无 reject。** 随后真实 `git apply` 也 `APPLY_EXIT=0`。**结论: 该补丁对我们的 b11053 树可直接 apply, 不需要手工 rebase。**

> 顺带一个文档级不一致: patch 头部说 base 是 `d230ddd` (= tag b10793), 而 hunk #9-#16 的旧行号确实按 b10793 排; 这说明 `d230ddd` 在 320 行之后与 tag b10793 一致, 但与 b11053 差那 2 行。这不影响可用性。

### 4.3 本地文件与上游的关系 (实测)

把本地 `F:\vllm+llama.cpp\llama.cpp\ggml\src\ggml-cuda\fattn-vec.cuh` 去掉 CRLF 后与上游两个 tag 对比:

```
local == upstream b11053 : True
local == upstream b10793 : False
local lines: 610   b11053 lines: 610
```

=> 我们 fork 的 13 个改动文件确实不含 `fattn-vec.cuh` (与 `AGENTS.md` §5 的清单一致), 本地就是干净的 b11053。

### 4.4 `fattn.cu`: 变化很大, 但与本补丁无关

同样做 LCS diff (**实测**): `fattn.cu` 在 b10793 有 723 行, b11053 有 746 行, **209 行变动**。主要两处:

1. `ggml_cuda_flash_attn_ext_vec` 从「函数内宏展开 `FATTN_VEC_CASE`」重构成「`ggml_cuda_get_fattn_vec_case(head_size, type_K, type_V)` 返回函数指针 `fattn_vec_case_t`」, 并用 `if constexpr (GGML_CUDA_FA_<K>_<V>)` 按 CMake 选项门控 (b10793:377-404 -> b11053:395-417)。
2. `mma_f16_switch_ncols2` 里新增了 `amd_wmma_available` 分支 (b11053:224-240)。

**对移植的影响: 无。** 补丁不碰 `fattn.cu`; 两条派发路径最终都调同一个 `ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>` (定义在 `fattn-vec.cuh:544`), 打包判定在该函数体内, 所以与新旧派发机制都兼容。

### 4.5 `fattn-common.cuh`: 19 行变动, 与本补丁无交集

变更集中在 `launch_fattn` 的 `stream_k` 判定被抽成 lambda `should_use_stream_k` (`fattn-common.cuh:1136-1150`), 对 Volta + 非 stream_k 的路径没有行为差异。vec kernel 传的 `stream_k = false` (`fattn-vec.cuh:541` 最后一个参数), 所以连碰都碰不到。

---

## 5. 生效前置条件 (对我们配置的适用性) —— 最重要的实操结论

补丁要真正跑起来, **同时**满足:

| # | 条件 | 代码位置 | 我们满足吗 |
|---|---|---|---|
| C1 | **K 与 V 都是 F16 或 BF16** | 补丁新 583-586 行 `packing_supported` + `if constexpr` | **不满足 (当前)**。所有口径都是 `-ctk q8_0 -ctv q8_0` |
| C2 | `gqa_ratio % 3 == 0` | 补丁新 591 行 | 满足 (6%3==0) |
| C3 | `Q->ne[2] % 3 == 0` | 补丁新 591 行 | 满足 (24%3==0) |
| C4 | `Q->ne[1] == 1` (纯 decode) | 补丁插在 `fattn-vec.cuh:552` 的 `if (Q->ne[1] == 1)` 内 | 只有关 MTP/no-MTP 的裸 decode 满足 |
| C5 | 选核器判到 `BEST_FATTN_KERNEL_VEC` | `fattn.cu:645-647` | 满足 (1 * 2 <= 2) |
| C6 | `K->ne[1] % 256 == 0` | `fattn.cu:611` + `fattn-common.cuh:9` | 满足 (KV cache 按 FATTN_KQ_STRIDE=256 对齐) |
| C7 | D = 64 / 128 / 256 之一 | `EXTERN_DECL_FATTN_VEC_CASES` `fattn-vec.cuh:587-609` | 满足 (D=256) |

### C1 的证据 (决定性)

第三方自己的启动脚本 `serve.sh` (**实测抓取**):

- 第 223 行注释: `# - -ctk f16 -ctv f16: F16 KV cache required for numerical stability and T2-001 head packing.`
- 第 232-233 行: `-ctk f16` / `-ctv f16`

他们**明确知道这是前置条件并写进了脚本**。

我们这边 (**实测 grep 工作区**): `stress-test-256k.md:24`、`l3-acceptance.md:4`、`phase0-round-breakdown.md:16` 全是 `--cache-type-k q8_0 --cache-type-v q8_0`;`FINAL-REPORT.md:203,207` 是 `-ctk q8_0 -ctv q8_0`。

**推论 (实测 + 代码推导)**: 在 q8_0 KV 下, `ggml_cuda_get_fattn_vec_case(256, Q8_0, Q8_0)` 返回 Q8_0-Q8_0 的 case, 其 `packing_supported` 在**编译期**为 false, `if constexpr` 整段被丢弃 => **行为与未打补丁完全相同, 零收益零风险**。

### 与既有 P0 项的关系 (协同)

`1CAT-PORT-BACKLOG.md:82` 的 P0 就是「KV dtype A/B (零代码): 长上下文口径 q8_0 KV -> f16 KV」, 依据是 1cat 1.3.0 验收表的 128K prefill +61% / 256K prefill +59% / 128K decode +11%。

=> **正确顺序是: 先做 P0 (切 f16 KV, 零代码), 再在 f16 口径上评估 T2-001。** 两件事共用同一次配置变更。如果先打补丁再切 KV, 会得到「补丁无效」的假结论。

### 范围界定 (避免误期待)

- **对 256K prefill / TTFT 无帮助**: prefill 时 `Q->ne[1]` 大, `1 * gqa_ratio_eff` 远超 2, 走 TILE/MMA (`fattn.cu:648-651`)。而 prefill 是我们当前公开口径里最大的一块 (HANDOFF: 256K prefill 370 t/s、TTFT 672 s)。本补丁对这一项**没有贡献**。
- **对投机解码验证步无帮助**: DFlash2 的 `Q->ne[1] = 8` -> `8*2 = 16 <= 16` -> TILE (`fattn.cu:649`)。这与第三方 README 的「MTP 上只 +1%, 所以 MTP 模式继续用 stock build」一致 (文档宣称)。
- **对权重带宽无帮助**: 只影响 KV 读取。

---

## 6. 风险清单

| 级别 | 风险 | 依据 | 缓解 |
|---|---|---|---|
| 中 | **短上下文掉并行度**。打包后每序列块数 24 -> 8; 在 `ntiles_KV` (= n_kv/256) 小的浅上下文, 总块数可能凑不满一波。第三方自报 1K 处 **-2.0%** (文档宣称, 未复现) | `fattn-common.cuh:1201-1203` 块数公式; `fattn-common.cuh:1181-1199` 波次效率搜索 | 只在长上下文口径启用; 或按 `K->ne[1]` 阈值门控 (补丁没做) |
| 中 | **占用率变化不可控**。`parallel_blocks` 由 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 现场测 (`fattn-common.cuh:1127-1129`), 打包改了寄存器/共享内存占用 => KV 切分数会变 | 同上 | 观察 `parallel_blocks`; 必要时对打包版本单独调 |
| 中 | **现有测试矩阵覆盖不到打包路径** (见 §7.2) | `tests/test-backend-ops.cpp:10719` 的 nr2 列表 `{1,4,8,12,16,20,32}` 里**没有 3 或 6** | 必须用真实模型 + f16 KV 做端到端门 |
| 低 | **ALiBi 行为与 tile/mma 不一致**。tile/mma 在 `ncols2 > 1` 时把 slope 强制为 1.0 (`fattn-tile.cuh:866`, `fattn-mma-f16.cuh:1894,1941`), 而本补丁是逐头算真 slope。我们的模型 max_bias=0, `get_alibi_slope` 直接返回 1.0 (`common.cuh:969-974`) => **对我们完全惰性** | 三处 file:line 如上 | 无 (对本模型) |
| 低 | **sink 从循环外移到循环内**, 每 `j0` 迭代多读一次 global | 补丁 hunk 10/11 | sinks 只有 n_heads 个 float, 必在 L1/L2; 我们模型没有 sinks |
| 低 | **编译时间**: F16/BF16 的 4 个类型对 x D(64,128,256) x softcap(0/1) = 24 个额外 kernel 实例化 | `template-instances/fattn-vec-instance-*.cu` | 可接受 |
| 低 | **D=64/128 的打包实例也会被编出来**, 但只有 D=256 会被走到 (C7) | `fattn-vec.cuh:587-609` | 无 |
| 低 | **`dst_meta` 重写是行为保持的重构**, 但确实换了写法 (运行期 `KQ_max[tid]` -> 展开的编译期索引) | 补丁 hunk 14 | 门 2 覆盖 (`parallel_blocks > 1` 的长上下文才会走到) |
| 未验证 | 补丁自报的寄存器/溢出数字 (`~252-255 regs`、`30 STL + 17 LDL`) | 补丁注释 | 上机后用 `--ptxas-options=-v` 或 cuobjdump 核对 |
| 未验证 | 补丁自报的 26.37 -> 8.59 GB/token 与 16.45 -> 23.83 t/s | patch header + patches/README | 见 §8 复现口径 |

**算术自洽性检查 (本次实算)**: 128K 上下文, 16 层全注意力, 4 个 KV 头, D=256, K+V, 2 B/元素:
`2 x 131072 x 16 x 4 x 256 x 2 = 8589934592 B = 8.59 GB`。
=> 第三方的 "8.59 GB/token" 恰好等于 **KV cache 本身的大小**, 即"每个字节只读一次"的理论下限。
=> `26.37 / 8.59 = 3.07`, 与块数比 `24/8 = 3` 吻合。**这两个数字与代码机制自洽** (但绝对值仍是第三方实测, 我们未复现)。

---

## 7. 验证方法 (可执行门)

### 7.0 环境与红线

- 编译/测试只在 AC922 (`ssh -o BatchMode=yes root@192.168.50.235`), 按 `AGENTS.md` §3 的 gcc-toolset-12 + CUDA 12.4 配方; 源码树 `/root/llm/test/v100-opt/llama.cpp`。
- 两侧必须**同源 A/B** (同一 build dir、同一 CMake 参数), 用整目录 + `LD_LIBRARY_PATH` 切换, 并 `md5sum` 两侧 `libggml-cuda.so` **和** `libllama-common.so` 自证有效 (`AGENTS.md` §4.2)。
- 每配置 >= 2 次并报离散度; 正式数字带 `drop_caches`。

### 7.1 门 0: 编译

```
git apply patches/0001-t2-001-gqa-packing-sm70.patch   # 预期 16 个 hunk, #9-#16 offset -2
```
然后按 AC922 配方重编。**重点看 nvcc 是否报 `ncols2` 相关的模板/static_assert 错误**; 预期没有 (理由见 §3「不需要改的地方」)。
建议同时记 `--ptxas-options=-v` 的寄存器数, 用于核对 §6 的「未验证」项。

### 7.2 门 1: 后端算子测试 (回归 + 已知覆盖缺口)

```
/root/llm/test/v100-opt/llama.cpp/build-nccl/bin/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0
```

**这个门能证明什么、不能证明什么 (本次实测核实):**

- 能: 证明 ncols2=1 的所有既有形状 (含 D=256 F16 的 GQA=4, 走 TILE) 没有回归。
- **不能: 证明打包路径正确。** 理由 (**实测读码**): `tests/test-backend-ops.cpp:10719` 的 nr2 (即 gqa_ratio) 取值是 `{1, 4, 8, 12, 16, 20, 32}`, **没有 3 也没有 6**; 唯一能被 3 整除的 nr2=12 被 `:10721` 限制为 `hsk == 128`, 而 `:10732` 又限制非 F16 KV 只能用于 hsk 64/72 => 即使跑到 nr2=12/hsk=128/F16, 在 V100 上 `gqa_ratio_eff = 8`, `1*8 = 8 > 2` (`fattn.cu:645`) 会把它判给 **TILE**, 根本进不了 vec 核。
  => **stock 测试矩阵在 Volta 上完全覆盖不到 `ncols2=3`。**

**补一个能覆盖的用例 (由主代理决定是否做)**: 最小复现形状就是我们的几何 —— `hsk=hsv=256, nh(KV 头)=4, nr2=6, nb(Q 行)=1, kv=512, mask=true, max_bias=0, type_KV=F16`, 即往 `tests/test-backend-ops.cpp:10719` 的 nr2 列表加 `3, 6` (或加一条显式 `test_cases.emplace_back(...)`)。这属于**改文件不是新增 `tests/*` 文件**, 但仍应先问用户 (`AGENTS.md` 红线)。加完后预期: 打补丁前该 shape 走 VEC-ncols2=1, 打补丁后走 VEC-ncols2=3, 两者都应与 CPU 参考一致。

### 7.3 门 2 (硬门): greedy temp 0 逐位一致

用户指定的正确性门。要求: `-ctk f16 -ctv f16`(C1), 单卡, 同一 prompt, `--temp 0`, 固定 seed, 两边生成完全相同的 token 序列 => `sha256` 相同。**注意必须用 f16 KV 才有意义**: 用 q8_0 KV 时两边的 kernel 是同一份, 这个门恒过, 是假门。

补充建议: 除了 sha256, 再比一次 **logits 层面的 KLD / top-1** (补丁自报 `KLD = 0.000000`、`Same Top-1 = 100%`, 属文档宣称), 因为 temp 0 + 短序列可能掩盖极小的数值差。

### 7.4 门 3: AL 不变

用生产口径 (Q8_0 权重 + DFlash2 n=7 + TP + NCCL + P2P) 跑 `>= 3 个 prompt + 固定 seed`, 比对 **AL (接受长度)**: 打包只改 attention 的归约顺序, 不应改变 logits, 因此 AL 必须逐位不变 (`AGENTS.md` §4.4)。同时报 `t/s`、`AL`、`ms/轮` 三件套。

⚠️ 此处有个**预期落空点要先说清**: DFlash2 验证步的 `Q->ne[1] = 8` 走 TILE, 不受补丁影响 (§5)。所以在带投机的口径上, 补丁**只影响每轮里那次单 token 的 target decode**, 增益会被摊薄。这也正是第三方 README 说 "MTP 上只 +1%" 的原因 (文档宣称)。

### 7.5 门 4: 性能与流量侧证

- `llama-bench` 单卡、无投机 (`--spec-type none`)、f16 KV、`-p 512 -n 128`, 扫 1K/8K/32K/64K/128K; 预期与第三方同形: 1K 略降, 32K 起转正, 128K 最大 (**数字为第三方宣称, 我们要自己测**)。
- 若想验证 "KV 流量下降 3x" 这个机制而非只看 t/s: 用 `test-backend-ops perf -o FLASH_ATTN_EXT` (`build-nccl/bin/`, 见 `AGENTS.md` §4.10) 在固定 shape (D=256, 24/4 头, n_kv=131072) 下对比同一 kernel 的耗时; 这是唯一能绕开 ncu/nsys 的通道 (ncu/nsys 对 18-29 GB 模型不可用, 见 `AGENTS.md` §1)。

### 7.6 门 5 (可选): 短上下文回归

在 1K 上下文确认掉幅。第三方自报 -2.0%; 若我们测到更大, 说明块数 24->8 的并行度损失在我们 6 卡/TP 切分下更严重, 需要加阈值门控。

---

## 8. 不可用时的替代做法

按「先做便宜的、先做与 T2-001 共路的」排序:

**A. (首选, 零代码, 与 T2-001 同路) 先做 P0: q8_0 KV -> f16 KV。**
依据 `1CAT-PORT-BACKLOG.md:82` (P0, 零代码): 1cat 1.3.0 验收表给出 128K prefill +61% / 256K prefill +59% / 128K decode +11%。这一步本身就有独立收益, 而且是 C1 的前置。
代价: f16 KV 显存是 q8_0 的 1.88 倍 (128K 时 4.56 GB -> 8.59 GB, 按 §6 的算术); 需要确认剩余显存够 (4 卡时我们 Q8_0 权重 7.25 GB/卡, 见 `AGENTS.md` §3)。
风险: 必须先确认 f16 KV 在我们口径下不劣化 (它同时会改变 FA 选核: `fattn.cu:616-619` 的未量化分支条件与量化分支不同)。**建议: 把 "切 f16 KV" 本身当成一个独立 A/B 实验, 不要和打补丁混在一起。**

**B. 若必须保留 q8_0 KV: 本补丁无效, 需要另写量化版打包。**
难点已定位: 量化路径的 `nthreads_KQ_q` 已经是 32 (`fattn-vec.cuh:82-83`), **没有 "摊到更多线程" 的余量**; 3 列累加器 (`VKQ[3][4]` vs `[1][4]` = +8 个 32-bit 寄存器) 加上 `Q_i32[3][2] / Q_ds[3][2]` (`:166,167`) 就是溢出的来源 (补丁注释自报 `30 STL + 17 LDL`)。
可能的出路 (均**未验证**, 成本明显高于 A): 让 3 列**串行**复用同一份 K tile, 而不是并行 3 列累加 —— 这样不增加列数, 但需要在同一个 block 里保存 3 个头各自的 `KQ_max/KQ_sum` 与输出累加器, 收益和复杂度都要单独评估。**不建议在 A 之前做。**

**C. 若目标是 256K prefill / TTFT: 本补丁无关, 不要指望它。**
prefill 走 TILE/MMA (`fattn.cu:648-651`), 而 TILE 核已有自己的 GQA 优化 (`fattn-tile.cuh:1253` `use_gqa_opt`)。prefill 要另找方向 (HANDOFF §1「真正待查」第 1 项)。

**D. 若只想省事: 完全不打补丁。**
理由: 在 q8_0 KV 下它是严格 no-op; 只有在切到 f16 KV 之后它才可能带来长上下文 decode 收益 (第三方宣称 128K +44.9%), 但同期会带来浅上下文 -2.0% 的代价和一次额外的验证成本。**决策点应放在 "f16 KV A/B 结果出来之后"。**

**E. 若 f16 KV 不可行 (显存不够) 但需要长上下文 decode 提速**: 
退回到 "提高共享同一 KV 头的 6 个 block 的时间局部性" 这条零代码路线 (L2 已经吃到一部分: 26.37 GB 而非朴素 51.5 GB, 见 §6 算术), 例如调整 KV 切分/块调度让同 KV 头的块落在同一波次。**未验证, 仅作为方向记录。**

---

## 9. 附录 A: 本地证据索引 (file:line)

全部相对 `F:\vllm+llama.cpp\llama.cpp\`:

| 主题 | 位置 |
|---|---|
| vec 核 ncols2 写死为 1 | `ggml/src/ggml-cuda/fattn-vec.cuh:541` |
| vec 核模板参数 (原始) | `ggml/src/ggml-cuda/fattn-vec.cuh:19` |
| ic0 / sequence / head 拆分 (原始) | `ggml/src/ggml-cuda/fattn-vec.cuh:104,106,107` |
| gqa_ratio 定义 (device 侧) | `ggml/src/ggml-cuda/fattn-vec.cuh:108` |
| Q_reg / VKQ 列数组 | `ggml/src/ggml-cuda/fattn-vec.cuh:125,128,162,164` |
| KQ_max / KQ_sum 等 | `ggml/src/ggml-cuda/fattn-vec.cuh:152,153,279,281` |
| nthreads_KQ_q / nthreads_V_q = 32 (D=256 量化) | `ggml/src/ggml-cuda/fattn-vec.cuh:82,83` |
| 量化 V 的 32 线程布局已在跑 | `ggml/src/ggml-cuda/fattn-vec.cuh:324,333,344,351` |
| 选核器入口 | `ggml/src/ggml-cuda/fattn.cu:517` |
| can_use_vector_kernel | `ggml/src/ggml-cuda/fattn.cu:611` |
| ncols2_max / gqa_ratio_eff | `ggml/src/ggml-cuda/fattn.cu:638-642` |
| Volta 分流 (VEC/TILE/MMA) | `ggml/src/ggml-cuda/fattn.cu:644-652` |
| MMA 的 2 的幂 ncols2 阶梯 (Volta) | `ggml/src/ggml-cuda/fattn.cu:200-222` |
| vec case 取用 (新派发) | `ggml/src/ggml-cuda/fattn.cu:400,412,477,486` |
| launch_fattn 模板与网格 | `ggml/src/ggml-cuda/fattn-common.cuh:975,1090-1093,1201-1203` |
| occupancy 决定 parallel_blocks | `ggml/src/ggml-cuda/fattn-common.cuh:1127-1129` |
| FATTN_KQ_STRIDE = 256 | `ggml/src/ggml-cuda/fattn-common.cuh:9` |
| vec_dot_fattn_vec_KQ_f16 (nthreads 泛化) | `ggml/src/ggml-cuda/fattn-common.cuh:87-115` |
| get_vec_dot_KQ | `ggml/src/ggml-cuda/fattn-common.cuh:620-639` |
| cpy_nb = 16 (Volta+) | `ggml/src/ggml-cuda/common.cuh:399-409` |
| volta_mma_available | `ggml/src/ggml-cuda/common.cuh:360-362` |
| ggml_cuda_highest_compiled_arch | `ggml/src/ggml-cuda/common.cuh:159-177` |
| get_alibi_slope (max_bias<=0 -> 1.0) | `ggml/src/ggml-cuda/common.cuh:969-974` |
| tile 核的 gqa_opt | `ggml/src/ggml-cuda/fattn-tile.cuh:1253` |
| tile/mma 里 ncols2>1 时 slope 强制 1.0 | `ggml/src/ggml-cuda/fattn-tile.cuh:866`; `fattn-mma-f16.cuh:1894,1941` |
| vec case 实例化宏与清单 | `ggml/src/ggml-cuda/fattn-vec.cuh:574-609` |
| 生成器 | `ggml/src/ggml-cuda/template-instances/generate_cu_files.py:24-26` |
| 每个类型对的实例化 TU | `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu:5-7` (及其他) |
| FA 测试用例枚举 (覆盖缺口) | `tests/test-backend-ops.cpp:10701-10754`, 特别是 `:10719,10721,10727,10732` |

## 10. 附录 B: 外部证据索引

| 内容 | URL |
|---|---|
| 补丁 (16534 B) | https://raw.githubusercontent.com/jackinthebox52/qwen38-v100-serve/main/patches/0001-t2-001-gqa-packing-sm70.patch |
| patches/README (机制描述, 与代码不符见 §2.2) | https://raw.githubusercontent.com/jackinthebox52/qwen38-v100-serve/main/patches/README.md |
| 仓库 README (性能表) | https://raw.githubusercontent.com/jackinthebox52/qwen38-v100-serve/main/README.md |
| serve.sh (C1 证据: `-ctk f16 -ctv f16`) | https://raw.githubusercontent.com/jackinthebox52/qwen38-v100-serve/main/serve.sh |
| 仓库树 API | https://api.github.com/repos/jackinthebox52/qwen38-v100-serve/git/trees/main?recursive=1 |
| 上游 b10793 `fattn-vec.cuh` | https://raw.githubusercontent.com/ggml-org/llama.cpp/b10793/ggml/src/ggml-cuda/fattn-vec.cuh |
| 上游 b11053 `fattn-vec.cuh` | https://raw.githubusercontent.com/ggml-org/llama.cpp/b11053/ggml/src/ggml-cuda/fattn-vec.cuh |
| 上游 b10793 `fattn.cu` | https://raw.githubusercontent.com/ggml-org/llama.cpp/b10793/ggml/src/ggml-cuda/fattn.cu |
| 上游 b11053 `fattn.cu` | https://raw.githubusercontent.com/ggml-org/llama.cpp/b11053/ggml/src/ggml-cuda/fattn.cu |

## 11. 附录 C: 文档宣称 vs 实测 - 一览

| 说法 | 来源 | 判定 |
|---|---|---|
| vec kernel 按 2 的幂打包到 ncols2=2, 故剩 3x 冗余 | 仓库 README | **错**。vec 核 ncols2 恒为 1 (`fattn-vec.cuh:541`); "2" 是 selector 的 `gqa_ratio_eff`。真实是 6 个块/KV 头 |
| 每 KV 头被读 3 次 | patches/README | **错**(同因)。真实是 6 次 -> 打包后 2 次 |
| 消掉因子 3 | patch header | **对** (块数 24 -> 8) |
| KV 流量 26.37 -> 8.59 GB/token | patch header | 算术**自洽** (8.59 GB = 128K KV cache 精确大小); 绝对值**未复现** |
| 128K decode 16.45 -> 23.83 t/s (+44.86%) | patch header / patches/README | **未复现**。注意 README 表里写 16.49, patches/README 写 16.45, 仓库内部不一致 |
| 1K 处 -2.0% | patches/README | **未复现**; 机制解释成立 (块数少 3 倍) |
| KLD = 0.000000, top-1 100% | patch header | **未复现** |
| 量化路径 ~252-255 寄存器, 打包会 30 STL + 17 LDL 溢出 | patch 注释 | **未复现** |
| MTP 上只 +1%, 故 MTP 用 stock build | README + serve.sh | **机制吻合** (验证步走 TILE, 见 §5); 数字未复现 |
| 补丁只改 1 文件 +71/-21 | patch header | **对** (实测: 610 -> 661 行) |
| 补丁 base = b10793 (d230ddd) | patches/README | **对**; 且能干净落到 b11053 (实测 offset -2) |

---

## 12. 一句话给主代理

补丁已拿到、已确认可无冲突 apply 到 b11053、逐处改动已定位到行; **但它是「f16 KV 之后才存在」的优化, 在 q8_0 KV 下编译期即被丢弃**, 且**只对单 token decode 生效、对 prefill/投机验证步无效**。建议动作顺序: (1) 先独立做 P0 的 f16 KV A/B; (2) 若 f16 口径成立, 再打补丁并跑 §7.3/7.4 两个硬门; (3) 不要为它动 `fattn.cu`。
