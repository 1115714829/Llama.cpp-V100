# M5 - Volta (SM70) D=256 FlashAttention 配置表实测与补丁

> 目标：为 llama.cpp (base b11053) 在 V100 上找到比现状更快的 `ggml_cuda_fattn_mma_get_config_volta()` 参数。
> 结论：**有**，而且是数量级的单点收益（+53% ~ +113%），根因是 **Q_in_reg=true 在该形状上把寄存器用爆（255 reg + 2 KB spill）**。
> 测量者：M5 子代理；日期 2026-09-20/21；机器 AC922 (6x V100-SXM2-16GB)，GPU3 独占。

---

## 0. TL;DR

| 形状（真实几何） | 现状 TFLOPS | 新配置 TFLOPS | 提升 |
|---|---:|---:|---:|
| kv=65536, nb=512（prefill 尾块） | 19.39 | **29.73 - 30.39** | **+53% ~ +57%** |
| kv=8192, nb=64 | 15.53 | **30.39** | **+96%** |
| kv=8192, nb=16 | 8.62 | **18.37** | **+113%** |
| kv=8192, nb=8  /  kv=131072, nb=8 | 10.02 / 10.16 | 10.06 / 10.15 | **0%（配置表不参与，见 §2 注意）** |

改动只有一行表的语义：把 D=256 的 volta 行从 Ampere 回落值改成 **`Q_in_reg=false`**（其余参数不变）。
补丁：`F:\vllm+llama.cpp\.dsh\tmp\m5-fattn-volta.patch`（4 行，只动 `ggml/src/ggml-cuda/fattn-mma-f16.cuh`）。

**适用范围警告**：`nb<=8` 时 Volta 走 **TILE kernel**（判据 `Q->ne[1]*gqa_ratio_eff <= 16`，本模型 gqa_ratio_eff=2），
配置表不参与 ⇒ **8-token 小批量验证口径不受影响、也不受益**；受益的是 **prefill（ubatch >= 9 token）**，也就是 256K 场景里 TTFT 那一大块。

---

## 1. 测量条件（读数字前必看）

- 量具：`LD_LIBRARY_PATH=/root/m5-lib /root/m5-lib/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 -p <regex>`
  （`test-backend-ops` 是我用**私有**副本编的，为了加 nb=8/16/64 形状；见 §5）
- 硬件/环境：`CUDA_VISIBLE_DEVICES=3`，V100-SXM2-16GB，跑前 GPU 0 MiB / 0% util；驱动 550.54.15，CUDA 12.4。
- 编译：`--generate-code=arch=compute_70,code=[sm_70]`，`-O3 -DNDEBUG -std=c++17 -use_fast_math -extended-lambda`（与 build-instr 的 flags.make 一致）。
- 形状：`hsk=hsv=256, nh=4, nr23=[6,1]`（24 Q 头 / 4 KV 头，GQA=6）, causal mask, prec=f32, type_K=type_V=f16, kv_view=1。
  即 Qwen3.8-27B 的真实几何。
- **每个点 3 次**（3 个独立进程）。报告用 median，括号里给 max。
- **噪声**：同一 build 内 rep1 系统性比 rep2/3 低约 1-3%（首跑冷时钟）；不同 build 之间同配置 max 相差 ~2%。
  ⇒ **小于 3% 的差异不要当成真实结论**（本报告里 k64 的 +3% 就在这个边缘）。
- TFLOPS 是 `op_flops` 名义值（`2*nh*nr2*nb*(hsk+hsv)*kv`，按 dense 计）。
  ⚠ 这个 perf 用例是 **causal mask 但按 dense 工作量执行**（`nb=512` 时 `Q->ne[1] >= 1024` 不成立，KV_max 扫描被跳过）。
  对我们的场景正好合适（最后一个 prefill 块必须看全 131072 个 KV），但对"块首 prefill"偏悲观。

---

## 2. 候选表（第一批：只改一个旋钮，只测 ncols=64 行；kv=65536 / kv=4096，nb=512）

| tag | (nthreads,occ,nbatch_fa,K2,V2,combine,nstages,Q_in_reg) | kv=65536 med (max) | kv=4096 med (max) | 相对基线 |
|---|---|---:|---:|---:|
| **base**（现状 = Ampere 回落行） | 128,2,32,128,128,128,2,**true** | 19.74 (19.77) | 19.22 (19.31) | 1.00 |
| **qnoreg（最佳）** | 128,2,32,128,128,128,2,**false** | **30.07 (30.39)** | **29.20 (29.20)** | **1.52x** |
| k64 | 128,2,32,**64**,128,128,2,true | 20.34 (20.40) | 19.85 (19.87) | 1.03x（噪声边缘） |
| fa64 | 128,2,**64**,128,128,128,2,true | 19.99 (20.13) | 21.20 (21.64) | 1.01x / 1.10x |
| st1 | 128,2,32,128,128,128,**1**,true | 19.69 (19.73) | 19.15 (19.22) | 1.00x（**完全无效**） |
| v64 | 128,2,32,128,**64**,128,2,true | 17.22 (17.37) | 16.99 (17.01) | **0.87x（更慢）** |
| c64 | 128,2,32,128,128,**64**,2,true | 14.50 (15.10) | 16.22 (17.03) | **0.73x（更慢）** |
| thr256 | **256**,1,32,128,128,128,2,true | `CUDA error: invalid argument` | 同 | **不可用（起不来）** |
| thr256o2 | **256**,2,32,128,128,128,2,true | `CUDA error: invalid argument` | 同 | **不可用（起不来）** |

绝对时间参考：kv=65536/nb=512 时 824.63 GFLOP/次；19.39 TFLOPS ⇒ 42.5 ms，30.07 TFLOPS ⇒ 27.4 ms。

## 3. ncols 覆盖 + 小批量（第二批探针；行 = 256/256/{8,16,32,64}，全部 Q_in_reg=false）

| 形状 | 实际命中的行 / kernel | base | q3（加 16/32/64 行） | q4（加 8/16/32/64 行） | 提升 |
|---|---|---:|---:|---:|---:|
| kv=8192, nb=8 | **TILE kernel**（配置表不参与） | 10.02 | 10.07 | 10.06 | 0% |
| kv=8192, nb=16 | ncols=32 行 | 8.62 | 18.37 | 18.24 | **+113%** |
| kv=8192, nb=64 | ncols=64 行 | 15.53 | 30.30 | 30.39 | **+96%** |
| kv=131072, nb=8 | **TILE kernel** | 10.16 | 10.14 | 10.15 | 0% |
| kv=65536, nb=512 | ncols=64 行 | 19.39 | 29.99 | 29.73 | **+53%** |

（q3 与 q4 在运行期完全等价 —— ncols=8 在 Volta 上不可达，见 §4；q4 用来证明"8 行也能编过"。）

## 4. 为什么有效（ptxas 证据）+ 分发兼容性

### 4.1 根因：寄存器溢出

`ptxas -v`（sm_70，`flash_attn_ext_f16<DKQ=256,DV=256,ncols1=32,ncols2=2,softcap=false,V_is_K_view=false,sparse=false>`）：

| 配置 | registers | stack frame | spill stores | spill loads |
|---|---:|---:|---:|---:|
| `Q_in_reg=true`（现状） | **255** | 480 B | **2060 B** | **1888 B** |
| `Q_in_reg=false`（新） | 254 | 48 B | **0 B** | **0 B** |

Q 常驻寄存器时，ncols=64/D=256 的 Q 片（每线程 64 half2 = 64 个 32-bit 寄存器）叠加 VKQ/KQ 累加器，
把 NVCC 顶到 255 上限并产生 **每次迭代 ~2 KB 的 spill 流量**（spill 落在 local memory = 走 L1/L2/HBM）。
把 Q 放到 smem（`Q_in_reg=false`）后 spill 归零，吞吐 19.4 -> 30 TFLOPS。

### 4.2 分发兼容性（回答"该元组是否对所有 ncols 已实例化"）

- `(nthreads=128, occupancy=2, nbatch_fa=32, nbatch_K2=128, nbatch_V2=128, nbatch_combine=128, nstages=2, Q_in_reg=false)`
  **编译期对 ncols = 8/16/32/64 全部成立**（q4 四个行全部编过、链接过、跑过）；静态断言
  `nthreads%32==0 && <=512`、`nbatch_fa%32==0 && <=256`、`K2%4==0 && <=512`、`V2%4==0 && <=256`、`combine%4==0 && <=128`
  以及 kernel 内部的 `DV % (2*nbatch_V2)==0`、`(DV/2) % nbatch_combine == 0` 全部满足。
- **Volta 上 D=256 运行期可达的 ncols 只有 {16, 32, 64}**：`switch_ncols1<...,ncols2>` 里
  `Q->ne[1] <= 8/ncols2`（→ ncols=8）那一条被 `turing_mma_available(cc)` 门控，V100 永远不进；
  最小落到 `Q->ne[1] <= 16/ncols2` ⇒ ncols=16。
  （但该分支的模板**仍会被实例化**，所以 ncols=8 的行必须"能编译"——q4 已验证。）
- **nstages_target 在 Volta 是死参数**：host 侧 `cp_async_available(cc)==false` ⇒ 返回 0；device 侧
  `CP_ASYNC_AVAILABLE` 只对 `__CUDA_ARCH__ >= 800` 定义 ⇒ 也返回 0。两者一致（都为 0），
  所以 `nstages=2` 与 `nstages=1` 实测完全相同（表 st1 行 = 基线）。补丁里保留 2 只是"与 Ampere 回落行同值、最小改动"。
- **注意 nibble**：每一条 `(256,256,ncols)` 行必须**逐个补齐**，否则缺的那个 ncols 会静默落回 Ampere 行（Q_in_reg=true）——
  即同一个模型在不同 batch 下性能断层。所以补丁把 8/16/32/64 四行都写了。
- 运行期实测覆盖：ncols=32（nb=16）、ncols=64（nb=64/512），gqa_ratio=6 ⇒ ncols2=2。
  ncols=16/8 在当前模型不可达（ncols=8 全平台不可达），只验证了"能编译"。

### 4.3 和 1cat 的距离

- 1cat 同类核 causal 61.09 TFLOPS；我们改前 19.4（3.1x），改后 ~30（**2.0x**）。
- 这个形状本身是**访存受限**的（估算）：每个输出 tile（32 Q 列）要把自己那个 KV 头的整段 KV 读一遍，
  `ntiles_dst = 16 Q tile x 3 zt_gqa x 4 KV head = 192`，每 tile ~67 MB ⇒ **~12.9 GB/op**；
  V100 900 GB/s ⇒ 理论下限 ~14.3 ms ⇒ 名义上限约 **58 TFLOPS**。
  我们现在 27.4 ms，即 **~52% 的带宽上限**。想再往上基本要靠改 kernel/分发（提高 ncols1 复用），不是配置表能做的。

---

## 5. 未生效 / 变慢的候选及可能原因

| 候选 | 现象 | 可能原因 |
|---|---|---|
| `nbatch_combine=64` | **-25%**（14.5 / 16.2） | DV/2=128 被拆成 2 趟 combine，smem 里多一遍读改写；combine 阶段指令数翻倍，而 smem 变小带来的 occupancy 收益不足以抵消 |
| `nbatch_V2=64` | **-13%**（17.2 / 17.0） | V tile 只有半行，global->smem 的 16B 突发变短，且 V 的读取趟数翻倍 |
| `nstages_target=1` | **完全无变化** | 见 §4.2：Volta 无 cp.async，host/device 都返回 nstages=0，该参数根本不参与 |
| `nthreads=256`（occ 1 或 2） | `CUDA error: invalid argument`（launch 期） | (nthreads=256, ncols1=32) 组合的 warp 布局（np = nwarps/(ncols/cols_per_warp)）不满足 kernel 隐含约束；静态断言查不出来，只在 launch 时炸。**不要用 256** |
| `nbatch_K2=64` | +3%（噪声边缘） | K tile 半行加载，收益与额外趟数抵消 |
| `nbatch_fa=64` | +1% / +10%（kv 越短越明显） | KQ 累加器翻倍、KV 循环减半；长 kv 时访存主导，收益被吃掉 |
| 测 `nb=8`（8-token 验证口径） | **一点没变**（10.0） | `Q->ne[1]*gqa_ratio_eff = 16 <= 16` ⇒ 走 **TILE kernel**，MMA 配置表不参与。想让 8-token 口径受益必须动 `fattn.cu` 的分发判据（本次红线不允许） |

---

## 6. 我最没把握的地方

1. **只验证了 gqa_ratio=6 ⇒ ncols2=2**。同一 `(256,256,ncols)` 行还会被 ncols2=1/4/8 的分裂共享
   （例如 `gqa_ratio=8` 会走 ncols2=8、`gqa_ratio=4` 走 ncols2=4）。这些组合用的是同一行、同一 kernel 模板，
   但 (ncols1,ncols2) 不同 ⇒ warp 布局不同。**没有运行时验证**（需要别的模型几何才能触发）。
   nthreads=256 那个 "invalid argument" 说明"能编译 != 能跑"，所以这条风险是真的。
2. **ncols=16 / 8 行没有跑到**：本模型 nb<=8 被 TILE 拦掉，ncols=8 在 Volta 上不可达。
   它们只过了"能编译、能链接"。
3. **正确性覆盖不全**：`test-backend-ops test -p 'hsk=256,hsv=256,nh=4,nr23=[6,1]'` 覆盖了
   kv=512/4096/16384 (nb=512)、kv=4096 (nb=1/8/16)、kv=8192 (nb=64)，全部 OK；
   但没有覆盖 ncols=16/8 的 MMA 路径（走不到）和 ncols2≠2 的实例。
4. **噪声与仪器**：`test-backend-ops perf` 每个用例只测 1 秒左右，rep 间有 1-3% 抖动，且 **rep1 系统性偏低**；
   我报的是 median/max。另外所有测量都建立在"私有 nvcc+ld 流水线"上（见 §7），
   它复刻了 build-instr 的 flags，但不是 `cmake --build` 本身。
5. **KV_max 扫描**：nb=512 时 perf 用例不做 causal 剪枝，做的是 dense 工作量。
   真实 prefill 的**中间块**（Q 在序列中部）只需看一半 KV，收益会比这里更大；而 **kv 缓存没对齐 256** 的情况
   （`K->ne[1] % 256 != 0`）会关掉 GQA opt ⇒ 走 ncols2=1 路径，本次没测。
6. **没有端到端验证**：按主线要求没跑 llama-bench / 服务口径。prefill 端到端收益应该小于微基准的 +53%
   （FA 只是 prefill 的一部分），具体多少需要主线用 256K TTFT 口径复核。

---

## 7. 复现命令

### 7.1 应用补丁

```sh
cd /root/llm/test/v100-opt/llama.cpp          # 或任意 pristine b11053 检出
git apply -p1 < m5-fattn-volta.patch          # patch 只改 ggml/src/ggml-cuda/fattn-mma-f16.cuh
```

编译（关键：**必须让 template-instances 一起重编**）：

```sh
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
cmake --build build-instr --config Release -j82
cp -a build-instr/bin/. /root/libdir-instr/
```

⚠ **坑**：FA 的 device kernel 实例化**不在** `fattn.cu.o` 里，而在
`build-instr/.../CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_{8,16,32}-ncols2_2.cu.o`。
只重编 `fattn.cu` 会得到**完全无效**的 A/B（配置怎么改数字都一样）。
cmake 会正确处理（.d 依赖里有这个 .cuh），但手工流水线必须显式重编这些 TU。

### 7.2 量具（形状已在 perf 列表里：11309-11316 行）

```sh
export CUDA_VISIBLE_DEVICES=3
L=/root/libdir-instr
LD_LIBRARY_PATH=$L $L/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 \
  -p 'nr23=\[6,1\],kv=65536,nb=512'      # prefill 尾块
LD_LIBRARY_PATH=$L $L/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 \
  -p 'nr23=\[6,1\],kv=4096,nb=512'
```

小批量（nb=8/16/64）需要临时往 `make_test_cases_perf()` 里加形状
（我在私有副本 `/root/m5-src/test-backend-ops.cpp` 里加了 nb=8/16/64 @ kv=8192 与 nb=8 @ kv=131072，
**没有**写进共享树）；最终补丁不含 tests/ 改动。

### 7.3 正确性

```sh
LD_LIBRARY_PATH=$L $L/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0 \
  -p 'hsk=256,hsv=256,nh=4,nr23=\[6,1\]'
# => kv=512/4096/16384 nb=512, kv=4096 nb=1/8/16, kv=8192 nb=64 全部 OK（对 CPU 参考实现）
```

### 7.4 看寄存器/溢出

```sh
nvcc ... -Xptxas -v -c template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_2.cu
# 关注 flash_attn_ext_f16ILi256ELi256ELi32ELi2E... 的
#   "bytes spill stores / spill loads" 与 "Used N registers"
```

---

## 8. 交付物

| 文件 | 说明 |
|---|---|
| `F:\vllm+llama.cpp\.dsh\tmp\m5-fattn-volta.patch` | 只含 `ggml/src/ggml-cuda/fattn-mma-f16.cuh` 的 unified diff，基于 pristine HEAD（=b11053），插入 volta 函数开头；已用 `git apply -p1` 在 pristine 副本上验证通过（结果与预期文件 md5 一致） |
| 本文件 | 数据、结论、风险、复现路径 |
