# 查漏补缺：外部 V100 参考项目分析（2026-09-21）

> 来源：用户提供的 6 个仓库，克隆在 `F:\vllm+llama.cpp\v100-refs\`。
> 本节是**中期结论**，由子代理调研得出，主线已核对关键数字的出处。

## 1. ⚠️ 先纠正一个诱惑：「单卡 V100 219 tok/s」不是可比的数字

两个仓库各有一个 219，但口径完全不同：

| 来源 | 数字 | 真实口径 |
|---|---|---|
| `v100-skinny` README:34 | 219.1 tok/s | **4×V100-SXM2-16GB TP4 + k=7 投机验证**，26.9 ms/轮，5.89 tok/轮 |
| `ninfer-v100` README:23,54 / docs/v100.md:55 | 218.98 tok/s | **单卡 V100-PCIE-32GB**，但 K=1 投机跑在**合成续写语料**上、99.2% 接受率；
| | | README 自己标注 *synthetic-corpus ceiling, not a general-generation rate*；artifact 本体 23.72 GB，
| | | 按 900 GB/s 上限物理上不可能是普通 decode => 该语料触发了 context-lookup 长验证路径 |

**ninfer 真正可比的单卡数字**（`docs/v100.md:111-117`，varied-context sweep）：

| 配置 | tok/s |
|---|---:|
| 2K 无投机 | 29.28 |
| 2K + MTP K=3 | 67.66 |
| **8K + MTP K=3** | **92.74** |
| 32K | 55.04 |
| 150K | 43.63 |

⚠️ **但这一组数字本身就足够刺眼**：**单张 V100、8K、MTP K=3 = 92.74 tok/s**，
而我们 **3 张 V100、8K、DFlash2 = 99.97 tok/s**。按每卡效率算，我们落后约 2.8 倍。
（口径需再确认：模型同为 Qwen3.8-27B 系；MTP K=3 与 DFlash2 n_max=7 不是同一件事，但量级差摆在这里。）

## 2. 低比特权重：真实存在，但**不是那条大鱼**

两个仓库都**没有绕开 tensor core**，都用 `mma.sync.m8n8k4`：
- `v100-skinny kernels/skinny_kernels.cu:1145`：`MMA_8N8K4` -> `mma.sync.aligned.m8n8k4.row.col.f16.f16.f32`
  做法：**四个 quadpair 全部铺在 N 维、共享同一块 A 激活（QPN）**；4-bit 权重**保持压缩穿过 HBM**，
  在寄存器里直接解包成 mma 需要的 B fragment（nibble 预交织、无 shuffle、主循环无 smem、56 reg、零 spill）。
- `ninfer src/ops/common/volta_mma.cuh:111` `volta_mma_qp_n` 完整复刻了这一套；
  `nvfp4_volta_qpn_gemm.cuh` 头注释直接写 *This is v100-skinny's skinny_nvfp4_qpn2 ... Copy the decoder and scale-cadence choices*。

**实测有效带宽**（作者自测、同 harness 同一次 sitting）：

| 项 | 值 | 占天花板 |
|---|---:|---:|
| 读天花板 | 879 GB/s | - |
| NVFP4 (0.5625 B/权重) M=1 | 679.5 GB/s | 77% |
| NVFP4 M=8 | 619.8 GB/s | 71% |
| lm_head M=1 | 842.9 GB/s | 96% |
| FP8 (1 B/权重) M=1-4 | 718.6-720.5 GB/s | 82% |
| ninfer W8 GEMV T=1 | 752 / 该卡实测 ~794 GB/s | 95% |

### 对我们「HMMA 天花板 1.36x」结论的修正
那条是按 **Q8_0 自身 n=1 屋顶 757 GB/s 的 73%** 推的（`HANDOFF.md:443`，形状还是合成的 m=4096 k=14336，AUDIT E5 已标注）。
**它没错，但口径偏保守**：换成 4-bit 存储后，同一张卡的有效**权重**吞吐是 `679.5/0.5625 = 1208 G 权重/s`，
而 Q8_0 屋顶是 `757/1.0625 = 712 G 权重/s` => **同一张卡「换格式」的权重吞吐上限约 1.70x，不是 1.36x**。

**但端到端上限仍然很小**：权重流只占我们 12.1 ms / 55.5 ms => **低比特权重最多值 +10~13%，不是 2x**。
=> **不要把这条当成主线。**

## 3. ★ 真正的大鱼：两个参考实现都把**主机侧开销压到接近零**

| 参考 | 实测 |
|---|---|
| `v100-skinny`（4 卡） | target verify(G0->G1) **17.48 ms** + drafter 5.85 ms => **整轮 25 ms** |
| `ninfer`（**单卡**，2K 无投机） | **整轮 34.2 ms** |
| **我们（3 卡）** | target 整步 32.3-34.3 ms + draft 13.6 + selector 4.3 => **55.5 ms/轮** |

**我们 3 张卡的 target 整步，约等于 ninfer 1 张卡的整轮。** 低比特权重最多只能解释其中 22%。

两个实现的做法：
- vLLM 侧：**persistent-metadata 投机轮 + 固定形状 CUDA graph**；
- ninfer 侧（架构级规则，`docs/maintainer/engine-architecture.md:489`）：
  *CUDA Graph 按「合法的 exact-B topology」建立，**request identity 与 page IDs 是稳定输入数据，不是 graph key***。

=> **这正对着我们 53% 的 meta 后端主机循环**（每轮 29.4 ms），也印证了我们已闭合的那条链
（图重建 -> uid 重发 -> 属性抖动 -> direct）。**外部先例表明这条路是对的，而且有人已经走到位了。**

## 5. ★★ attention 与服务层线（中期，证据级）

### 5.1 最可能直接解决我们长上下文问题的一条：**GQA read-once**
`sglang-V100` fork 内 `python/sglang/srt/layers/attention/tilelang_fa_v100/`：
- **一个 CTA 吃掉一个 KV 头的全部 6 个 Q 头 => K/V 只读一次**
- K 轴 split-KV：decode 目标 **160 个 CTA**；prefill 尾块 64/32/16/8/4/2 路
- 1 字节 E5M2 KV + 移位转换
- 作者自测：把 **128K decode 从 30.04 tok/s 救回 200K 49.58 tok/s**

**为什么这条对我们特别关键**：我们的模型 **24 Q 头 / 4 KV 头（GQA=6）**，而我们的长上下文实测是
**有效 KV 带宽约 105 GB/s、roofline 800 GB/s**（256K decode 34 t/s）。
若我们的 FA 内核是「每个 Q 头各读一遍同一份 K/V」（llama.cpp 常见组织方式），则 KV 读流量是最小值的 **6 倍**。
**105 x 6 = 630 GB/s，正好接近 roofline** —— 这条线索可以解释我们长上下文的大部分差距。
对照：`flash-attention-v100` 自己**没有**做 GQA read-once（`template.h:58,73` 用 `kv_head_idx = head_idx/(H_Q/H_K)`，6 个 Q 头各读一遍），
而且它的 decode **明确不支持 split-KV**（`fused_mha_forward_kvcache.cu:462` 有硬 `TORCH_CHECK`：`num_splits > 1 not supported now`）。
=> **两个仓库正好形成对照：谁做了 GQA read-once 谁就把长上下文救回来了。**

### 5.2 与我们主机侧问题直接对应的配置纪律：**只捕获单形状 CUDA 图**
`sglang-V100` 的服务参数：`--max-running-requests 1` + **`--cuda-graph-bs 1 --cuda-graph-max-bs 1`**（只捕获单形状图）
+ `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`（规划流与计算流重叠）；KV 量化用 `--kv-cache-dtype fp8_e5m2`。
=> 这正是我们缺的**静态形状纪律**，也解释了为什么他们的图不会像我们这样每轮属性抖动 17,000 次。

### 5.3 D=256 在 Volta 上的两个硬约束（与我们已做的 FA 工作对照）
- `flashinfer-sm70.patch` 新增了 D=256 的 kernel spec：`head_size=256`、`warps_m=1`/`warps_n=2`、**`share_smem_k_v=True`**、`loop_step=16`；
  原文注释：*share_smem=True required - D=256 overflows Volta's 96KB SMEM without sharing*。
- 同 patch 把继承自 Turing 的 **`warps_n=8` 全部改成 `warps_n=2`**（针对 sm70 的 hmma884 重新调参）。
- `flash-attention-v100` 侧同样的结论：D=256 时 smem 合计约 **88.1 KB => V100 上只能 1 CTA/SM**，
  K/V 共用一块 smem 是能放下的**唯一原因**（`include/forward.h:42-61` 的 union 布局）。
=> **我们应核对自己的 FA 在 D=256 下的 `warps_n` 与 `share_smem_k_v` 是否为 sm70 调优值。**

### 5.4 其他可借的点
- `marlin-v100-*.patch`：外部仓库 `zhinianqin/marlin_v100` 的 MoE **W4A16** 内核；并新增 `csrc/sm70_bf16_compat.h`
  给 sm70 补 `__bfloat1622float2` / `__hfma2` 等在 CUDA 12.x 下对 sm_80 以下屏蔽的向量 intrinsic。
  （对我们意义有限：我们的模型是**稠密**的，不是 MoE。）
- `flash-attention-v100` 的一个重要否定性结论（`utils/docs/volta.md:139,143`）：
  **sm_70 上 `nvcuda::wmma` 只有 m16n16k16，且 `ldmatrix` 要 sm_75 => 拿 wmma 直接写 Volta attention 走不通**，必须手写 PTX。
  这与我们早前「自写 WMMA 原型只有 90 GB/s」的实测一致。

## 4. 待深挖（子代理正在做）
- `v100-skinny fork_patches/gdn_attn.py`：**GDN 投机状态的快速元数据构建**，作者自测 **-1.4 ms/step**，
  消掉 **21 次 device sync + 约 70 次 copy** —— 我们的 direct 抖动源正是 **GDN 递归状态视图**，这条高度相关。
- `ninfer` 的 **ReplaySSM**：不做 T 份 state trajectory，只存 raw record 再 fold。
- 两者的图捕获 / 内存池细节。
