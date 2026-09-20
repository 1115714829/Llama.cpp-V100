# FlashAttention-V100 (1Cat-vLLM SM70) — reference doc (archived verbatim from user attachment, 2026-09-20)

> 来源：用户提供的 1Cat-vLLM V100 优化说明文档（附件）。本文件原样存档，作为
> llama.cpp V100 专项优化的主要参考。配套分析见同目录 `vllm-vs-1cat-vllm-diff.md`、
> `core-changes.md`、`dflash2-llama-cpp-research.md`。

## 核心论点
我们不是"让 FlashAttention 在 V100 上能编译"，而是在为 Volta 重建数据流 (dataflow)。

FlashAttention 本质上是一个 IO 与调度问题：
- 减少 HBM 往返；
- 让 Q/K/V 和中间状态尽可能久地留在片上；
- 提高复用；
- 减少物化 (materialization)；
- 减少 barrier；
- 持续喂满 Tensor Core。

现代 FlashAttention 实现是围绕 Ampere / Hopper / 更新 GPU 设计的。

Tesla V100 是 SM70。它**没有**：
- Ampere `cp.async`；
- Turing/Ampere 风格、可供新 Tensor-Core kernel 用的数据通路；`ldmatrix`
- Hopper TMA；
- 原生 FP8 Tensor Core；
- Blackwell FP4 Tensor Core。

直接做兼容性移植可能能跑，但常常让 GPU 吃不饱 (underfed)。

正因如此，1Cat-vLLM 围绕 Volta **实际拥有**的能力重建执行路径。

## ⚙️ SM70 上的软件重建 Async / Matrix Feed
我们**不声称** V100 执行 `cp.async` / `ldmatrix`。

相反，1Cat-vLLM 用以下手段重建这些机制背后的设计目标：
- `LDG` / `STS` / `LDS`
- 寄存器预取 (register prefetch)
- 双缓冲 (double buffering)
- 共享内存 swizzle
- 显式 HMMA fragment 映射
- 跨 tile / 跨 stage 的软件流水线 (software pipelining)

目标相同：
overlap 内存搬运与计算 → 提高片上复用 → 缩短依赖链 → 减少 barrier 与 replay → 持续喂满 HMMA

代表性技术：
- 寄存器预取与双缓冲；
- 让下一 K tile 的加载与当前 QK 计算重叠；
- 在 HMMA 仍在执行时预置 PV 操作数；
- 按 phase swizzle 的共享内存布局；
- 128-bit 向量化访存；
- 显式的 QK/TN 与 PV/TT HMMA fragment 归属；
- 跨 tile 与 stage 边界的软件调度。

我们不是去模拟一条 cp.async 指令，而是重建现代硬件指令所设计的"内存/计算重叠"。

## Layer 1 — 把 KV Cache 搬得正确、宽、且只一次
Paged KV 把逻辑 token 映射到物理 page。朴素的 SM70 路径反复：
load Page ID → 算地址 → load 窄 FP8 fragment → 转换 → 重复
这浪费在地址运算、依赖等待、标量访存上。

#268 的 128-bit XQA 工作复用 page 元数据并做配对对齐加载。

代表性整模型 batch 结果：B16 / 16K: 529.071 → 570.982 tok/s (+7.92%)。
对应算子增益在代表性长上下文 XQA shape 上约 21%–26.5%。

## Layer 2 — 把 D=256 Attention 重写成 Volta 原生流水线
降低数据搬运开销后，Attention 主体本身被重构。关键组件：
- **D256 Split-D**：把 D=256 拆成四个 D64 slice。配对 warp 共享 QK 概率计算，
  在不重算相同 QK 的前提下提高 PV 并行度。
- **N32 Online Softmax**：保留 causal online-softmax 与 FP32 累加契约，
  不物化完整 score 矩阵。
- **K-stage Ping-Pong**：在不同共享内存 stage 间交替 K/D64 panel，降低 barrier 与等待压力。
- **Split-KV3**：把长前缀 KV 工作拆成三个分区（有用时），再合并 FP32 部分状态。
- **GQA Multi-Head Packing**：把 6 个 GQA query head 打包进更宽的 Tensor-Core 工作。
- **Wide QK / PV**：把许多碎片化的小 Tensor-Core 操作变成更大、更规整的 QK/PV GEMM 风格工作。
- **Prefix / Causal-Tail Separation**：把完全可见的长前缀与精确的 causal 尾部分开调度，
  再合并 online-softmax 状态。

这一优化族历经 PR #198、后续 D256 / Split-KV3 工作、v1.3.0、PR #286。

结果：17.92 TFLOP/s → 46.63–47.1 TFLOP/s → ≈60.8 TFLOP/s。
同一代 GPU。同一批 Tensor Core。软件不再浪费它们。
≈79 TFLOP/s 保留为实验性研究上限，不作为默认生产质量声明。

## Layer 3 — 稀疏 Attention 也必须原生适配 V100
Qwen3.8 Flash Next QSA 需要的不只是"少选一些 token"。运行时还要处理：
sparse block selection；physical-page mapping；Page4 K/V 复用；每行精确 mask；最终 QK/PV 计算。

PR #387 把 8 个相邻 query 行分组，使重叠的 Page4 K/V block 只加载一次，
同时保留每行精确的 4-bit mask。然后直接用 Volta WMMA 做 QK 与 PV。

代表性结果：
- 旧 QSA 路径：55.151 ms/layer/rank
- Grouped Page4：9.632 ms/layer/rank (+0.362 ms planner)
- Attention 加速：5.518×
- 整模型纯 prefill 提升：32K +32.36%，64K +29.93%，131K +32.69%

## 🧩 由 Profiling 驱动的优化
1Cat-vLLM 不会在一个 kernel 变快后就停下。当 QSA 加速后，profiling 显示下一个热点
移到了 NVFP4 MoE prefill。PR #390 用 indexed W13 执行 [tokens × topK, hidden]
消除了输入扩展瓶颈。

代表性结果：
- 8K 算子链：6.026752 → 4.235264 ms (1.423×)
- 整模型纯 prefill：32K 5998.65 → 6507.10 tok/s；64K 5777.43 → 6241.48 tok/s；131K 5450.92 → 5871.47 tok/s
- PR #393 进一步融合精确 FP16 SwiGLU，把 N320 W13 尾部拆成 N256+N64，去掉浪费的尾 tile 工作。

这是本项目的优化哲学：Profile 真实模型 → 移动瓶颈 → 再 Profile。

## 🎯 先有快的 target-only decode，再谈投机解码
在依赖 DFlash2 或 MTP 之前，target 模型本身必须快。
PR #415 报告 Qwen3.8-Flash-Next-NVFP4 on 4× V100：8K 输入 / 512 输出，无 MTP，full CUDA Graph。
- Control：65.864 tok/s (15.183 ms TPOT)
- Candidate：80.732 tok/s (12.387 ms TPOT)
这是 target-only 吞吐。

该路径还通过：GSM8K 15/16 strict；Natural stop 16/16；加权自然输出 decode 80.935 tok/s。

## ⚡ DFlash2 on SM70
传统自回归 decode 每产出一个 token 需要一次 target-model pass。DFlash2 改变执行模型：
一个 block-diffusion 草稿模型提出若干未来 token，target 一起验证它们。

有效服务循环变成：
draft 若干候选 → target 验证一个 block → 接受多个 token → 每轮 target 前进超过 1 个 token。

对 Qwen3.8 DFlash2，面向发布的 SM70 栈还优化：draft Attention；selector；grouped verifier；
GDN metadata；sparse rejection；NVFP4/QPN 路径；sampling；CUDA Graph；prefix state；
Mamba align；tool / structured-output state。

其 draft Attention 本身用 `FLASH_ATTN_V100`，而不是退回无关的通用路径。

### DFlash2 长上下文衰减
长上下文不应让投机验证成本不必要地增长。PR #328 改非锚定 paged-prefill 循环，
让它从 draft 实际用到的第一个 sliding-window tile 开始。
在 256K：
- Draft attention：0.422912 → 0.246784 ms/layer
- 五层投影：2.114560 → 1.233920 ms
- 候选中位数：32K 0.252928 ms；128K 0.243712 ms；256K 0.246784 ms
该 draft-attention 分量在 32K 之后的上下文斜率几乎被消除。

## 🔢 量化 / 算子栈
V100 早于当前 LLM checkpoint 使用的许多格式。1Cat-vLLM 因此把量化支持当作**算子设计**问题，
而不只是 loader 问题。当前 SM70 工作包括：
AWQ / W4A16；TurboMind SM70 kernels；compressed-tensors；FP8 E4M3 / E5M2 KV 存储；
ModelOpt NVFP4；MXFP4；Quark W4A16 INT4 / UINT4；QPN8；QPN4；QPN2；grouped MoE；
exact-shape decode GEMV；custom SM70 sampling 路径。

目标不是"该 dtype 能解析"。目标是：量化格式在 Volta 上成为**可用的高性能服务路径**。

### Qwen3.6-35B-A3B NVFP4
PR #270 为混合 ModelOpt NVFP4 checkpoint 增加精确 SM70 路由。要点：
FP8 dense 投影；W4A16_NVFP4 routed/shared experts；grouped TurboMind MoE；
duplicate expert-slot 保留；混合精度 GDN 路由；MTP 冷启动 warmup。

对齐 no-MTP：
- AWQ：prefill 0.3813 s；decode 113.71 tok/s
- NVFP4：prefill 0.4216 s；decode 116.99 tok/s
- MTP4：174.76 tok/s (1.49× NVFP4 no-MTP)
质量：GSM8K 122/128 (95.3125%)；invalid outputs 0；repetitive records 0。

### DeepSeek-V4 on V100
DeepSeek-V4 工作不止于单个稀疏 attention kernel。SM70 栈包括：
sparse MLA；FP8 dense 投影；MXFP4 experts；grouped MoE；Indexer；KPool；
Q 归一化 / RoPE / KV 插入；custom TP4 all-reduce；PP2×TP4 执行；exact GEMV 热点。

代表性结果：
- TP8 no-spec：≈65.1 tok/s
- PP2×TP4 严格质量控制：73.539 tok/s
- PP2×TP4 合并端点：73.613–73.646 tok/s
严格质量控制：GSM8K 64/64；HumanEval 29/32；LongBench 44.740。

### GLM-5.3 on V100
当前 GLM-5.3 SM70 路径用：
ModelOpt NVFP4 MoE；FP16 非 expert 权重；FP8 E4M3 KV；TP4 / PP2；sparse MLA；
exact KDA GEMV；fused KDA f/g；mHC；custom all-reduce；full decode CUDA Graph。

保留的稳定性结果：
- Decode：53.013085 / 53.018516 / 53.017527 tok/s
- 均值：53.016376 tok/s；均值 TPOT 18.862097 ms
- 1K prefill：266.039984 tok/s
质量审计还记录了一个 reasoning-mode 注意点：Max reasoning 在简洁代码任务上可能耗尽输出预算，
而定向的 low-reasoning 重跑能完成并同时通过 AST 与外部执行检查。

## 🧠 我们所说"让 Volta 再快起来"的意思
我们不声称 V100 有 A100 / H100 / Blackwell 相同的理论峰值。重点不同。

大量现代推理软件不再认真为 SM70 优化。这造成两个 gap：
hardware-generation gap + software-neglect gap。1Cat-vLLM 攻的是第二个 gap。

当代表性 Attention 有效算力从 17.92 TFLOP/s 到 46–47 TFLOP/s 再到 ≈60.8 TFLOP/s，
而真实 27B 256K decode 仍达 50.376 tok/s，结论不是 V100"变成了 A100"。
结论是：**软件不再浪费 V100**。
