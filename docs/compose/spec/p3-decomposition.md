---
feature: p3-decomposition
status: design
updated: 2026-09-24 (R299 design)
branch: feat/p3-decomposition
commits:
---

# P-P3 分解：79T/PR#286 复刻（GQA 打包 + prefix/tail 分离 + block 级 rescale）

## [S0] 依据（三层证据）

1. **R298-P 终局定性**：预填充 GPU-bound（GPU 等待 152.6 s = 墙钟 79%）；FA 占 GPU ~60-70% ⇒ **GPU 内核 = 剩余最大肉**，本刀为主刀。
2. **P3-DESIGN-RECON.md**（侦察，文件:行级证据）：79T 数据流全拆解、必须件 10 / 可省件 9、**op 单算子形态判定**、显存判定（16 GB 贴边账）、风险表（FP16 分子溢出实测 75310>65504、KV 非 32 对齐静默错值、数值四守卫）。
3. **R296b/R297**：桶化已治 CPU 侧（形状稳定、rebuild -86%）；1cat 梯子定位 = 我们 Path A 在 45.9 TFLOPS（Split-D 台阶），PR#286 台阶 = ≈60.8。

## [S1] 目标

FA 算子 **45.9 → ≥57 TFLOPS**（吃下 45.9→60.8 台阶的主要部分）；256K 预填充再进一档。

## [S2] 设计（形态判定引侦察 §2，实现引 §1/§3）

1. **单 ggml 算子**（内部双流 + 私有 workspace）——**不拆 QK/rescale/PV 三段**（拆段令 score 入图被多槽按槽翻倍 = R282 式 OOM，且丢 prefix/tail 重叠）。
2. **GQA 打包**：M = n_q×6（6 个 Q 头打包进 M 维，K/V 载入摊 6 倍）。
3. **QK = 一次 cuBLAS 大 GEMM**（cublasGemmEx N/T，m=n_q×6, n≤kBlockN, k=256, alpha=softmax scale, f16 输出）→ **score workspace**（f16 转置布局，私有、常驻复用）。
4. **PV = CUTLASS 大 GEMM**（FP16 操作数 + FP32 MMA 累加 + FP32 块输出）。
5. **block 级 rescale**：每 kBlockN 行一次 FP32 合并（vs 现行每 32 行）。
6. **prefix/causal-tail 分离**：前缀零 mask 大 GEMM；因果尾批处理三角（批 QK + 对角 mask + fine-PV + 首 64 token 精确修复）。
7. **kBlockN 三档自适应 + tail 串行**（显存判定）：n_q=2048 → 24576 块（score 0.60 GB/组，可行但贴边）；n_q=8192 → 8192 块（0.81 GB）；kBlockN>8192 且 n_q=8192 降档。**score 显存 = rows × kBlockN × 2B**。
8. **数值四守卫**（README:37-51 / stable_rows.cuh:8-94）：stride-8 采样 max + margin、指数上限、中心化 value、FP32 累加/块输出。
9. 门控：`LLAMA_SM70_FA_DECOMP`（**判值**，默认 OFF；=0 与现行逐位等价路径共存）。
10. 学习源（只读）：`v100-refs/1cat-vllm/csrc/attention/sm70_79t/*`（prefill.cu 数据流 / stable_rows.cuh 数值守卫）；**禁抄常量**（X16）；P-P3-T 的显式 mma PV 碎片工作可复用为尾部 fine-PV（P3-RECON §6）。

## [S3] Out of Scope

JS4 dual-CTA、P-P4 常量扫描、多请求 batch、legacy state 适配、E4M3 bridge/paged、权重格式/KV 压缩、上游贡献。

## [S4] 预登记及格线（对照 r=1.08 采用基线，stress-256k 同尺双 rep）

| 指标 | 基线 | 及格 | 期望 |
|---|---|---|---|
| FA 算子（n_q=2048/n_kv=262144 同形） | 287.5 ms（45.9 TFLOPS） | **≤230 ms（≥+25%）** | ≤215 ms |
| 256K TTFT（spec-on） | 208.9 s | **≤194.7 s（≥+7%）** | ≤185 s |
| 纯步（spec-off 锚） | 34.4 ms/token | 不回退 >2% | 平/升 |
| 显存 | 4 卡 11.3 GB | ≤4×16 GB 包络（score ws 计入） | — |
| 门值 | f3edac19/69207026 | **允许重立一次**（结构重写条款；重立须报） | 尽量原样 |
| 机制自证 | — | 新路径探针计数绿（=0 走原路径） | — |

及格 = 算子线 + TTFT 线至少其一达标且其余不回退。失败入失败记录（只改灰）。

## Tasks

- [ ] T1: 单算子骨架（新 op/FA 路由钩子 + 私有常驻 workspace + 门控）——covers S2-1/9；判据：=0 逐位等价
- [ ] T2: GQA 打包 QK（cuBLAS 块）+ PV（显式 mma/CUTLASS，FP32 块输出）+ score 布局——covers S2-2/3/4
- [ ] T3: block 级 rescale + prefix/tail 分离（批三角 + 对角 mask + 首 64 修复）+ kBlockN 三档降档 + 数值四守卫——covers S2-5/6/7/8
- [ ] T4: 算子级 A/B（test-backend-ops 同形）→ e2e stress-256k A/B（双 rep）→ 对照 [S4] 采用判定；门值重立须报；填 AGENTS §1.0 与账本 R299
