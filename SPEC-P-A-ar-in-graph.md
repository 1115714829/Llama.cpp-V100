# SPEC P-A：把 allreduce 纳入 CUDA 图捕获（R2）

> 目的：让 138 次/轮的主机介导 allreduce（事件计时 **53.0 µs/次**，每轮 **7.3 ms**）变成图内节点，
> 目标 **≤25 µs/次**（1cat 在同位置实测 18 µs/次，见 `csrc/custom_all_reduce.cuh:1944-1955`：push kernel **只在 `cudaStreamCaptureStatusActive` 时启动**）。
> 预期收益 **−4.8 ms/轮**（57.6 → ~53），是四项结构性工作里单项收益最大的一项。

## 1. 现状（已核实的源码事实）
- AR 由 **meta 后端在 `graph_compute` 之间、主机侧调用**：`ggml/src/ggml-backend-meta.cpp:2453`
  `backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());`
- CUDA 侧只把它注册成接口：`ggml/src/ggml-cuda/ggml-cuda.cu:5995-5996`（`ggml_backend_comm_allreduce_tensor`）。
- 因此每轮 **139 段 × 3 后端 = 417 次 graph_compute** + **138 次 AR**，全部在**任何被捕获的图之外**。
- 已证伪的替代路线：R1「让 AR 主机侧更便宜」**无效**（enqueue 省 4.8 ms/轮但轮时零改善；在役 61.5 vs NCCL 53.0 µs），
  见 `SESSION` §18.5/§18.7。**唯一有效方向是让 AR 进入图。**

## 2. 目标形态（两条候选路线，按侵入度排序）
**A1（推荐先试，改动最小）**：让 meta 后端**不要**在 AR 处切断图 —— 即把 allreduce 表达为**图内节点**：
- 新增一个 ggml 层算子/组合（例如 `GGML_OP_ALLREDUCE` 或"peer-copy + spin + 求和"的复合节点），
  由 CUDA 后端实现（**必须是 capture-safe 的：只用设备侧标志/自旋，主机零交互**，这正是 patches/0006 里已在小样验证 1000 轮正确的协议）。
- meta 后端在切图时把该节点留在子图内（不再单独调用 comm_allreduce）。
- 需要：ggml 算子注册 + CUDA 后端实现 + meta 后端切图逻辑的最小改动（**这是上游架构级改动，须逐步小样验证**）。

**A2（备选，侵入更小但收益可能打折）**：让**同一次 AR 的 3 个设备端 kernel 被同一张图捕获** ——
即把"按 backend 逐个 submit"改成"预先把 3 个后端的工作合并为一次可捕获提交"。若 A1 工作量过大可选此路。

## 3. 可复用的既有资产
- `1cat-vllm-v100-study/patches/0006-device-side-push-allreduce-refuted.patch`：
  内含**已证明正确的设备侧协议**（slot+epoch 防跨迭代覆盖、逐块 flag 无全局屏障、自旋上限）、
  以及 `GGML_CUDA_AR_DEVICE` 的接线方式（`comm_init_device` + `try_allreduce_device`）。
  **R1 证伪的是"主机侧更便宜"这条理由，不是协议本身** —— 该协议在 160 KB 下 1000 轮零错误、零自旋超时 ✓。
- `/root/ar_push_test.cu`（独立小样，nvcc 单文件编译）：用来在**不加载模型**的前提下快速验证任何协议改动（每次 ~1 分钟）。

## 4. 实施顺序（每步都要独立可验证）
1. 在小样（`ar_push_test.cu`）里把 AR 改成**可捕获**形态并验证：**在 `cudaStreamBeginCapture/EndCapture` 内调用**，
   捕获后 `cudaGraphLaunch` 重放多次，结果与逐次调用一致（含跨迭代 slot 复用）。
2. 在 llama.cpp 里做 env 门控的最小接线：`GGML_CUDA_AR_IN_GRAPH=1` ⇒ 启用图内 AR 路径；默认关（零行为变化）。
3. 端到端 A/B（同库、env 门控）：`[AR] ar_us_avg`、每轮 ms、tg、AL；**sha256 同配置可复现**。

## 5. 验收门（硬性）
- `[AR] ar_us_avg` **≤25 µs**（现状 53.0）且每轮 **≤50 ms**（现状 57.6）；
- 同配置 greedy sha256 **可复现**（结构性改动不要求与旧值逐位相同，但必须自洽；若与旧值相同更好）；
- **AL 不劣化**；
- 构建三查 + 四库 md5 + 二进制标记串（`strings <lib> | grep -c GGML_CUDA_AR_IN_GRAPH`）；
- 测量独占机器、正式数字带 drop_caches、报告同时给 AL 与 ms/轮。

## 6. 已知风险与止损
- 图捕获对**自旋/原子操作**的容忍度、以及捕获期间的 `cudaMalloc`/主机同步（必须全部预先分配、零主机交互）；
- meta 后端切图逻辑是上游核心路径，改动要**最小且可回退**；
- **止损条件**：若第 1 步（小样内捕获+重放）无法在 ~1 小时内得到"与逐次调用逐位一致"的结果，则判定该路线在本机不可行，
  归档为负面结论并把优先级让给 P-B/P-D。
