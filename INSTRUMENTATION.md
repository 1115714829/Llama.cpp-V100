# INSTRUMENTATION.md — 本项目自建量具（env-gated，未设时零开销）

> 背景：ncu / nsys 在本机**无法 profile** 18-29 GB 模型（device memory save/restore 崩），
> 所以 kernel 级证据只能靠 in-tree、env-gated 的计时量具。本文档列出全部量具、用法与解读规则。
> 2026-09-20 由 DSH 在 B1/B2 批次补强（新增 LLAMA_ROUND_TIMING_SYNC 与 GGML_CUDA_OP_TIMING）。

## 1. 量具清单

| env | 位置 | 作用 |
|---|---|---|
| LLAMA_ROUND_TIMING=1 | src/llama-context.cpp 的 process_ubatch / perf_get_data | 每轮 build / alloc / set_inputs / enqueue 四段耗时 + reuse/rebuild 计数 |
| LLAMA_ROUND_TIMING_SYNC=1 | 同上（**2026-09-20 新增**） | 在 graph_compute 后**显式同步设备**并另计一份 ⇒ 报告里的 sync_us 才是**含设备等待的真时间** |
| LLAMA_SPEC_TIMING=1 | common/speculative.cpp 的 draft() | 每轮 draft_decode / selector / walk |
| GGML_CUDA_OP_TIMING=1 | ggml/src/ggml-cuda/ggml-cuda.cu（**2026-09-20 新增**） | **按算子聚合的 GPU 时间**（逐节点 CUDA event 对），进程退出时打印一张表 |
| GGML_CUDA_DISABLE_GRAPHS=1 | common.cuh | 关 CUDA graph 捕获 —— **per-op 计时必须配它**（图内 event 无效） |
| GGML_CUDA_P2P=1 | ggml-cuda.cu:391 | 开 cudaDeviceEnablePeerAccess（复现「当前最好」必须带） |

## 2. 报告字段与解读规则（**重要**）

### 2.1 [RT] perf 行

来源是 **target context**（tools/server/server-context.cpp:656 调用 llama_perf_context(ctx_tgt)）。格式：

    [RT] perf: ctx=<模型名> n_ctx=<N> rounds=<N> reuse=<N> rebuild=<N> | build_us=.. alloc_us=.. setin_us=.. enqueue_us=.. sync_us=..

| 字段 | 它**是**什么 | 它**不是**什么 |
|---|---|---|
| enqueue_us | ggml_backend_sched_graph_compute_async() 的**墙钟窗口**（异步提交 + 调度器内部跨设备拷贝等待） | ⚠️ **不是 GPU 执行时间** |
| sync_us | 同一次 graph_compute **加上显式 ggml_backend_sched_synchronize()** 的墙钟 | 仍是 host 视角，但已含设备等待 |
| build / alloc / setin_us | 建图 / 分配 / set_inputs 的 host 时间 | — |

⚠️ **必须记住**：llama-context.cpp 里 `graph_compute(res->get_gf(), ubatch.n_tokens > 1)` 的第二个参数是 **batched（选线程池）**，不是同步开关。
真正的同步在 `:1376`，条件是 cparams.pipeline_parallel，而它要求 split_mode==LAYER 且 !has_tensor_overrides()；
本 harness **永远传 --tensor-split** ⇒ **恒不同步**。所以旧文档里「M>1 才同步所以可信 / layer 的数字是假的」**两条都不成立**。

### 2.2 [RT] target decode+sync 行

server-context.cpp:3709 —— llama_decode(ctx_tgt, batch) 加 llama_synchronize(ctx_tgt) 的墙钟，**包含设备等待**，是 target 整步的可信总账。

### 2.3 [OP] 表（新增）

进程退出时打印，例如：

    [OP] graphs=12 nodes=1987 gpu_total=21.830 ms/ubatch
    [OP] MUL_MAT           calls=1234  total= 12.345 ms  56.6%  avg= 0.010 ms
    [OP] FLASH_ATTN_EXT    calls=16    total=  5.274 ms  24.2%  avg= 0.330 ms

- gpu_total 是**每 ubatch 求和后的平均值**（不是墙钟，不含 host 间隙）。
- 与墙钟的差额 = host 发射 / 同步 / 间隙 ⇒ 这正好回答「21.8 ms 提交窗口里有多少是 GPU 真活」。

## 3. 标准用法

    # (a) 真 GPU 时间与分段（不关图）
    LLAMA_ROUND_TIMING=1 LLAMA_ROUND_TIMING_SYNC=1 LLAMA_SPEC_TIMING=1 \
      CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 bash /root/p60-ab-harness.sh

    # (b) 按算子聚合（必须关图；有 serialization 代价，仅用于归因）
    LLAMA_ROUND_TIMING=1 GGML_CUDA_OP_TIMING=1 GGML_CUDA_DISABLE_GRAPHS=1 ... 同上

## 4. 本批次（B1/B2）改动清单

| 文件 | 改动 |
|---|---|
| src/llama-context.h | 新增 t_rt_sync_us 计数 |
| src/llama-context.cpp | 新增 LLAMA_ROUND_TIMING_SYNC 分支（显式同步 + 计时）；[RT] perf 行加 ctx=<模型名>、n_ctx 与 sync_us |
| ggml/src/ggml-cuda/ggml-cuda.cu | 新增 ggml_cuda_op_timing（GGML_CUDA_OP_TIMING）：逐节点 CUDA event 对，按算子聚合，退出时打印 |

> 全部 env-gated：**未设环境变量时零开销**；ggml-cuda.cu 的事件记录只在 !use_cuda_graph 时发生。