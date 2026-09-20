# goal-phase1-findings.md — Goal 阶段进展与学到的东西（2026-09-20）

> 本文件是 Goal「把 1cat-vLLM 的 V100/SM70 优化方法学进 llama.cpp，以固定口径下单流吞吐中位数验收」的过程档案。
> 口径固定的部分：target=`Qwen3.8-27B-Q8_0.gguf`（29,047,086,048 B，sha256 校验通过）+ 官方 DFlash2 draft `n=7`、
> ctx 8192、官方 thinking 采样（temp1.0/top_p0.95/top_k20/min_p0/presence0/rep1.0）、seed 42、512 tokens、固定 3 prompt 集。
> 卡数与切分模式是我的决策变量，必须记录。

## 1. 工具箱：可复现的评测脚本（Goal Done-when 第 1 项，已完成）

`/root/p60-ab-harness.sh`（参数化：`CARDS`/`SPLIT`/`SPEC`/`TAG`/`NPRED`/`L`/`NODROP`）：
- 打印 libs md5（`libllama.so` / `libggml-cuda.so` / **`libllama-common.so`** —— 最后这个是必须的，
  因为 spec 代码在 common 库里）、模型大小、每 GPU 显存、每 prompt 的 tg 与 AL、**中位数**、
  `LLAMA_ROUND_TIMING` / `LLAMA_SPEC_TIMING` 分解行、以及 **temperature=0 的 greedy 输出 sha256**（正确性基线）。
- **`NODROP=1`**：跳过 `drop_caches`，模型保持 page-cache 热 ⇒ 每臂加载 **15 s 而不是 ~270 s**
  （本机无盘、`/` 是 NFS，drop 掉缓存后要重新读 29 GB）。**诊断用 NODROP；要写进结论的数字必须带 drop。**

**基线已记录** `/root/goal-baseline.txt`（未改动树 = b11053 + C4 + C5 + PR#27858 + 埋点）：

| 臂（3 卡同 NUMA 0/1/2） | 每 prompt tg | 中位数 | AL | greedy sha256 |
|---|---|---|---|---|
| `layer`（工作基线） | 66.52 / 70.09 / 87.83 | **70.09** | 5.16/5.21/6.62 | `ccc284e4ec…` |
| `tensor` | 55.95 / 45.68 / 66.86 | **55.95** | 5.21/4.11/6.08 | `69207026ca…` |

⚠️ **两个臂的 greedy 输出不同**（`ccc2…` vs `6920…`）⇒ 张量并行改变数值（allreduce 顺序），
所以 **layer vs tensor 不是单变量对比**，跨模式比吞吐时必须说明这一点。

## 2. ★首个落地的优化：并行化 CPU selector（+26.8%，且逐位一致）

**问题**：`common/speculative.cpp::build_dflash2_selector_cpu()`（我按 PR #27858 移植的，仅 `--split-mode tensor` 下启用）
每轮对 8 个 block 位置各做一次全词表 top-k + 门控矩阵乘，**单线程** ⇒ **23.45 ms/轮**。

**关键数据**：词表 `n_vocab = 248,320`（官方 padded），selector **rank = 72**
（推法：`selector_predecessor.weight` 大小 35,758,080 B ÷ 2 B/元素(fp16) ÷ 248,320 = 72）。
8 个位置**彼此独立**。

**改法**：8 个位置按线程切分（`std::thread`，`min(n_tokens, 8)` 个），每个位置的算术**逐算子、逐顺序不变**
⇒ 结果**逐位一致**。同时把 8 个位置的 logits 指针先取好（指针运算，读操作），并行体只读。

**实测（tensor，3 卡，NODROP）**：

| | 改前 | 改后 |
|---|---|---|
| CPU selector | **23.45 ms/轮** | **4.41 ms/轮**（5.3×） |
| 每 prompt tg | 55.95 / 45.68 / 66.86 | **70.92 / 56.62 / 81.29** |
| 中位数 | 55.95 | **70.92（+26.8%）** |
| AL | 5.21 / 4.11 / 6.08 | **完全相同** |
| greedy sha256 | `69207026ca43b19f5df2871a658d9f88bf3c583387a39d4a5ecceb8b21eaa147` | **同一个** |

⇒ **正确性由"greedy 输出逐位一致 + AL 完全相同"证明**（正是 1cat 的验收口径）。

## 3. ★我的一个错误结论与纠正（重要，勿沿用）

**我曾断言**："1cat 的 4 张 V100 在一个 NVLink 岛内，我们是 3+3 跨 NUMA，所以 TP4 必然更慢" —— **错的**。

**事实（我自己的测量反证）**：用户的生产单元 `vllm-1cat.service` 在本机用的是
`CUDA_VISIBLE_DEVICES=0,1,3,4` ⇒ **0/1 在 NUMA node 0、3/4 在 NUMA node 8，就是跨岛 TP4**，
而它跑出 **221.6 / 230.8 / 263.2 tok/s**。⇒ **跨岛 TP4 不是障碍**；差距是 llama.cpp 的实现问题。

**真正原因（代码证实）**：`ggml/src/ggml-cuda/allreduce.cu:399-403`
```cpp
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int * devices, size_t n_devices) {
    if (n_devices != 2) {
        GGML_LOG_DEBUG("%s: internal AllReduce only supports n_devices=2 (got %zu); falling back\n", ...);
        return nullptr;
    }
```
⇒ **llama.cpp 的快速内部 allreduce 只支持 2 张卡**；TP3/TP4 全部退回通用（慢）路径，**逐层**付费。
另外 `ggml/src/ggml-cuda/ggml-cuda.cu:391` 的 **`GGML_CUDA_P2P`** 才开启 `cudaDeviceEnablePeerAccess`，
我们从未设置过。可用旋钮有 13 个 `GGML_CUDA_AR_*`（`MAX_BYTES`/`COPY_THRESHOLD`/`COPY_CHUNK_BYTES`/`KERNEL_BLOCKS`/`BF16_THRESHOLD`…）。

**实测（3 卡同 NUMA）**：`layer` 70.45（NV2 内）｜`tensor` 55.95→**70.92**（修 selector 后）｜
**TP4**（跨岛，0,1,2,3）：`layer` 67.83、`tensor` 50.67。
⇒ 修 selector 后，**TP3-tensor 与 TP3-layer 基本持平（70.9 vs 70.5-74.5）**，而 tensor 的 target 前向更省（见 §4）。

**外部先例**：另一个 V100 fork（`anyei/llamacpp-v100`）的改动 #3 就是
"One-shot P2P AllReduce（NVLink 直连小张量，`GGML_CUDA_ALLREDUCE=p2p`，上限 4 MB）"——**方向与我们的缺口一致**。

## 4. M 曲线（每轮成本随验证批 M=n_max+1 的变化）—— 决定杠杆归属

| 配置 | M | tg | 每轮 |
|---|---|---|---|
| layer + none | 1 | 24.47 | **40.9 ms** |
| layer + dflash n=1 | 2 | 36.23 | **52.4 ms** |
| layer + dflash n=7 | 8 | 74.51 | **69.9 ms** |
| tensor + none | 1 | 31.19 | **32.1 ms**（= token 成本） |

- layer 模式：M=1 40.9 ms ⇒ **layer 把流水线串行化**（比 tensor 的 32.1 ms 还慢），
  但每加一行只有 2.9–11.5 ms 边际；M=8 时 69.9 ms。
- tensor 模式（修 selector 后）：每轮 ≈ 73 ms = target **29.7** + draft **15.1** + selector **4.4** + walk 0.09 + host **2.4**
  ⇒ **合计 51.6 ms，尚有 ~21 ms 未归因**（待查：CPU 采样 / 接受回滚 / server 开销）。
  ⚠️ 但 tensor 的 target M=8 是 **29.7 ms**（`graph_compute` 在 M=8 是同步的，可信）；
  layer 的 target 反而是 ~58 ms（由 M 曲线反推）⇒ **tensor 的 target 前向效率高约 2×**。
- **带宽下限换算**：Q8_0 29 GB / 3 卡 = **9.7 GB/卡**，按 ~900 GB/s ⇒ **≈10.8 ms/轮**（与 M 无关）。
  ⇒ tensor M=8（29.7 ms）是下限的 **2.8×**；整个轮（73 ms）是 **6.8×**。1cat 4 卡 FP8（6.75 GB/卡、17.5 ms/轮）
  约为其下限的 2.3×。**⇒ 差距是"效率"，不是带宽，也不是拓扑。**

## 5. 量具与踩坑（都实测过，值得复用）

1. **`LLAMA_ROUND_TIMING`**（`src/llama-context.{h,cpp}`）：build/alloc/set_inputs/`graph_compute` + reuse/rebuild。
   **注意**：`graph_compute` 只在 `ubatch.n_tokens > 1`（即 M=8 验证批）时同步，所以 **M=8 的数字可信**；
   **layer 模式下这个数字是假的**（1.54–5.0 ms/轮，低于 10.8 ms 的带宽下限 ⇒ 一定是异步/流水，
   不要用它做归因）。
2. **`LLAMA_SPEC_TIMING`**（`common/speculative.cpp`）：`draft()` 内 draft_decode/selector/walk + 每 32 轮一行。
   ⚠️ 报告必须从**每轮都会执行**的函数里 tick。放在 `common_speculative_process()` 里**不会打印**
   （它是 prefill 钩子）；放在 `common_speculative_accept()` 里**也可能不打印**（`impl_last[seq_id] == nullptr` 时提前 return）。
   **正确位置是 `common_speculative_draft()`**。
3. **libllama 里用 `LLAMA_LOG_INFO` 埋点可能看不到**（连试 3 次无输出，原因不明）⇒ 先用 `fprintf(stderr, ...)` 证明函数真的被执行。
4. **`llama-bench` 完全不打印 libllama 的 INFO 日志**（实测 0 条）⇒ 不能用它验证库内埋点。
5. **无盘网络启动**：`/` 是 NFS ⇒ 下载 / 加载模型 / **编译** / `sha256sum` 都吃同一张网卡，不要并行；用户也提醒过"可以多等等"。
6. **显存"残留"= HBM2 映射进系统内存导致的页缓存**：`sync; echo 3 > /proc/sys/vm/drop_caches` 即可清掉
   （实测 buff/cache 70 GB→1 GB，GPU 1–5 从 2.4–3.1 GB 回到 0–1 MiB）。**不要**误判为泄漏、更不要 reset GPU。
7. **`nsys` 必须在被测进程"优雅退出"时才写 `.nsys-rep`**；`kill -9` 会丢报告（两次实测）。
8. **拓扑事实**（`nvidia-smi topo -m`）：GPU0/1/2 = NUMA0（CPU 0-87），GPU3/4/5 = NUMA8（CPU 88-175）；
   组内 NV2，跨组 SYS。**但见 §3：跨组不等于不可用**（1cat 就在跨组跑 222–263）。

## 6. 当前状态与下一步（按价值排序）

- **当前最好**：TP3（0,1,2）`tensor` + 并行 selector = **70.92 t/s 中位数**（目标 ≥110，还差 +55%）。
- **下一步候选**：
  1. **定位 tensor 模式那 ~21 ms/轮**（把报告 tick 移到 `draft()`，再测）。
  2. **`GGML_CUDA_P2P=1` 实验**（零风险开关，可能省掉 host 暂存）+ 尝试 `GGML_CUDA_AR_*` 旋钮。
  3. **修多设备 allreduce**（现在只支持 2 卡；参考 anyei fork 的 one-shot P2P）——TP3/TP4 的逐层付费。
  4. **target M=8 走 Volta FP16 HMMA**（29.7 ms/轮 = 整轮的 41%，是最大单项；M 曲线证明其高于带宽下限 2.8×）。
  5. **draft 前向 15.1 ms**（1.14 GB ⇒ 有效 ~76 GB/s，dispatch-bound）。
- **验收提醒**：所有对外数字必须带 `drop_caches`、同源 A/B（**含 `libllama-common.so` 的 md5**）、
  ≥3 prompt、固定 seed，并同时报 AL 与每轮 ms；正确性用 greedy 逐位一致或 AL 不变来证。

## 7. 第二个落地优化：`GGML_CUDA_P2P=1`（+10.6%，AL 不变）

`GGML_CUDA_P2P`（`ggml-cuda.cu:391`）此前**从未设置**，而它才是开 `cudaDeviceEnablePeerAccess` 的开关。
开启后（tensor，3 卡，NODROP）：

| | 每 prompt tg | 中位数 | target `graph_compute`/轮 |
|---|---|---|---|
| 不开 | 70.74 / 56.39 / 80.32 | 70.74 | 29.7 ms |
| **`GGML_CUDA_P2P=1`** | **78.21 / 63.71 / 93.17** | **78.21（+10.6%）** | **24.2 ms** |

AL **完全相同**（5.21/4.11/6.08）⇒ 数值不受影响（只是对等拷贝路径变了）。
**累计：55.95 → 78.95 t/s（+41%）**，两个改动都通过"逐位一致 / AL 不变"的正确性门。

## 8. ★决定性负面结果：加卡不能买带宽（allreduce 才是瓶颈）

全部带 P2P、NODROP、同口径、3 prompt：

| 配置 | 中位数 | target `graph_compute`/轮 | draft/轮 |
|---|---|---|---|
| **TP3 tensor（最好）** | **78.95** | 24.1 ms | 14.5 ms |
| TP4 tensor | 75.34 | **41.0 ms** | 18.7 ms |
| TP6 tensor | 48.60 | **62.1 ms** | 26.1 ms |
| TP6 layer | 72.94 | （该埋点在 layer 下失真） | 4.5 ms |

⇒ **target 的每轮成本随卡数"上升"**（24 → 41 → 62 ms），因为 3+ 卡走的是慢 allreduce 路径
（`allreduce.cu:403`：快速内部 allreduce **只支持 `n_devices == 2`**）。
⇒ **"Q8_0 固定量化下、用更多卡拉低每卡字节数"这条路被实测否定**（TP3 仍最好）。
⇒ 同时也否定了我之前的"跨岛所以 TP4 慢"解释：**是 allreduce 实现，不是拓扑**（1cat 跨岛 TP4 在本机跑到 222–263）。

**字节/卡是结构性事实（口径固定时无法改）**：1cat 用 4-bit（~13.5 GB/4 卡 ≈ **3.4 GB/卡/轮**），
我们 Q8_0 29 GB/3 卡 = **9.7 GB/卡/轮** ⇒ 他们每轮每卡少读 **~2.9×** 字节。
所以 65 ms vs 17.5 ms 中约 2.9× 来自字节数，只有约 1.3× 来自实现效率。
（带宽下限：9.7 GB ÷ 900 GB/s ≈ **10.8 ms/轮**；我们实测 65 ms = 6× 下限；1cat 4.6× 其下限。）

## 9. 一轮"排除法"：剩下 ~18 ms 不在哪（全部实测）

| 假设 | 实测 | 结论 |
|---|---|---|
| CPU 采样（`set_logits` 全词表） | **~1 ms/轮**（0.03–0.2 ms/次 × ~4.7 次/轮） | 排除 |
| `common_sampler_sample` 里的 `llama_synchronize` | **0.009 ms/次** | 排除 |
| `accept()` 记账 | ~0（且 `impl_last==nullptr` 时会提前 return） | 排除 |
| `process()` 目标特征钩子 | **0.06 ms/轮** | 排除 |
| target 主机侧图工作（build+alloc+setin） | **2.5 ms/轮** | 排除（占 4%） |
| **speculative checkpoint 的存/取**（hybrid 模型回滚） | **0.027 ms/轮**（`PARTIAL_ONLY` 使其极便宜） | 排除 |
| `--spec-draft-device` 把 draft 钉到单卡 | **两边都 `Aborted (core dumped)`** | **崩溃，此路不通** |

⇒ 剩 ~18 ms 与 **target 的 M=8 前向里"非 `graph_compute` 显式耗时"**部分重合（等待/串行化/
allreduce 落在别处）。**结论：瓶颈在 target 前向本身（GEMM 效率 + allreduce），与 1cat 的判断一致。**

## 11. 正式测量结果（2026-09-20，带 drop_caches，3 prompt，seed 42）

| | 基线 | 现在（本轮交付） |
|---|---|---|
| 配置 | `--split-mode tensor`，3 卡 0/1/2 | 同左 **+ `GGML_CUDA_P2P=1`** |
| 每 prompt tg | 55.95 / 45.68 / 66.86 | **79.23 / 64.23 / 92.49** |
| **中位数** | **55.95** | **79.23（+41.6%）** |
| AL | 5.21 / 4.11 / 6.08 | **5.21 / 4.11 / 6.08（完全相同）** |
| temperature=0 greedy sha256 | `69207026ca43b19f5df2871a658d9f88bf3c583387a39d4a5ecceb8b21eaa147` | **同一个（逐位一致）** |
| CPU selector | 23.45 ms/轮 | **4.29 ms/轮** |
| target decode+sync | — | 37.5 / 49.0 / 53.3 ms/轮（≈整轮 74%；带宽下限 10.8 ms ⇒ 3.5–4.9×） |
| `git status --porcelain` | — | **13 文件，全部在 `llama.cpp/` 内**，无未跟踪（+587/−33） |

**两项落地改动**（都通过"逐位一致 / AL 不变"的正确性门，符合 1cat 的验收口径）：
1. `common/speculative.cpp::build_dflash2_selector_cpu()` —— 8 个 block 位置并行（`std::thread`）⇒ selector 23.45→4.41 ms/轮。
2. `GGML_CUDA_P2P=1` —— 开 `cudaDeviceEnablePeerAccess` ⇒ +10.6%（target 每轮 29.7→24.2 ms 的"发射"实测；真实 decode+sync 48→…）。

**距目标 ≥110 仍差 +39%，剩余量级与两个抓手（需大工程）**：
- target M=8 校验前向 37.5–53.3 ms/轮，是带宽下限（9.7 GB/卡 ÷ ~900 GB/s = 10.8 ms）的 3.5–4.9×；
- ① **M≈8 的量化 GEMM 走 Volta FP16 HMMA**（1cat 核心方法）；② **多设备 all-reduce**（`allreduce.cu:399-403` 仅支持 2 卡）。
- 结构性事实：口径钉死 Q8_0 ⇒ 9.7 GB/卡/轮 vs 1cat 4-bit ≈ 3.4 GB/卡/轮（少读 ~2.9×）。



1. **target M=8 走的 `mul_mat_vec_q`（ncols_dst=8）在 Volta 上从未调过**：
   `mmvq.cu` 的 `MMVQ_PARAMETERS_VOLTA` 对 ncols 5–8 给 `nwarps=2, rows_per_block=2`
   （注释自承**只调过 ncols=1**）⇒ 值得扫 `nwarps ∈ {4,8}`。
2. **多设备 allreduce**（现在只支持 2 卡；外部先例 anyei fork 的 one-shot P2P）——加卡能否买带宽取决于此。
3. **M≈8 走 Volta FP16 HMMA**（1cat 的方法；本机 M 曲线证明 M=8 高于带宽下限 2.8×）。
4. draft 的 14.5 ms（tensor 下）vs 4.5 ms（layer 下）——draft 在 tensor 下白付 allreduce。

## 12. ★★本轮最大发现：本机从未启用 NCCL —— TP3/TP4 的 allreduce 全程走的不是快速路径

### 12.1 问题
`ggml/src/ggml-cuda/allreduce.cu:400-405`：
```cpp
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int * devices, size_t n_devices) {
    if (n_devices != 2) { /* 只支持 2 卡 */ return nullptr; }
```
而 `ggml-cuda.cu:1208-1245` 的初始化链是 **nccl -> internal -> none**，Linux 平台默认走 **nccl**。
所以只要 NCCL 没编进去，链就退化成：nccl 不可用 -> internal 拒绝 `n_devices != 2` -> `none`（meta-backend butterfly，经主机内存中转）。

### 12.2 证据：我们的构建里 NCCL 根本没找到
`build/CMakeCache.txt`：
```
GGML_CUDA_NCCL:BOOL=ON          <- 默认就是 ON
NCCL_INCLUDE_DIR:PATH=NCCL_INCLUDE_DIR-NOTFOUND
NCCL_LIBRARY:FILEPATH=NCCL_LIBRARY-NOTFOUND
```
`ldd libggml-cuda.so` 里**没有** libnccl。上游 `docs/multi-gpu.md:97` 明说 NCCL 不随 CUDA 分发、需要自己装。
=> **我们所有 TP3/TP4/TP6 数据都是在 butterfly 上测的**（这也解释了为什么 target 每轮成本随卡数 24->41->62 ms 上升）。

### 12.3 关键：本机其实"已经有" NCCL —— 在 1cat 的 venv 里
```
/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/
    include/nccl.h                      (NCCL 2.29.7)
    lib/libnccl.so.2                    (59.9 MB)
```
- `ldd libnccl.so.2` 只依赖系统库（libc/libstdc++/libpthread...），**不直接依赖 CUDA runtime** —— 与我们的 CUDA 12.4 无 ABI 冲突。
- 导出 llama.cpp 需要的全部符号：`ncclCommInitAll` / `ncclAllReduce` / `ncclCommDestroy` / `ncclGetErrorString`。
- **所以不需要下载任何东西**（1cat 用的就是这一份，等价于"用 1cat 的通信层"）。
- 注意只有 `libnccl.so.2`，没有 `libnccl.so`；FindNCCL.cmake 用 `find_library(NAMES nccl)` 找不到，
  必须直接把 `-DNCCL_LIBRARY=<绝对路径>/libnccl.so.2` 传进去。

### 12.4 做法与单变量纪律（已验证）
新开 `build-nccl/`（不动原 `build/`），配置：
```
cmake -B build-nccl -DCMAKE_BUILD_TYPE=Release -DLLAMA_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
  -DCMAKE_INSTALL_RPATH=/root/llm/llama.cpp/lib64 \
  -DNCCL_INCLUDE_DIR=<pkg>/include -DNCCL_LIBRARY=<pkg>/lib/libnccl.so.2
```
- 配置输出：`Found NCCL: .../libnccl.so.2`；`Using CMAKE_CUDA_ARCHITECTURES=70-real`。
- **缓存 diff（`build/` vs `build-nccl/`，过滤 LLAMA_/GGML_/NCCL_/BUILD_TYPE/ARCH）只有 2 行**：
  `NCCL_INCLUDE_DIR` 与 `NCCL_LIBRARY` 从 NOTFOUND 变成实际路径 -> **唯一变量就是 NCCL**。
- `ldd build-nccl/bin/libggml-cuda.so.0.24.0` 出现 `libnccl.so.2 => .../ac922_nccl_runtime/lib/libnccl.so.2`。
- 快照 `/root/libdir-nccl`（含 `llama-perplexity`，供 ppl 验收）。

**源码一致性（顺手做掉的一个隐患）**：Windows 本地 13 个改动文件与服务器副本 md5 不同，
实测是**行尾差异**（本地 `git ls-files --eol` = `i/lf w/crlf`，`core.autocrlf=true`；未改过的 `src/llama.cpp` 同样是 `w/crlf`）。
把服务器文件 `sed 's/$/\r/'` 后 md5 **13/13 与本地完全一致** => **内容逐位相同，只有 CRLF/LF 之别**，服务器构建对应的就是我本地那份交付源码。

### 12.5 实现细节与已知代价（重要，涉及正确性）
`ggml-cuda.cu:1000-1072` 的 NCCL 路径：
- `ne < 131072`（3 卡）-> **FP32 归约**（`ncclFloat`）；
- `ne >= 131072` -> **先压成 BF16 再归约**（`ncclBfloat16`），再转回 FP32。
- M=8 时 FFN 的 17408 宽张量 = 139264 元素 **正好越过 131072** -> 走 BF16 压缩归约。
- => **换 NCCL 会改变数值**（归约顺序 + 部分 BF16 精度），`greedy sha256` 预计与基线不同
  => 按 Goal 第 4 项走"AL + ppl 对比（ppl 相对变化 <= 0.1%）"，不是逐位一致。
  （ppl 语料已固定：`/root/ppl-corpus.txt`，336180 B，md5 `6737ebfc032119d7daa00ae5046252c2`。
  ppl 脚本 `/root/ppl-ab.sh`，`-c 512 --chunks 64`，两侧同语料同参数，只换 allreduce 实现。）
- 另：`GGML_USE_NCCL` 编译进来后，`ggml-cuda.cu:606-609` 会把 VMM 缓冲的 peer access 强制打开
  （等价于我们已经在用的 `GGML_CUDA_P2P=1`）=> 这里不引入新变量。

### 12.6 实测结果（NODROP，同 harness v2，3 prompt，seed 42，全部同会话交错）

| 臂 | split | 卡 | 中位数 t/s | AL | ms/轮 = AL/tg | greedy sha256 |
|---|---|---|---|---|---|---|
| `ab-bf-tp3` | tensor | 0,1,2 | 78.20 | 5.21/4.11/6.08 | 66.6 / 64.1 / 66.2 | `69207026ca…` |
| **`ab-nccl-tp3`** | tensor | 0,1,2 | **94.18** | 5.55/4.22/6.38 | **58.9 / 55.9 / 59.2** | `f3edac1944…` |
| `ab-nccl-tp4` | tensor | 0,1,2,3 | 65.66 | 4.19/3.93/6.08 | 63.8 / 61.8 / 64.6 | `ccc284e4ec…` |
| `ab-nccl-tp6` | tensor | 0-5 | 70.28 | 5.01/3.98/6.29 | 71.3 / 69.9 / 72.9 | `69207026ca…` |
| `ab-bf-layer` | layer | 0,1,2 | 74.52 | 5.16/5.21/6.62 | 71.7 / 69.9 / 71.2 | `ccc284e4ec…` |
| **`ab-nccl-layer`（对照）** | layer | 0,1,2 | **74.71** | 5.16/5.21/6.62 | 同左 | `ccc284e4ec…` |
| `env-plain`（NCCL 复现） | tensor | 0,1,2 | 94.02 | — | — | — |
| `env-tp2`（NCCL） | tensor | 0,1 | 80.38 | — | — | — |

**四条结论**
1. **NCCL 在 TP3 tensor 上 +20.4% t/s**（78.20 -> 94.18）；**按每轮 ms（66.6 -> 58.9）= −11.6%**（AL 变了 5.21->5.55，
   所以**t/s 不是干净指标，ms/轮才是**）。
2. **layer 模式是完美对照**：layer **没有 allreduce**，NCCL 前后 **74.52 vs 74.71（+0.25%，噪声内）且 greedy sha256 完全相同**
   ⇒ 证明 tensor 的收益**确实来自 allreduce**，NCCL 没有偷偷改变别的东西。（这也是"单变量"的行为学证据，因为两版二进制不是逐位可复现的。）
3. **AL 与 greedy hash 是"每个 build 确定、可精确复现"的**：`env-plain`/`env-debug` 两次独立启动都得到 AL 5.55/4.22/6.38 与同一个 hash `f3edac1944…`；
   bf-tp3 两次都得到 5.21/4.11/6.08 + `69207026ca…`。**所以 AL 不是"随机漂移"，而是换了 allreduce 实现后数值真的变了。**
   （⚠️ 推翻我 §5/§0.15 里"AL 会随机漂移、不可比"的粗结论：那次的 5.21 vs 4.96 应是**不同 prompt/配置**所致；
   这里同配置同 prompt 下 AL **逐位复现**。）
4. **TP3 仍是最优卡数**（NCCL 下 TP4 65.66 / TP6 70.28 / TP2 80.38，都不如 TP3 94.18）：
   NCCL 的 ring 延迟随 N 增长，抵消了"每卡字节更少"的收益。

**NCCL 真的在用快路径（`NCCL_DEBUG=INFO` 实证）**：`NCCL version 2.29.7+cuda12.4`、
`Check P2P Type isAllDirectP2p 1 directMode 1 isAllCudaP2p 1`、`Channel 00/0 : 0[0] -> 1[1] via P2P/direct pointer`、
`Connected all rings, use ring PXN 0 GDR 1`、`8 coll channels`；两次 `ncclCommInitAll`（target + draft 两个 context）。

**数值为何会变**（Goal 第 4 项因此走 AL+ppl 而非逐位一致）：见 §12.5 的 FP32/BF16 归约分界。

### 12.7 顺手确证的两条"此路不通/已排除"
- **`--spec-draft-device` + `--split-mode tensor` 的崩溃根因（已定位，不是 allreduce 问题）**：
  `ggml/src/ggml-backend.cpp:941: pre-allocated tensor (output.weight) in a buffer (Meta()) that cannot run the operation (NONE)`
  -> `ggml_backend_sched_split_graph` -> `llama_context::graph_reserve` -> `sched_reserve` -> `common_speculative_init_result`。
  即**给 draft 单独指定设备时，`output.weight` 被落到 Meta 缓冲**，调度器无 op 可用 -> `ggml_abort`。
  日志里那条 `common_fit_params: ... not implemented for SPLIT_MODE_TENSOR, abort` 其实是**被捕获后降级的 warning**（正常也会打印），
  **不是**崩溃原因（我此前记错了）。⇒ 想让 draft 避开 allreduce，需要在 common/src 层修这个放置问题，属新工程。
- `fit_params` 默认 `true`（`common/common.h:476`），但 tensor 模式下 `common_params_fit_impl` 会抛异常并被降级为 warning
  （`common/fit.cpp:184`），**不影响**运行 —— 所以 `-fit off` 不必加。

## 13. 第二个候选：Volta 的 Q8_0 在 ne11=8 走了未被调过的 MMVQ 分支

### 13.1 代码事实（本次核对）
- `mmvq.cuh:3`：`#define MMVQ_MAX_BATCH_SIZE 8`。
- `mmvq.cu:375-385` 的 Volta 分支**只对 K-quant 特判**：
  ```cpp
  case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
      return ne11 <= MMVQ_VOLTA_MAX_BATCH_SIZE_K;   // 我加的 C5，值 4
  default:
      return ne11 <= MMVQ_MAX_BATCH_SIZE;           // = 8
  ```
  **Q8_0 不是 K-quant ⇒ 落 `default` ⇒ `ne11 <= 8` 为真 ⇒ 投机验证批（M=8）走 MMVQ**。
  而 dispatch 顺序是 `mmf -> mmvq -> mmq -> cublas`（`ggml-cuda.cu:1856-1877`）⇒ **MMQ 根本没被考虑**。
- `mmq.cu:334`：`return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;`
  `fp16_mma_hardware_available(700)=true`（`common.cuh:326`：NVIDIA && cc >= VOLTA）⇒ `ne11 < 64` ⇒ **MMQ 本身接受 ne11=8**。
- `mmq.cu:266-307`：MMQ 支持 Q8_0；且要求 `smpbo >= 48 KiB`（V100 支持到 96 KB，满足）。

### 13.2 为什么怀疑现在的选择是错的
- **上游自己就做了同样的事**：C5 已经证明 Volta 上 **K-quant 在 ne11=8 时 MMQ 快**（+2.6%），并把门槛降到 4；
  只有 legacy dp4a 型（Q8_0 等）还留着默认 8 —— **没人给 V100 调过**。
- **外部实测曲线佐证**：anyei fork（2×V100，27B）步时 **8 tok = 69.6 ms vs 12 tok = 65.8 ms**
  ⇒ **12 token（走 MMQ）比 8 token（走 MMVQ）还快**，正是 M=8 这个点上的错误选择。
- M=8 恰好是**本项目口径里唯一重要的形状**（DFlash2 n=7 ⇒ M=8），错在这里的代价直接体现在验收数字上。

### 13.3 已落地的改动（等 A/B 判定）
- `mmvq.cuh`：新增 `#define MMVQ_VOLTA_MAX_BATCH_SIZE_LEGACY 4`。
- `mmvq.cu`：Volta 分支加 `case GGML_TYPE_Q8_0: return ne11 <= MMVQ_VOLTA_MAX_BATCH_SIZE_LEGACY;`
  ⇒ 只影响 **Volta + Q8_0 + ne11 ∈ 5..63**（即投机验证批），M=1 decode 与 prefill 大 batch 都不动。
- 实验：`build-nccl` 原地重编（cmake 配置不变，**唯一变量 = 这两处源码**）→ 快照 `/root/libdir-nccl-q8mmq`，
  与 `/root/libdir-nccl` 做 NODROP A/B（TP3 tensor，3 prompt）。
- ⚠️ **数值可能变**：MMVQ 用 `q8_1` 激活量化、MMQ 用另一套 block 布局 ⇒ 可能要回到"AL + ppl"验收（同 §12.5）。

### 13.4 若成立，下一步（都在"便宜"一侧，非 HMMA）
`mmq.cuh:252-253,279-280`：Volta 的 MMQ 配置被**直接路由到 Ampere 表**（`ggml_cuda_mmq_get_config_ampere`，**没有 Volta 表**）。
Ampere 表按 `(type, J, fallback)` 给出 `I/J/nthreads/J_tile/sram_layout/K_vram/occupancy`，
而 Ampere 的 smem 上限（164 KB）≠ V100（96 KB）⇒ **给 Volta 写一张自己的 MMQ 配置表**（`mmq-config-volta.cuh`）
是 QWEN.md §0.3 一直列的第 1 优先方向，且不需要新 kernel，只需按 V100 实测调 tile 参数。

### 13.5 实测结论：假设**被推翻**，已回退
| 臂（TP3 tensor, NODROP） | 中位数 t/s | AL |
|---|---|---|
| MMVQ@8（现状，`/root/libdir-nccl`） | **93.5 / 93.54** | 5.55/4.22/6.38 |
| MMQ@8（打上本改动，`/root/libdir-nccl-q8mmq`） | **64.47 / 64.93** | 4.37/3.90/6.38 |

⇒ **MMQ 在 ne11=8 上比 MMVQ 慢约 31%**，与 K-quant 的情况**相反**。anyei 那条"12 tok 快于 8 tok"**不能迁移**到
本项目（不同量化/不同配置）。**结论：`ne11 <= 8 -> MMVQ` 在 Volta+Q8_0 上是正确选择，此路封闭。**
两处源码改动**已在本地与服务器双向回退并核对**（服务器 `mmvq.cu`=ffe63a60…、`mmvq.cuh`=dfd78c4f…，
其 CRLF 形态与本地 md5 逐位一致；grep `MMVQ_VOLTA_MAX_BATCH_SIZE_LEGACY` = 0）。

## 14. 最终正式测量（同一 `/root/p60-ab-harness.sh` + `drop_caches` + 3 卡 0/1/2 + `--split-mode tensor`）

| | 臂 A（butterfly，基线） | 臂 B（NCCL，本轮交付） |
|---|---|---|
| `libggml-cuda.so` md5 | `ed1b2f7fe61e300040d82ff392cfc15e` | `faf9cc468a0a263997e15c7f885c75b9` |
| 每 prompt tg | 80.42 / 63.47 / 92.81 | **95.20 / 78.21 / 112.07** |
| **中位数** | **80.42** | **95.20（+18.4%）** |
| AL | 5.21 / 4.11 / 6.08 | 5.55 / 4.22 / 6.38 |
| greedy sha256 | `69207026ca…` | `f3edac1944…`（数值变了 ⇒ 走 AL+ppl 验收） |
| `LLAMA_SPEC_TIMING` | selector 4.37 ms/轮、draft_decode 14.40 | 同量级 |
| `LLAMA_ROUND_TIMING` | enqueue 7157848 us / 302 rounds = **23.7 ms/轮** | — |
| health ok | after 270 s（冷 NFS） | — |

**相对最初的未改动基线（`/root/goal-baseline.txt` 的 55.95）＝ +70%**；**相对同代码的 butterfly 臂＝ +18.4%**。

**正确性（Goal 第 4 项的 fallback，因 NCCL 改归约顺序导致 greedy 不一致）**：
同语料 `/root/ppl-corpus.txt`（336180 B，md5 `6737ebfc…`）、同参数（`-c 512 --chunks 64`、TP3 tensor、FA on、KV q8_0）：

| 臂 | libggml-cuda | 日志自证 | PPL |
|---|---|---|---|
| bf | `ed1b2f7f…` | `NCCL not compiled in` → butterfly | **3.7745 +/- 0.06409** |
| nccl | `faf9cc46…` | （无 fallback 警告） | **3.7751 +/- 0.06410** |

⇒ **PPL 相对变化 +0.016%，远小于 0.1% 门槛** ✓。两臂的 `llama-perplexity` 启动壳 md5 **完全相同**（`5005b9d3…`）
⇒ 唯一变量是库（内核/通信实现）。

**仍未达成**：≥110 tok/s（现 95.20，差 +15.5%）。剩余量级全部落在 target 的 M=8 校验前向上（~40 ms/轮，
是带宽下限 10.8 ms 的 3.7×），已排除的便宜路径见 §8/§9/§12/§13。

## 15. 为什么"把 draft 挪到单卡"这条路走不通（根因已定，勿再试）

**动机**：`LLAMA_SPEC_TIMING` 显示 draft 前向在 **tensor 模式 13.25–13.35 ms/轮**，而 layer 模式只有 **4.44 ms/轮**
（1.14 GB 的 draft ⇒ 86 GB/s，远低于带宽）⇒ 怀疑 ~9 ms/轮 是 draft 白付的 tensor-split allreduce；
若能像 layer 那样跑，round 59 → ~50 ms ⇒ 约 110 t/s，**正好补上缺口**。

**做法**：`--spec-draft-device CUDA0`（以及 CUDA1）。`common_base_params_to_speculative()`
（`common/speculative.cpp:2772-2779`）本来就有针对单卡的兜底：draft 只给一个设备时把 `split_mode` 强制成 `LAYER`
（注释："a draft pinned to a single device doesn't need the meta wrapper an inherited -sm tensor would give it"）。

**实测：两个方向都仍然 `Aborted (core dumped)`**，且报错与 §12.7 那次同源：
```
ggml-backend.cpp:941: pre-allocated tensor (output.weight) in a buffer (Meta()) that cannot run the operation (NONE)
  <- ggml_backend_sched_split_graph <- llama_context::graph_reserve <- resolve_fused_ops <- sched_reserve
  <- llama_context ctor <- llama_init_from_model <- common_speculative_init_result ctor
```
（发生在**加载 draft** 之后、为 draft 建 context 时。）

**根因（上游源码自己写明，非推测）**：`src/models/dflash.cpp:172-175`
```cpp
// optional: reduced-vocab drafts ship their own lm head, full-vocab drafts can share the target's via ctx_other
// a draft with its own embeddings + head references no target tensors and can run on devices the target does not use (e.g. -devd with a tensor-split target)
output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab_draft }, TENSOR_NOT_REQUIRED);
```
我们的 `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` 是**全词表 draft、lm head 借用 target 的**（`TENSOR_NOT_REQUIRED`
⇒ 该 GGUF 里根本没有 `output.weight`）。一旦用 `-devd` 把 draft 放到另一组设备，两个 context 不再共享设备集，
借来的 head 无处安放 ⇒ 落到 **`Meta()` 占位缓冲** ⇒ 调度器无法为其派生算子 ⇒ `ggml_abort`。
⇒ **`--spec-draft-device` 只对"自带 embedding + 自带 head"的 reduced-vocab draft 成立**；本模型不满足。
（不是 bug、不是配置问题；换 draft 模型或给它补一个专属 head 才可能成立，属模型侧工程。）

同时确认的正面结果：`--spec-draft-device` 缺席时系统稳定复现 **96.60 / 96.35 t/s**（两次独立，
AL 5.55/4.22/6.38 与 greedy `f3edac1944…` 完全相同，selector 4.21、draft_decode 13.25/13.35），
说明当前配置是**可重复**的。

## 16. 本期已封闭的杠杆清单（勿重复）

| 杠杆 | 结果 |
|---|---|
| 启用 NCCL（构建开关） | **+18.4%**（80.42 → 95.20 正式；96.6 NODROP）—— 唯一的大赢 |
| NCCL 旋钮（Ring/LL/1ch/CTAS/BUFFSIZE…） | 默认最优，无一超过 |
| 卡数（TP2/TP3/TP4/TP6） | TP3 最优 |
| Volta Q8_0 在 ncols_dst=5..8 的 nwarps 扫描 | 中性（ms/轮 相同） |
| 把 Q8_0 在 ne11=8 改走 MMQ | **−31%**，已回退 |
| `--spec-draft-device` 让 draft 免 allreduce | **不可行**（§15，draft 无专属 head） |
| CPU selector 并行 | 已落地 23.45 → 4.21 ms/轮 |
| target 主机侧图工作（build/alloc/setin） | 2.5 ms/轮，仅 4% |
| KV/采样/回滚/checkpoint | 全部 ~1 ms 级，已排除 |
| **唯一剩余量级杠杆** | **target M≈8 前向用 Volta FP16 Tensor Core（HMMA m8n8k4）** —— 多日级 kernel 工程，外部先例（anyei fork）结论同：high effort / uncertain payoff |

## 17. Goal 2（HMMA 路径）侦察结论 —— 前提已量化证实，但实现是多日工程

### 17.1 ★上限证据：M≤8 的量化 GEMM 在 V100 上离带宽屋顶 2.1×（`test-backend-ops perf`，单卡，无需模型）

命令：`/root/libdir-nccl/test-backend-ops perf -o MUL_MAT -b CUDA0`（日志 `/tmp/mb-mulmat.log`，521 行）

| 形状 (type_a, m=权重行, n=token, k) | µs/run | TFLOPS | 按字节折算有效带宽 |
|---|---:|---:|---:|
| q4_K, m=14336, **n=1**, k=4096 | 46.73 | 2.51 | **729 GB/s** |
| q4_K, m=14336, n=2, k=4096 | 52.74 | 4.45 | — |
| q4_K, m=14336, n=4, k=4096 | 78.04 | 6.02 | — |
| q4_K, m=14336, **n=8**, k=4096 | 97.09 | 9.68 | **340 GB/s** |
| f16, m=4096, n=1, k=14336 | 137.45 | 854 GFLOPS | 854 GB/s（同样带宽受限） |

**两条硬结论**：
1. **n=1 时两条路都贴着带宽屋顶**（q4_K 729 / f16 854 GB/s，峰值 ~900）⇒ **M=1 decode 上 Tensor Core 帮不上忙**（与 §0.12 的物理判断一致）。
2. **n=8 时量化路掉到 340 GB/s**（同一份字节在 n=1 能跑 729）⇒ **M≤8 这段有 ~2.1× 的屋顶空间**，而且缺的**不是 MMA 吞吐**：n=8 的实际 HMMA 数学量只需约 7.5 µs，而实测比 n=1 多花了约 50 µs。
   ⇒ 这 50 µs 就是"多出来的 7 列"造成的 per-column 开销（dp4a/反量化/重复取数），正是"反量化一次 + 用 mma 复用 8 列"能拿回来的部分。

### 17.2 1cat 的可用结论（`docs/design/sm70_awq_small_n_hmma_operator.md`，已读全文）

- 他们的 M=5 HMMA 基线**慢的原因不是带宽也不是 MMA**，而是**网格过小 + shared 冲突**：stock `8x256x64` 对 72 个 SM 只发 **68 个 CTA**（占用率 6.25%），A tile 行距 128 B 导致 8 行撞同一 bank group（约 210 万次冲突）。
- **被接受的修法是纯结构性、不改数学**：`CTA_N=32/64`（68 → 272 CTA）＋ **无冲突 A 共享布局** `SmemLayoutV2<8,64,8,64,Swizzle<3,3,3>>`（`p' = p xor ((p & 0x1c0) >> 3)`）＋ **两寄存器全局 lookahead**。结果：**11.3419 → 9.0439 ms（−20.26%）**，且 **236/236 逐位一致**（保留 FP32 累加/K 序/split-K 契约）。
- 端点收益 **+4.795%**（95.9467 → 100.5474 tok/s）—— 注意**端点传输率只有微基准的 82.6%**（图关键路径重叠）。
- 反面清单（勿重试）：N32 无 swizzle 只有 1.1%；N32+swizzle 比 N64 慢；把 gate/up 改 split 2 虽快但产生 530/87040 个 1-ULP 差异 ⇒ **破坏逐位契约，否决**；对 MLP down 强上 N64 反而回退。

### 17.3 `mma.m8n8k4` 的 fragment 布局：已在本机**实测确定**（不靠文档）

工具：`/root/hmma_probe`（第一次：确认指令语义 `{1,2,4,8}·ones=15`）、`/root/hmma_layout`（one-hot 探针，256 组）。
`mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`：**A = 4 halves、B = 4 halves、D = 8 个 f32 每线程**（真机确认）。

实测映射（lane = threadIdx%32；reg 0..7）：
- **A 探针**（A 某 half 置 1、B 全 1）：A-lane `sel` 恰好点亮 **8 个 D 槽**：
  lanes `{4*(sel>>2) + (sel&1), 该值+2}`，寄存器组 = `{0,1,4,5}`（当 `((sel>>1)&1)==0`）否则 `{2,3,6,7}`。
- **B 探针**（B 某 half 置 1、A 全 1）：B-lane `sel` 点亮 lanes `{sel&14, +1, +16, +17}`，
  寄存器组 `{0,2}`/`{1,3}`（低半）或 `{4,6}`/`{5,7}`（高半），由 `sel&1` 与 `(sel>>4)&1` 选择。
- **关键结构性事实**：32 个 A-lane × 8 槽 = **256 = 全部 D 槽位，且 32 组互不相交**
  ⇒ D 是 8 行 × **4 份复制**，**每个 A-lane 独占一份私有复制的 8 个槽**（这解释了"8×8 却要 8 个 f32/线程"的算术疑点）。
- 1cat 的对应实现（可直接参照移植，用户已授权搬运）：
  `lmdeploy/src/turbomind/kernels/core/mma.h::mma_m8n8k4_row_col`、
  `.../gemm/arch/mma_sm70.h::SM70_MMA_884`（含 `thread_offset_C()/static_offset_C()/ReshapeC()`）、
  `.../gemm/arch/smem_copy_sm70.h::SmemCopy_MMA_884_A/B`、`.../gemm/arch/operand_sm70_s884.h`（含 swizzle 版 A operand）。

### 17.4 实现计划（已成形，尚未落地）

1. 把这些 fragment 规则固化成一个小型 `hmma884_volta.cuh`（A/B 装载 + mma + C 读回），先用**独立微基准**对 CPU 参考逐元素校验（本机已验证指令语义，故只剩簿记）。
2. 写 Q8_0 专用 kernel：warp 负责 8 权重行 × 8 token，K 方向按 4 递增；权重驻 smem、寄存器内反量化成 f16 后喂 mma；FP32 累加。
3. 接进 `ggml-cuda` 的 mul_mat 分派（仅 Volta + Q8_0 + ne11 ≤ 8），并在 `mmq.cuh` 侧给 Volta 一张自己的配置表（现在被直接路由到 Ampere 表）。
4. 验收：`test-backend-ops` 形状微基准 → 再按 Goal 口径跑 harness + AL/ppl。

**结论**：前提（有 2.1× 屋顶空间、缺的是 per-column 开销而非 MMA）**已被实测证实**；参考实现、fragment 布局、验收口径都已就位；但**落地是多日 kernel 工程**（含簿记校验、smem 布局与 swizzle、分派接线、正确性与调优两轮）。

## 18. ★★结论修正：上面 §17.1 的"2.1× 屋顶空间"是 q4_K 的假象；**我们的 Q8_0 口径已经没有这个空间**

§17.1 只看了 q4_K。补上同批日志里 q8_0 / q4_0 的数据（`/tmp/mb-mulmat.log`，按"权重字节 ÷ µs"折算有效带宽）：

| 权重类型 | n=1 | n=4 | n=8 | n=8 走哪条路 | n=8 效率（相对自身 n=1） |
|---|---:|---:|---:|---|---:|
| **q8_0（我们的口径）** | **757 GB/s** | 683 GB/s | **555 GB/s** | MMVQ | **73%** |
| q4_0 | 729 GB/s | 581 GB/s | 359 GB/s | MMVQ→MMQ | 49% |
| q4_K | 729 GB/s | 423 GB/s | 340 GB/s | MMQ（**Ampere 配置**） | 47% |

**三条修正后的结论**：
1. **n=1 时三种量化都贴着屋顶（729–757 GB/s）** ⇒ 那才是"读这些权重"的物理上限。
2. **我们的 Q8_0 在 n=8 已经是 555 GB/s = 自身 n=1 屋顶的 73%** ⇒ **"2.1× 空间"不适用于我们的口径**，它是 q4_K/q4_0 的（它们 n=8 掉到 47–49%）。真实剩余空间只有 **~1.36×**。
3. 因此 §0.12 的"差距主因 = M=8 GEMM 没用 Tensor Core"这一判断**被实测削弱**：我们的 GEMM 路径本身已接近其屋顶。

### 18.1 HMMA 原型的实测（本机真实实现，非推算）

| 版本 | 设计 | 有效带宽 |
|---|---|---:|
| v1 `wmma_perf.cu` | 朴素：每个 (row,q) 重复读同一个 Q8_0 block（~32× 冗余全局流量） | **8.8 GB/s** |
| v2 `wmma_perf2.cu` | smem 暂存，每个 block 只读一次 | **73 GB/s** |
| v3 `wmma_perf3.cu` | 再加双缓冲软件流水 + 全线程参与暂存 | **90 GB/s** |
| —— 现役 llama.cpp Q8_0@n=8（MMVQ） | —— | **555 GB/s** |

- 形状均为 N=17408、K=5120、8 token、94.7 MB 权重、单卡。
- **正确性已验证**：`wmma_q8_bench.cu` 里 WMMA 16×16×16 探针对 CPU 参考 **max rel err = 0.000e+00（MATCH）**，Q8_0 GEMM 128/128 单元全部产出、误差仅 f16 反量化量级。
- 因此"用 WMMA 绕开 m8n8k4 手写簿记"这条路**技术上走得通且已跑通**，但**性能比现役慢 6 倍**：裸 HMMA 在 n=8 这种极瘦形状上要把 smem 暂存/同步/占用率全部工程化到位才可能接近屋顶，而**即便做到屋顶也只有 1.36×**。

### 18.2 对目标的含义（重要）

- 用 Q8_0 时，target 的 M=8 GEMM **已经在其屋顶的 73%**；把它重写成 Tensor Core 路径，**理论上限约 1.36×**，实测原型为 **0.16×**。
- ⇒ **这条路径不可能把 95.20 推到 ≥110**（需要 +15.5% 的整轮收益，而它只在"GEMM 部分"有 ≤36% 的理论空间，且现实实现远达不到屋顶）。
- ⇒ 与 1cat 的 ~4× 差距，**不能归因于"M=8 GEMM 没用 Tensor Core"**；结构性差异更可能来自：**口径钉死 Q8_0（9.7 GB/卡/轮）vs 他们 4-bit（≈3.4 GB/卡/轮）、融合管线、以及更高的接受长度**。
- `mma.m8n8k4` 的 fragment 布局仍已在 §17.3 实测确定，WMMA 路线也已跑通——**这些资产保留**，但**不建议再投入多日工程**去追这条已被实测限定到 ≤1.36× 的路。

