# HANDOFF — llama.cpp V100 / SM70 专项优化项目交接

日期：2026-09-20（周日）
交接原因：用户要把执行体从 Qwen Code 迁移到 DeepSeek 专用 dsh。
本文件是**唯一的权威交接文档**；开始工作前先读本文件，再读 `FINAL-REPORT.md`。
持久规则在**工作区根** `F:\vllm+llama.cpp\QWEN.md`（每次会话自动加载，含红线与度量纪律），冲突时以 QWEN.md 为准。

---

## 0. 30 秒速览

- **北极星（不变）**：让 `llama.cpp` 在 V100 上**追平 1cat-vLLM 的速度与效果**。目标是改 `llama.cpp/`，`vllm/`、`1cat-vllm/` 只是**抄作业的参考项目**。
- **对标实测值**：1cat 生产服务 `vllm-1cat.service` 实测 **221.6 / 230.8 / 263.2 tok/s**，AL 4.06-5.21，**17.463 ms/轮**（单流，短上下文，DFlash2）。⇒ 差距约 2.3 倍（我们现在 95.20）。
- **我们现在**：固定口径单流 decode 中位数 **95.20 tok/s**（正式、带 drop_caches），**NODROP 最好 96.60**（TP3 0,1,2）。起点基线 **55.95** ⇒ **+70%**。
- **三项已落地且已验证正确性的提速**：① 并行化 CPU selector（+26.8%）；② `GGML_CUDA_P2P=1`（+10.6%）；③ **把 NCCL 编进来**（+18.4%）。
- **已排除的头号假设**：把 MMQ 的 dp4a 换成 Volta FP16 HMMA。实测**这条路的天花板只有 ~1.36×**，够不到 110（详见 §6）。**不要再从零重做这条**。
- **下一步已定**：Goal 8dbd30ea，**第一步量化 M=8 验证轮里 GDN 与 attention 的成本占比**（一次 ncu 已经跑过但结果没读，见 §10）。
- **当前服务状态**：`vllm-1cat` 与 `llmscope` 已 **stop**（我操作腾卡的）。恢复：`systemctl start vllm-1cat llmscope`。

---

## 1. 北极星目标与验收口径

**最终目标（用户 2026-09-20 明确，不随迭代改变）**：
> 让 `llama.cpp` 在 V100 上追平 1cat-vLLM 的速度与效果。

推论（务必记住）：
- **1cat-vLLM 的实测数字是天花板参考**；`llama.cpp` 自己跟自己的 A/B 只是过程指标，不是终点。
- **交付物必须落在 `llama.cpp/` 内**（用户原话："我们要开的是llama.cpp分支而不是vllm"）。
- **用户已放宽"禁止搬运代码"**：原话"1cat-vLLM 的 csrc/sm70_turbombind 884 tile 是参考思路，但如果有直接可用的代码 可以直接搬运到llama.cpp"（注意：仍需逐行能理解、能维护，这是 AGENTS.md 的硬要求）。

**固定基准口径（不动这个口径去抬数字）**：
- 模型 `Qwen3.8-27B-Q8_0.gguf`（29,047,086,048 B，sha256 `a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348`，ModelScope `unsloth/Qwen3.8-27B-GGUF`）
- draft `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`（官方基座 draft），`--spec-type draft-dflash --spec-draft-n-max 7`
- ctx 8192，官方采样，seed 42，`--split-mode tensor`，卡数**动态决定但必须记录**
- 度量脚本 `/root/p60-ab-harness.sh`；正式数字必须带 `drop_caches`

---

## 2. 环境（服务器 / 编译 / 凭证）

**AC922 开发测试服务器**（所有编译 / 测试 / bench 都在这里，不是本地 Windows）

- 连接：`ssh -o BatchMode=yes root@192.168.50.235`（本机已授权 key，非交互）
- 架构 **ppc64le（IBM POWER9，176 核）**，AlmaLinux 8.10，kernel 4.18.0-553
- **6× Tesla V100-SXM2-16GB**，driver 550.54.15，**CUDA 12.4**（`/usr/local/cuda-12.4/`）；无 Rust、无 docker
- **无盘网络启动**（`/` 是 NFS `192.168.50.84:/mnt/IBM-AC922/rootfs`）⇒ **下载 / 加载模型 / 编译吃同一张网卡，不要并行**（用户明确说"可以多等等"）
- V100 HBM2 被映射进系统内存（PPC64LE 特性）⇒ 页缓存会吃显存，测前要 `sync; echo 3 > /proc/sys/vm/drop_caches`
- 性能工具**不在 PATH**，必须全路径：`/usr/local/cuda-12.4/bin/ncu`、`/usr/local/cuda-12.4/bin/nsys`

**编译配方**（系统 gcc 8.5 会失败，必须 gcc-toolset-12）：
```sh
CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc \
CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++ \
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
  -DCMAKE_INSTALL_RPATH=/root/llm/llama.cpp/lib64
cmake --build build --config Release -j82     # 空载时从 -j82 起步
```
**编 NCCL 版必须额外加**（`find_library(NAMES nccl)` 找不到，因为包里只有 `libnccl.so.2`、没有 `libnccl.so`）：
```sh
  -DNCCL_INCLUDE_DIR=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/include \
  -DNCCL_LIBRARY=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/lib/libnccl.so.2
```

**源码 / 构建位置**：
- 本项目源码+构建：`/root/llm/test/v100-opt/llama.cpp`（build 目录 `build/`、`build-nccl/`）
- ⚠️ `/root/llm/llama.cpp/bin/*` 与 `/root/llm/test/llama.cpp/build/bin/*` 都是**上游 `0.4.0-dev 434ddbb`，不是 b11053** —— 别拿它们当本项目 binary
- 模型：`/root/llm/models/`（`Qwen3.8-27B-GGUF/`、`Qwen3.8-27B-DFlash2-GGUF/`、`Qwen3.8-27B-TurboFCFusion-gguf/`）

**红线（只两条，其他全放开了）**：
1. **不要改** `/root/llm/systemd/llama-server.service`（原生产 unit，用户要求原样保留）
2. **不要改** `vllm/`、`1cat-vllm/`、`vllm-forkpoint/` 三个目录的内容（只读对照）

6 张 V100 全部归本项目，可自主 `systemctl stop/start`、kill、抢占、重编，无需逐次同意。
（`vllm-1cat.service` / `llmscope.service` 恢复命令 = `systemctl start vllm-1cat llmscope`；`new-api.service` 不占显存，一直在跑。）

**拓扑（`nvidia-smi topo -m`）**：GPU 0/1/2 在 NUMA0，两两 **NV2**；GPU 3/4/5 在 NUMA8；**跨组是 `SYS`**（走主机内存）。
⇒ **只有 3 张卡能全直连 P2P**。NCCL 实测确认：TP3 `isAllDirectP2p 1`，TP4 `isAllDirectP2p 0`。
（用户明确说卡号分配是随手填的，0,1,2,3 与 0,1,3,4 **没有区别** —— 别把卡号当变量，把**同组/跨组**当变量。）

---

## 3. 代码现状（已提交什么）

### 3.1 `F:\vllm+llama.cpp\llama.cpp`（本项目 fork）

- **HEAD = `79504e72b`**，`git status --porcelain` **为空（干净）**
- parent = `1af554f8f` = 上游 **b11053**（base 钉在 b11053，不跟 master）
- 相对 b11053 的改动：**13 个文件，+587 / −33**，内容分四类：

| 类别 | 文件 | 说明 |
|---|---|---|
| **C4** Volta MMVQ 参数 | `ggml/src/ggml-cuda/mmvq.cu` (+35/−1), `mmvq.cuh` (+4) | `MMVQ_PARAMETERS_VOLTA`：V100（CC 700）ncols=1 → nwarps=2；`MMVQ_VOLTA_MAX_BATCH_SIZE_K 4`。tg128 36.23 → 37.52 |
| **C5** K-quant 交叉点 | 同上（`ggml_cuda_should_use_mmvq`） | V100 的 mmvq↔mmq 交叉点 = **4**（上游独缺 Volta，落到默认 8）。ne11=8 +2.6% |
| **PR #27858** DFlash2 CPU selector | `common/speculative.cpp` (+386), `common/speculative.h` (+16), `common/common.cpp` (+6), `src/llama-ext.h`, `src/llama-model.cpp/.h`, `src/models/dflash.cpp` | 修掉 `tensor + draft-dflash` 的硬崩（`ggml-backend-meta.cpp:543`）；**`build_dflash2_selector_cpu()` 已按 8 个 block 位置并行（8 线程）** → 23.45 → 4.29-4.41 ms/轮，逐位一致 |
| **计时量具**（`LLAMA_ROUND_TIMING` / `LLAMA_SPEC_TIMING` / `[RT]` 探针） | `src/llama-context.h/.cpp`, `common/sampling.cpp`, `tools/server/server-context.cpp` | env-gated、未设时零开销 |

⚠️ **树里目前留有 7 处 `[RT]` `fprintf(stderr, ...)` 调试探针**（`llama-context.cpp` 4、`server-context.cpp` 2、`sampling.cpp` 1）。下一步可以清理掉（另开一个 commit）。

### 3.2 `F:\vllm+llama.cpp\1cat-vllm-v100-study`（研究档案，独立的 git 仓库）

- 本会话开始时 HEAD = `8a18a5b`；本次交接新增提交见 §3.3
- 这是**所有结论的存档**：`FINAL-REPORT.md`（总交付）、`goal-phase1-findings.md`（§1-§18，最全）、`premise-check-1cat-vs-llamacpp.md`、`phase0-round-breakdown.md`、`1cat-sm70-gemm-tactics.md`、`c5-volta-crossover.md`、`same-source-ab.md`、`dflash-in-llamacpp.md` 等

### 3.3 本次交接的提交

见本文档末尾 §12 的提交清单（`archive : ...`）。

---

## 4. 实测成果

### 4.1 总账

| 阶段 | 配置 | 中位数 t/s | 备注 |
|---|---|---|---|
| 起点基线 | 原版 b11053，TP tensor | **55.95** | `/root/goal-baseline.txt`（layer 70.09） |
| + 并行 selector | 同上 | 70.92 | +26.8%，逐位一致 |
| + `GGML_CUDA_P2P=1` | 同上 | ~78.4 | +10.6% |
| + **NCCL** | 同上 | **95.20** | **正式口径（带 drop_caches）**；NODROP 94.18 / **96.60**（TP3 0,1,2） |

**总计 55.95 → 95.20 = +70%**（对照最初的未改动树）。

### 4.2 三项已落地提速（都验证过正确性）

**① CPU selector 并行化**（`common/speculative.cpp::build_dflash2_selector_cpu()`）
- 这是 PR #27858 引入的、**只在 `--split-mode tensor` 下启用**的 CPU 侧 selector：每轮对 8 个位置做**全词表 top-k**（`n_vocab = 248,320`，padded 词表）+ gate 矩阵乘，rank=72，**单线程 POWER9 上 23.4-23.5 ms/轮**。
- 改法：按 8 个 block 位置分 8 线程并行，每位置的算术完全不动。⇒ **23.45 → 4.29-4.41 ms/轮**，tensor 中位数 **55.95 → 70.92（+26.8%）**。
- 正确性：**AL 完全相同**，`temperature=0` 输出 **sha256 逐位一致**。

**② `GGML_CUDA_P2P=1`**（+10.6%）
- `ggml-cuda.cu:391` 的 `GGML_CUDA_P2P` **默认未设置**，而它才是真正调 `cudaDeviceEnablePeerAccess` 的开关。

**③ 把 NCCL 编进 `libggml-cuda.so`**（**+18.4%**，单次最大收益）
- 根因：`build/CMakeCache.txt` 里 `GGML_CUDA_NCCL:BOOL=ON`（默认就是 ON）但 `NCCL_LIBRARY-NOTFOUND`、`ldd` 没有 libnccl ⇒ 初始化链 **nccl → internal → none**（`ggml-cuda.cu:1208-1245`）直接掉到最后一级 **butterfly（经主机内存中转）**。
- 而 `allreduce.cu:399-403` 的快速内部 allreduce 在 `n_devices != 2` 时**直接 return nullptr** ⇒ 3 卡以上全部走 butterfly 并**逐层**付费。
- **本机其实已经有 NCCL 2.29.7**（就在 1cat 的 venv 里，见 §2 的路径），**无需下载**。
- 结果：**78.20 → 94.18（NODROP）**、**80.42 → 95.20（正式，带 drop_caches）**。

### 4.3 数值验收

- `llama-perplexity` 门：**3.7745（butterfly）→ 3.7751（NCCL）= +0.016% ≤ 0.1%** 通过
- 两侧 launcher shell **逐字节相同**（md5 `5005b9d3...`）；语料 `/root/ppl-corpus.txt`（336180 B，md5 `6737ebfc032119d7daa00ae5046252c2`）
- ⚠️ **NCCL 会改数值**：`ggml-cuda.cu:1000-1072`，3 卡时 `ne < 131072` 走 FP32、`>= 131072` 压成 **BF16** 归约；M=8 的 FFN `17408x8 = 139264` 正好越过门槛。所以正确性验收走 **AL + ppl**，不是逐位。

---

## 5. 已排除清单（带数字，**不要重复**）

| # | 尝试 | 结果 | 结论 |
|---|---|---|---|
| 1 | NCCL 调参（buffer / channel / algo） | Ring **93.40**、LL **93.44**、1ch **77.67**、TP2 **80.38**，默认 **94.18** | **默认最好**，别再扫 NCCL 旋钮 |
| 2 | 加卡（NCCL，TP tensor，NODROP） | TP3 0,1,2 **96.60**；TP4 0,1,2,3 **67.54**；TP4 0,1,3,4 **72.11**；TP5 **73.09**；TP6 **58.26** | **TP3 最优**。`[RT] target decode+sync` 随卡数恶化：32.4 → 48.6 → 73.0 ms（只有 3 卡全直连 P2P） |
| 3 | Volta 上把 Q8_0 MMVQ 强推 MMQ | **−31%** | 已回退。`mmvq.cu:655` 区域判断在 V100 上 **MMVQ 是对的** |
| 4 | `--spec-draft-device CUDA0\|CUDA1`（配 tensor） | **两臂都 abort** | 根因**非 bug**：`src/models/dflash.cpp:172-175` —— 我们的 DFlash2 draft 是**全词表、借用 target 的 head**（GGUF 里没有 `output.weight`），所以它**必须**跑在 target 用的设备上 |
| 5 | 多形状 decode graph 缓存 | 主机侧合计 **2.10 ms/轮 = 2%**，复用率已 93%（reuse=94 / rebuild=7） | **收益上限 ~2%，放弃**。（外部 fork 说的 "TP alloc 38 ms" 是他们的配置，我们这里是 1.94 ms） |
| 6 | CPU 采样走回 GPU / 别的采样路径 | sampling ~1 ms/轮 | 不值当 |
| 7 | `GGML_CUDA_FORCE_CUBLAS` | 它**不 gate** `should_use_mmvq`；且会给每层每轮加 ~178 MB 反量化写出 | 放弃 |
| 8 | 手写 m8n8k4 fragment 映射 | 只产出**块对角**结果（rows0-3 × cols0-3 有值，rows4-7 全零） | 原因是合成后的 `(asup,bsup)` 对只允许两个对角 4×4 块 ⇒ **改走 WMMA** |
| 9 | WMMA HMMA 原型 | v1 **8.8 GB/s** → v2（smem staging）**73** → v3（双缓冲流水）**89.6-90.2 GB/s** | 对比在任实现 **555 GB/s** ⇒ **慢 6.2×**。**这就是 §6 的结论来源** |
| 10 | nsys / ncu profiling | nsys 两次都拿不到可用报告（`Importer error`）；两次成功的 `.nsys-rep` **不含 CUDA kernel 数据** | 见 §10 的替代方案 |
| 11 | `LLAMA_LOG_INFO` 在 libllama 里埋点 | 连试 3 次**无输出**（原因未查明） | 先用 `fprintf(stderr,...)` 证明函数被执行，再切日志 |
| 12 | `llama-bench` 验证库内埋点 | **一条 libllama INFO 日志都不打印** | 用 `llama-server`（会打印 `slot print_timing:`） |

---

## 6. 根因分析现状 + HMMA 结论（**最重要的修正，别再走错**）

### 6.1 差距在"每轮延迟"，不在草稿质量

| 栈 | AL | ms/轮 | tok/s |
|---|---|---|---|
| 1cat-vLLM FP8 + DFlash2（4×V100 TP4） | 4.06-5.21 | **17.463** | **221.6-263.2** |
| llama.cpp Q8_0 + DFlash2 n=7（3×V100 TP3） | 4.11-6.08 | ~55 | 95.20 |

⇒ **我们的 AL 不输甚至更好**；**我们只是每轮多花 3 倍时间**。抓手 = 每轮前向+验证的效率。

### 6.2 每轮成本分解（92.6 ms/轮那次实测，AL 5.21，55.49 tok/s）

| 成分 | ms/轮 | 占比 | 量具 |
|---|---|---|---|
| target verify（M=8） | **29.87** | 32% | `LLAMA_ROUND_TIMING` 的 `enqueue_us`（M>1 时同步，可信） |
| **CPU selector**（现已优化到 4.4） | 23.44 → 4.4 | 25% → 5% | `LLAMA_SPEC_TIMING` 的 `selector=` |
| draft 前向 | **15.01** | 16% | `LLAMA_SPEC_TIMING` 的 `draft_decode=` |
| target 主机侧图工作（build+alloc+setin） | 2.10 | 2% | build 0.13 + alloc 1.94 + setin 0.03 |
| 候选走查 + 其余 | ~22 | 24% | 余量 |

**draft 前向偏高**：1.14 GB / 15.01 ms ⇒ 有效带宽仅 **~76 GB/s** ⇒ 是 **dispatch / 小 batch 受限**，不是带宽受限。**待查**。

### 6.3 ★ HMMA 假设被实测否定（天花板只有 ~1.36×）

**背景（1cat 的做法是真的）**：`1cat-vllm/csrc/sm70_turbomind/`（vendored lmdeploy SM70 GEMM，"884" = **HMMA m8n8k4** tile + QPN ops）；其设计文档 `docs/design/sm70_awq_small_n_hmma_operator.md` 记录 M=5 形状从 11.3419 → 9.0439 ms（**−20.26%**），收益来自 `CTA_N=32/64` 注册项 + 无冲突 A 共享内存布局 `SmemLayoutV2<8,64,8,64,Swizzle<3,3,3>>`（`p' = p xor ((p & 0x1c0) >> 3)`），**逐位一致**。

**但是**：我们自己的实测（`test-backend-ops perf -o MUL_MAT -b CUDA0`，单卡，日志 `/tmp/mb-mulmat.log`）显示——

| 形状 | dtype | n=1 | n=8 | 结论 |
|---|---|---|---|---|
| m=4096 k=14336 | **q8_0** | 82.40 µs / **757 GB/s** | 112.48 µs / **555 GB/s** | n=8 已是**自身 n=1 屋顶的 73%** ⇒ 真实余量只有 **~1.36×** |
| m=14336 k=4096 | q4_K | 46.73 µs / 706-729 GB/s | 97.09 µs / **340 GB/s（48%）** | 这个才是"低效"的，但它**走 Ampere MMQ config**（两回事） |
| m=4096 k=14336 | f16 | 137.45 µs / 854 GB/s | — | — |
| m=4096 k=14336 | q4_0 | 45.34 / 729 | 91.96 / 359 | — |

**我此前"2.1× headroom"的说法是错的**——它来自 q4_K（n=8 走 Ampere MMQ config）。**我们固定口径的 q8_0 n=8 已经在自身屋顶的 73%**。
⇒ **HMMA 这条路最多 1.36×（55.95 × 1.36 ≈ 76），够不到 110。** 而且我的 WMMA 原型只有 90 GB/s（慢 6.2 倍）。

**M 曲线（3 卡、同 prompt/seed/512 tokens）**：`none`(M=1) **32.35** / `dflash n=3`(M=4) **67.2** / `dflash n=7`(M=8) **92.6 ms/轮**
⇒ 边际 **6.4-11.6 ms/行**（**不是 0**，所以 M=8 不是纯带宽受限）；纯带宽理想每行 32.35/8 ≈ **4.0 ms** ⇒ 实测约为带宽下限的 **2.9×**。
**这 2.9× 里"超出带宽下限"的那部分，才是真正可攻击的计算量。**

### 6.4 ★ 实测出来的 m8n8k4 fragment 布局（省得下一个人再逆一遍）

一条 `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32` **每线程**取 A=4 个 half、B=4 个 half、产出 D=8 个 f32。

- A 的 lane `L` 的寄存器 `{d0,d1,d4,d5}` ← A-lane `4*(L>>2)+(L&1)`；`{d2,d3,d6,d7}` ← 上面那个 `+2`
- B 的 lane `L`：`c0 = 2*((L&15)>>1)`；`{d0,d2}`←c0，`{d1,d3}`←c0+1，`{d4,d6}`←c0+16，`{d5,d7}`←c0+17
- 32 个 A-lane × 8 槽 = 256 = 全部 D 槽且互斥 ⇒ **D 是 8 行 × 4 份私有复制**
- ⚠️ **合成后的对是块对角的**（rows0-3 只与 cols0-3 乘；rows4-7 只与 cols4-7）—— 这就是手写映射只得到 32/64 个输出的原因

**WMMA 在 sm_70 可用，且能绕开所有 fragment 记账**：
```cpp
wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> a;
wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::col_major> b;
wmma::fragment<wmma::accumulator, 16,16,16, float> c;
wmma::load_matrix_sync(a, ptr, ldm);  wmma::mma_sync(c,a,b,c);
wmma::store_matrix_sync(out, c, ldm_out, wmma::mem_row_major);
```
对 `[n][k]`-major 的激活数组，`col_major` 配 `ldm = K` 才是正确配对（我一开始配对了 row-major 的 k-stride，得到 `rel err 1.375e10 → MISMATCH`；改成 `load_matrix_sync(b, B + k, K)` 后 16×16×16 探针 `0.000e+00 → MATCH`）。
Q8_0 GEMM 探针：128/128 单元填满，误差只在 f16 反量化层（8.25834 vs 8.25975）。

---

## 7. 1cat-vLLM 参考要点

### 7.1 它的真实运行配置（只读读到的，非 README 数字）

`vllm-1cat.service`：`CUDA_VISIBLE_DEVICES=0,1,3,4`（**跨 NUMA 组**）、`--tensor-parallel-size 4`、`--dtype half`、`--kv-cache-dtype fp8_e5m2`、`--gpu-memory-utilization 0.905`、`--max-model-len 262144`、`--max-num-seqs 1`、`--attention-backend FLASH_ATTN_V100`、speculative-config dflash。
⇒ `1cat-runtime.env` **没有任何 NCCL 调参**（只有 PATH / 缓存 / `VLLM_NCCL_SO_PATH` 指向同一个 libnccl.so.2）。
⇒ **它 4 卡能跑，不是因为卡数或 NCCL 调参**（我们实测 TP4 反而慢）。**多半是容量驱动**（FP8 权重 27 GB / 4 卡 ≈ 6.75 GB/卡）。

### 7.2 它自己实测过的开关（`docs/design/sm70_qwen38_default_fastpath.md`，可直接借鉴）

| 开关 | C=1 | C=4 | C=8 | C=16 | 结论 |
|---|---|---|---|---|---|
| `VLLM_SM70_TP4_PUSH_ALLREDUCE_CONCURRENCY` | — | — | — | **+3.4%** | 默认 on |
| `VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES` | — | — | — | **+1.0%** | 默认 on |
| 两者一起 vs 都关 | +7.8%* | 0% | −0.9%* | **+4.3%** | 已合并 |
| `VLLM_SM70_USE_BREAKABLE_CUDAGRAPH` | **−29.2%** | **−17.6%** | **−18.8%** | **−12.7%** | **保持关闭** |

`*` 标注的 C=1 / C=8 因为 off/on 范围重叠，不可分辨；只有 C=16 可分离。

**它自己的自研 allreduce** 在 `csrc/custom_all_reduce.cuh`（push-based），两个开关用 `std::getenv` 在 **kernel launch 时**读；`vllm/envs.py` 里的声明当时**没被消费**（所以我们一度以为它没生效）—— **一类和我们 NCCL 问题同源的 bug**。

### 7.3 它的 V100-native 清单（`sm70_qwen38_default_fastpath.md`）

5 个模型级默认开关（只在 Qwen3.8 Flash-Next + NVFP4 + FP16 act/KV + native FP32 SSM + 全 SM70 本地 TP4/PP1 时默认开）：
- `VLLM_SM70_QWEN38_FP16_GEMV`
- `VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16`
- `VLLM_SM70_QWEN38_FUSED_HC_FP16`
- `VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP`
- `VLLM_SM70_MOE_ADD_ALLREDUCE`

依赖项：dual compilation、hybrid PLE（CPU/磁盘 offload）、Model Runner V2、full+piecewise graphs、native RMSNorm。
历史 ~98 tok/s 基线 = FP16 act/KV + native FP32 SSM + NVFP4 experts + TP4/PP1 + 262144 ctx + 8192 prefill chunk + **无 MTP / 无 prefix cache**。

### 7.4 ★ 分数校准（别被 README 数字骗）

- 另一个把 V100 当第一目标的 fork（`github.com/anyei/llamacpp-v100`）**投入大量工程后单流 MTP 只到 66 t/s**，其自己的结论是：*"batch-1 tg headroom on Volta is now mostly exhausted at the orchestration/launch-config level. Remaining gains require sm70 kernel-algorithm work (MMVQ small-batch, fattn-vec) — high effort, uncertain payoff."*
- ⇒ **没有人用 llama.cpp 在 V100 上达到过 1cat 的 200+**。1cat 的优势来自 **decode GEMM 走 Tensor Core（TurboMind HMMA 884）+ NVFP4/FP8 + 融合管线 + 高接受 draft**。
- 上游官方（PR #12098，CUDA 维护者）：*"the MMQ kernels should only be compiled for batch sizes up to MMQ_DP4A_MAX_BATCH_SIZE if FP16 tensor core hardware is available but **int8 tensor core hardware is not (basically only V100s)**."* ⇒ **V100 是唯一"有 FP16 TC、没有 int8 TC"的 N 卡**；`mmq.cuh:190` 的 MMA layout 要 `turing_mma_available(cc)`（≥750），V100(700) 落 dp4a；`mmq.cu:334` 是 `ne11 < 64` 才用 MMQ。

---

## 8. 关键文件 / 脚本 / 快照 / 日志清单

### 8.1 服务器（AC922 `/root/`）

**二进制快照**（同源 A/B 必须整目录 `LD_LIBRARY_PATH` 切换，**不能只复制可执行文件** —— 壳的 `DT_RUNPATH` 指向 build 目录绝对路径）：
- `/root/libdir-rt` —— baseline butterfly 构建
- `/root/libdir-nccl` —— **NCCL 构建**（`libggml-cuda` md5 `faf9cc468a0a263997e15c7f885c75b9`）← **当前最好**
- `/root/libdir-nccl-q8mmq`（已否定的变体）、`/root/libdir-bf-ppl`、`/root/libdir-27858`

**度量脚本**：
- `/root/p60-ab-harness.sh` —— **主力验收脚本**。参数：`CARDS` / `SPLIT` / `TAG` / `NPRED` / `PORT` / `L` / `M` / `D` / `NODROP` / `P2P` / `ARENV` / `DEVD` / `NGLD`；输出 lib md5（含 `libllama-common.so.0.4.1`）、每 prompt 的 tg+AL、`MEDIAN_TG`、`[RT] perf:`、spec timing、greedy sha256
- `/root/p60-ab-harness2.sh` —— 超集（加 `LDEXTRA` / `ARENV2`，打印 allreduce-init 行与 `[RT] target decode+sync`）
- `/root/goal-baseline.txt`（layer 70.09 / tensor 55.95）
- `/root/ppl-corpus.txt`（336180 B，md5 `6737ebfc032119d7daa00ae5046252c2`）

**微基准 / 探针**（`/root/`）：`hmma_probe`、`hmma_layout`、`hmma_map`、`wmma_bench`、`wmma_perf{,2,3}`；源码 `hmma_probe.cu`、`hmma_layout.cu`、`hmma_map.cu`、`hm_q8_bench.cu`、`hm_dbg.cu`、`wmma_q8_bench.cu`、`wmma_perf{,2,3}.cu`
**日志**：`/tmp/p60-*.log`、`/tmp/mb-mulmat.log`（521 行，MUL_MAT perf）、`/tmp/cards2.log`、`/tmp/hmma_layout.out`、`/tmp/ncu-tgt.log`（**未读**）、`/tmp/nccl-ab-summary.log`、`/tmp/final-nccl.log`
**脚本**：`nccl-check.sh`、`nccl-find.sh`、`nccl-pkg.sh`、`nccl-plan.sh`、`build-nccl{,-inner}.sh`、`nccl-ab.sh`、`nccl-env-sweep.sh`、`nccl-knobs2.sh`、`final-nccl.sh`、`ppl-ab.sh`、`ppl-bf3.sh`、`mk-corpus.sh`、`q8mmq-ab.sh`、`revert-q8.sh`、`devd-retest.sh`、`profile-tgt{,2}.sh`、`ncu-tgt.sh`、`cards2.sh`、`collect2.sh`、`final-evidence.sh`

### 8.2 本地（`F:\vllm+llama.cpp\1cat-vllm-v100-study\`）

`FINAL-REPORT.md`（**总交付报告，先读这个**）、`goal-phase1-findings.md`（§1-§18，最全细节）、`premise-check-1cat-vs-llamacpp.md` + `premise-check.html`、`phase0-round-breakdown.md`、`phase1a-report.html`、`stress-test-256k.md`、`l3-acceptance.md`、`c5-volta-crossover.md`、`same-source-ab.md`、`mtp-sampler-cpu.md`、`dflash-in-llamacpp.md`、`dflash2-llama-cpp-research.md`、`baseline.md`、`WORKPLAN.md`、`core-changes.md`、`FlashAttention-V100-reference.md`、`fa-v100-verified.md`、`1cat-sm70-gemm-tactics.md`、`vllm-vs-1cat-vllm-diff.md`、`COMMIT-MSG.txt`

### 8.3 1cat 参考文件（只读）

- `1cat-vllm/docs/design/sm70_awq_exact_m5_batched_gemv.md` —— M=5 形状（`5x17408x5120` 等），每 rank 每验证轮 **236 次**调用
- `1cat-vllm/docs/design/sm70_awq_small_n_hmma_operator.md` —— M=5 被接受的 HMMA 路线（**−20.26%，逐位一致**）与**被否定的备选**（N32 无 swizzle 1.1%、N32+swizzle 比 N64 慢、gate/up 强拆 2 有 1-ULP 差、N64 用在 MLP down 会回退）
- `1cat-vllm/docs/design/sm70_qwen38_default_fastpath.md` —— **§7.2/§7.3 的来源**
- `1cat-vllm/csrc/sm70_turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/{mma_sm70.h,operand_sm70_s884.h,smem_copy_sm70.h}`、`gemm/kernel/sm70_884_{4,8,16}.cu`、`gemm/mainloop_sm70.h`、`ops/awq_sm70_gemm.cu`
- `1cat-vllm/csrc/custom_all_reduce.cuh` —— 自研 push allreduce
- `1cat-vllm/scripts/serve_qwen38_27b_nvfp4_v100.sh`

---

## 9. 度量纪律（**必守，本会话踩过的坑**）

1. **记录两侧 binary 的 `--version`**；A/B 必须**同一个 build dir、同一套 CMake 参数**，唯一变量是待测代码。
2. **每个配置至少跑 2 次并报告离散度**（本机 256k prefill 同配置能漂 8%）。
3. **投机解码的 tg 是"含接受的吞吐"**，随接受率一起动 ⇒ 归因 kernel 必须**关 MTP**（`--spec-type none`）或**固定 seed 让两侧生成同样文本**；**报 t/s 必须同时报 AL 与每轮 ms**（`ms/轮 = AL ÷ tg`）。
4. **投机解码对比必须 ≥3 prompt + 固定 seed**（单 prompt 结论不可信：接受率随 prompt 摆动 0.25-0.40，排序会翻转）。
5. **A/B 有效性自检**：`md5sum` 两边 `libggml-cuda.so` **和 `libllama-common.so`**（spec 代码在后者里），相同即 A/B 无效。
6. **`build/bin/` 整套目录切换 + `LD_LIBRARY_PATH`**；只复制可执行文件完全无效。
7. `LLAMA_ROUND_TIMING` 的 `graph_compute` 只在 **M>1** 时同步 ⇒ M=8 可信，**layer 模式下的该数字是假的**。
8. `LLAMA_SPEC_TIMING` 必须从**每轮都执行**的函数 tick（`draft()`）；`process()` 只 prefill、`accept()` 会提前 return。
9. **`llama-bench` 不打印库内 INFO 日志** ⇒ 用 `llama-server`；libllama 里 `LLAMA_LOG_INFO` 可能无效 ⇒ 先用 `fprintf(stderr,...)`。
10. **`nsys` 必须优雅退出才写 `.nsys-rep`**（`kill -9` 丢报告）。
11. 诊断可 `NODROP=1` 跳过 drop_caches（加载 15 s vs ~270 s），但**对外数字必须带 `drop_caches`**。
12. **无盘 NFS**：下载 / 加载 / 编译共网卡 ⇒ **不要并行**。
13. 跨架构：本地 Windows 树与服务器树 **md5 不同是因为行尾**（`core.autocrlf=true`，`i/lf w/crlf`）⇒ 用 `git ls-files --eol` 判定，别慌。

---

## 10. 下一步任务（按序，可直接执行）

### 已完成（本会话）
- [x] PR #27858 移植（修 DFlash2 × tensor 硬崩，selector 正确性验证）
- [x] 前提核实：1cat 实测 221.6/230.8/263.2（用户数字真实）；差距在每轮延迟
- [x] 三项提速落地：并行 selector / `GGML_CUDA_P2P=1` / **NCCL**（+70%）
- [x] 大量否定：NCCL 调参 / 加卡 / Q8_0 强推 MMQ / `--spec-draft-device` / 多形状图缓存 / HMMA（天花板 1.36×）
- [x] 存档 + 提交（本文件）
- [x] 1cat 自己的实测开关表 + V100-native 清单已抄出来（§7.2/§7.3）

### 待办 1（**立即做**）：读那次已经跑完的 ncu 结果
`/root/ncu-tgt.log` —— 一次 `ncu --launch-skip 4000 --launch-count 60` 针对**真实 M=8 验证轮**的 profile，脚本 `/root/ncu-tgt.sh`，输出 `/tmp/ncu-tgt`、日志 `/tmp/ncu-tgt.log`（**我发起后结果一直没读**）。
metrics：`gpu__time_duration.sum`、`sm__throughput.avg.pct_of_peak_sustained_elapsed`、`dram__throughput.avg.pct_of_peak_sustained_elapsed`、`l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`、`smsp__inst_executed.sum`。
**目的**：量化 **M=8 验证轮里 GDN（gated delta net）与 attention 各自的成本占比** —— 这决定下一步往哪走。
若 ncu 失败（本机对大模型 profile 常 `failed to load model`），**替代方案**：
- 在 `ggml_backend_cuda_graph_compute` 里加 env-gated 的**按 op name 聚合计时**（需 `GGML_CUDA_DISABLE_GRAPHS=1`，因为 capture 的图里每节点 event 无效）
- 或用现成 op 微基准：`test-backend-ops perf -o GATED_DELTA_NET` / `-o FLASH_ATTN_EXT`（`tests/test-backend-ops.cpp`：`test_gated_delta_net` 在 4635/10953-10995，含 head_dim 128 与 K=4 变体；`test_flash_attn_ext` 在 7748，hsk/hsv=256 在 10776-10786/11271-11286）

### 待办 2：按待办 1 的结论，从下面三个里选**最高价值的一个**落地到 `llama.cpp/`
| 候选 | 来源 | 预期 |
|---|---|---|
| ① `FUSED_GDN_INPUT_FP16` / `FUSED_HC_FP16`（GDN 输入/头融合 + FP16） | 1cat `VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16` / `_FUSED_HC_FP16` | 若待办 1 显示 GDN 占比大，这条最值 |
| ② FP16 GEMV（checkpoint-FP16 GEMV） | 1cat `VLLM_SM70_QWEN38_FP16_GEMV` | 需先搞清它和"FP16 act/KV 原生路径"的差别 |
| ③ push-based allreduce（`csrc/custom_all_reduce.cuh`，C=1 +7.8% / C=16 +4.3%） | 1cat 自研，**与所有现成通道都不同** | 前提是先把 NCCL 基线守住（见"Must not"） |

**明确禁止**：`VLLM_SM70_USE_BREAKABLE_CUDAGRAPH`（1cat 实测 −13% ~ −29%）。
**也别忘了**：draft 前向 15.01 ms 对应的有效带宽只有 76 GB/s（dispatch 受限）—— 这是待办 1 之后**第二个**该查的点。

### 待办 3：验收（Goal 8dbd30ea 的 Done when，逐条贴证据）
1. 贴 **GDN vs attention 成本占比**实测证据行
2. `/root/p60-ab-harness.sh` **带 drop_caches** 重跑，贴 `MEDIAN_TG=`（目标 **≥110 且 >95.20**）与所选**卡数 / split-mode**
3. 贴 `[RT] target decode+sync` 行
4. 数值验收：两臂 `llama-perplexity` 的 `Final estimate`（能逐位一致就改贴 greedy sha256，否则 ppl 相对变化 **≤0.1%**）
5. 贴 `git status --porcelain`，**改动只落在 `llama.cpp/` 内**

### 待办 4（清理）
树里 7 处 `[RT]` `fprintf` 调试探针可清掉（另开 commit）。

---

## 11. 红线 / 服务状态 / 恢复

**红线**：
- 绝不 `git push` / 代开 PR / 代写 upstream 提交描述（本项目是私有 fork，不做上游贡献）
- 不修改 `/root/llm/systemd/llama-server.service`
- 不修改 `vllm/`、`1cat-vllm/`、`vllm-forkpoint/` 内容
- `llama.cpp/` 源码只用 **ASCII**（禁 `—` `→` `×` `…`，用 `-` `->` `x` `...`）；不新增 `tests/*` 文件；大改动先问用户

**服务状态（2026-09-20 交接时）**：`vllm-1cat.service`、`llmscope.service` 已 **stop**（腾卡）；`new-api.service` 仍在跑（不占显存）。
**恢复命令**：`systemctl start vllm-1cat llmscope`

**要跑 1cat 对标**：`bash /root/p32-vllm1cat-bench.sh`（启动服务 + 3 请求 + 采 journal）→ 日志 `/tmp/vllm1cat-bench.log`

---

## 12. 立刻继续（复制粘贴区）

```sh
# 0) 看现状
ssh -o BatchMode=yes root@192.168.50.235 'nvidia-smi --query-gpu=index,memory.used --format=csv; systemctl is-active vllm-1cat llmscope'

# 1) 读那次没读的 ncu 结果（待办 1）
ssh -o BatchMode=yes root@192.168.50.235 'grep -aE "ncu version|error|Error|Duration|GATED|FLASH|MUL_MAT|===|NCU_TGT_DONE" /tmp/ncu-tgt.log | head -60; echo ===TAIL===; tail -25 /tmp/ncu-tgt.log'

# 2) 若 ncu 不可用：op 级微基准（需先确认 build/bin/test-backend-ops 在）
ssh -o BatchMode=yes root@192.168.50.235 'cd /root/llm/test/v100-opt/llama.cpp && LD_LIBRARY_PATH=build/bin build/bin/test-backend-ops perf -o GATED_DELTA_NET -b CUDA0'

# 3) 基线复现（正式口径，带 drop_caches）
ssh -o BatchMode=yes root@192.168.50.235 'CARDS=0,1,2 SPLIT=tensor TAG=handoff-base L=/root/libdir-nccl bash /root/p60-ab-harness.sh'
```

**本地仓库位置**：`F:\vllm+llama.cpp\llama.cpp`（fork，HEAD `79504e72b`）、`F:\vllm+llama.cpp\1cat-vllm-v100-study`（档案）
**权威规则**：`F:\vllm+llama.cpp\QWEN.md`
