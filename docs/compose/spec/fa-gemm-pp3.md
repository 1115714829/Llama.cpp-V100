---
feature: fa-gemm-pp3
status: in-progress
updated: 2026-09-23 (R289 重过滤更正: 删错立正 + 及格线重登记)
branch: feat/fa-gemm-pp3
commits: # 交付时填写
---

# P-P3-T 探针：FA 通用 PV 显式 mma + Wide + RegP（P-P3 分解线的前置件）

## [S0] 更正记录（2026-09-23 重过滤，删错立正）

1. **删「1.49 TFLOPS = 峰值 1.2%」**（漏乘 24 头，R269 已判死）：真基线 stock **35.8** / Path A **45.9 TFLOPS**（R269 口径）。
2. **删「提高每条 mma 指令的有效 FLOP」**：HMMA.884 是固定 8x8x4 的 ISA 原子，不存在"更大的原子"。Wide 的真机制 = **每 FLOP 的循环步 / smem<->reg 拷贝 / 同步开销摊薄**。
3. **删「rescale / online softmax 降频」承诺**：rescale 频率由 **kBlockN**（每 32 个 KV 行一次）决定，本合同 kBlockN=32 买不到；它属于 P-P3（分解版，每 ~24K 行一次）。
4. **及格线重登记**（见 [S4]）：旧线「FA >= +50% 或 pp32768 >= +5%」按死基线定出，换算真基线后高过 1cat 生产最佳（60.8 TFLOPS），**对本 scope 不可达**。
5. **T2 阻塞加深（复验）**：除 `load_v_fragment_tt` 偏移手调外，**`SmemLayoutV` 在 kDChunk=128 下非双射**——地址 `32*n0+128*n1+d0+64*d1`，(n0=0,n1=1,d0=0,d1=0) 与 (n0=0,n1=0,d0=0,d1=2) 同落偏移 128。=> `Sm70D256WideTraits` 现状是**错布局**，不只是"没调好"。
6. **参照事实（已核）**：1cat 1.5.0 的 `csrc/attention/sm70_v37/tail.cu:224-243` 仍是同一个硬编码 `BLayout Shape<_4,_2>` + `cute::gemm`——**他们也没泛化 PV B 碎片**，上台阶是另起 `sm70_79t/` 分解架构。=> 本探针的"通用显式 mma PV"是**自有地基**，无处可抄。
7. **P-P3 名实对齐**：`docs/v100-dev/1-计划突破方向.md` 的 P-P3 = PR#286/79T 分解（GQA 6 头打包 + prefix/causal-tail 分离 + block 级 rescale，45.9 -> 60.8 台阶）。本文件 = 其前置探针 **P-P3-T**。

## [S1] 问题

256K 预填充的墙是长 KV `FLASH_ATTN_EXT`：**峰值 ubatch 算子表占 78.4%**（R264 口径），**全程占比 32K ≈ 19% / 256K ≈ 60-66%**。

效率现状（R269 口径，FLOPs x 24 头）：stock MMA_F16 **35.8** -> Path A **45.9 TFLOPS**。1cat 同族梯子：**46.63-47.1（v1.3.0 Split-D/N32 = 本树移植源）-> ≈60.8（PR#286/79T）**，79 为实验上限。

=> 剩余台阶 = **FA 算子 +29%**，来自 1cat 的三个结构件：GQA 6 头打包进 M / wide QK-PV GEMM 化 / prefix-causal-tail 分离 + 每 ~24K 行 rescale。本探针只吃"开销摊薄 + RegP"的碎片，预期 **+0~10%**；其余留给 P-P3。

**前提（不重辩）**：线性 GEMM 本就在 FP16 张量核上（R213/R215，勿重做）；Path A 已发 HMMA.884，问题是**喂得太碎**（kBlockN=32、kDChunk=64、P 走 smem 往返）；Path B q8-direct 速度 -3.09% 已否（价值在显存）。

## [S2] 设计（探针版）

### 学习源（只读 `v100-refs/`，禁止改）

| 源 | 借什么 |
|---|---|
| `1cat-vllm/csrc/attention/sm70_79t/prefill.cu` | **P-P3 的主学习源**（分解架构：cublas QK 大 GEMM + CUTLASS PV + block 级 rescale + causal-tail 分离）——本探针不实现，只确认地基方向不与它冲突 |
| `1cat-vllm/csrc/attention/sm70_v37/tail.cu` | Split-D 同源后裔（`splitd_pv_gemm_tt` 同名同形态）；对照它可确认我们的偏差；**它没有泛化 B 碎片** |
| `jusko-llama-volta-qwen3flash/ggml/src/ggml-cuda/fattn-q8-volta.cuh` | 显式 `mma.sync.m8n8k4` 封装、fragment 寻址、寄存器 softmax->half2 |
| `ninfer-v100/src/ops/common/volta_mma.cuh` | Volta fragment load / `mma.m8n8k4` 封装（NF10） |

**禁抄常量**（X16）：只借结构；tile 尺寸按我们形状（D=256, n_q=2048, 长 n_kv, gqa=6, q8_0 KV）实测定。

### 改什么（代码落点 `llama.cpp/ggml/src/ggml-cuda/`）

1. **通用 PV 显式 mma 路径（本探针的真交付物）**：把 `splitd_pv_gemm_tt` 的手拼 B 碎片 + `cute::gemm` 换成显式 `mma.sync.m8n8k4`（jusko/NF10 形态），`BLayout` / `load_v_fragment_tt` / `SmemLayoutV` 折叠常量按 kDChunk 泛化。这是 Wide、RegP、以及 P-P3 的 GQA 打包共同的地基。备选（不推荐）：给 cute 做宽 BLayout 多原子拼接（形状摩擦大）。
2. **Wide：kDChunk 64 -> 128**（kDChunks 4 -> 2）：QK/PV 内层迭代减半，PV N tile 16 -> 32。smem 约 53 KB => **2 CTA/SM 掉到 1 CTA/SM**，占用减半是主要风险（A/B 判，允许 Wide 负收益）。
3. **RegP 组合臂**：单开曾测 0 收益；与 Wide 合并重测（组合条件重测纪律）。
4. **门控**：`LLAMA_SM70_FA_GEMM`（**判值**，默认 OFF）；`=0` 与 Path A 行为逐位等价；回退链 `LLAMA_SM70_D256=0` 仍强制上游 MMA_F16。

### 明确不改 / 不借

| 项 | 原因 |
|---|---|
| rescale 降频 / kBlockN 加大 | 属 P-P3（本合同 kBlockN=32 不变） |
| GQA 6 头打包、prefix/causal-tail 分离、score workspace | 属 P-P3（结构性，须用户批准） |
| JS2 整核（T=4）、JS4 dual-CTA、Path B、加卡、换格式 | 控制变量 / 已否 |

### 方法（用户 2026-09-23 五条，必须遵守）

1. 善用互联网搜索；2. 借鉴本地仓 `v100-refs/`；3. 习惯 A/B；4. **被否项在组合条件下重测**（RegP 单开 0 != Wide+RegP 为 0）；5. **禁止臆想**。

### 正确性与度量（红线不变）

- 同源 A/B，唯一变量 = `LLAMA_SM70_FA_GEMM`；**机制自证计数**必配（探针证明走了新 PV 路径）。
- 数值：greedy sha256 门；位漂移则**重立门并告知用户**（结构重写允许一次重立）。
- 预填充报 **FA 算子 ms/TFLOPS** + **pp（256K 为主锚，32K 回归监控）**；decode 不回退。
- `llama-bench -r >= 8` 入账；`GGML_GALLOCR_SLOTS=3`（>=32K 默认 8 槽 OOM，R282）。
- 正式数字 `drop_caches`；诊断 `NODROP=1` 须标明；禁止并发测量；构建/测量互斥。
- 源码 **ASCII**；不新增 `tests/*`；大改动模式已在本 Spec 批准范围内。

## [S3] Out of Scope

- **P-P3 分解版**（GQA 打包 / prefix-tail 分离 / block rescale / 2.4 GB score workspace / 新图 op）——独立大 feature，须用户批准后另立 Spec。
- JS4 G6 dual-CTA、P-P4 常量扫描、SG3 阈值表（后续刀）。
- P-P0 悬空实验（M1 头部失衡 / TP 扫描）——独立零代码线，见 `1-计划` §1.3。
- 线性 GEMM / cuBLAS 路径、权重格式、KV 压缩、decode P-D*、P-M1。
- 上游 PR / commit / push（红线）；生产 unit 与 `/root/llm/{systemd,llama.cpp,ac922env}`（只读）。

## [S4] 预登记（探针版，2026-09-23 重立；旧线作废）

| 指标 | 基线 | 预期 | 及格 |
|---|---|---|---|
| FA 算子（n_q=2048 / n_kv=262144） | **287.5 ms = 45.9 TFLOPS**（Path A，R270/R269 口径） | +0~10% | **>= +5%** |
| pp256K（TP3, ub2048, SLOTS=3） | 同轮重测（R272 参考 702 t/s） | +0~7% | **>= +3%** |
| pp32768 | 同轮重测 | - | 仅回归监控（FA 占比 ~19%，**不作锚**） |
| decode / tg | B5 线 | - | **零回退** |

**及格线 = FA >= +5% 或 256K pp >= +3%**（机制自证绿 + 门值过）；两线全负 -> 不采用，记入失败记录。

（参考：P-P3 分解版的预登记另立——FA >= +25% 或 256K pp >= +15%。）

## Tasks

- [x] T1: tile 目标契约——kDChunk 128 / kBlockN 32 / smem 约 53.8 KB（1 CTA/SM）/ RegP 并测；已含 [S0]-2/3 的机制更正（kBlockN 不动 => 无 rescale 收益）
- [ ] T2: **通用 PV 显式 mma 路径 + Wide/RegP 实现**（`LLAMA_SM70_FA_GEMM` 门控，判值，默认 OFF）——阻塞点已定性：B 碎片 `Shape<_4,_2>` + `cute::gemm` 形状约束 + `SmemLayoutV` 非双射；**解法已定：显式 `mma.m8n8k4` + 常量按 kDChunk 泛化**。验收：编译过；`=0` 与 Path A 等价；`=1` 走新路径探针计数（covers: S2; depends: T1）
- [ ] T3: 服务器可操作区构建 + 冒烟（`SLOTS=3`，pp8192/32768）——验收：无 launch 断言；机制计数绿；sha256 门或重立门（covers: S2; depends: T2）
- [ ] T4: 同源 A/B 三臂（Path A / Wide / Wide+RegP）+ 算子级计时——验收：报 FA ms/TFLOPS 与 pp（256K 主锚，>=2 臂 ±）；对照 [S4]；达标才采用（covers: S2; depends: T3）
