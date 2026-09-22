# DESIGN-W8VOLTA-MMA.md — Volta 专用 Q8_0 小批量 MMA 内核（**等用户批准后动手**）

> 状态（2026-09-22）：**设计就绪，未动一行代码**。红线「大改动 / 新模式 / 新子系统：先停下问用户」⇒ 需要用户明确同意才开工。
> 关联节点：`PLAN-GRAPH.md` 的 **HMMA**（decode 版）、**NWARPS**（已灰）、**PASSCOST**。

## 1. 为什么是这个（全部是实测，不是估算）

| 事实 | 数字 | 来源 |
|---|---|---|
| 一次 8-token 目标前向（= 一轮的主体） | **约 36 ms**（llama-bench pp8 56.1 ms 减去其约 17 ms/次固定项；或按轮时 50.25 - draft 5.9 - selector 1.2 - 主机暴露约 7） | R234 / R235 |
| 其中"权重流"部分 | 每卡 9.68 GB；单算子可测到 **715 GB/s（n=1，82.10 µs / 58.7 MB）**，而 **n=8 只有 526 GB/s（111.66 µs）** | R236 |
| ⇒ 若 n=8 能追平 n=1 的带宽 | 权重流 18.4 -> 13.5 ms ⇒ **每轮省约 4.9 ms（约 +10% tg）** | 推算 |
| 便宜的调参路已全部走死 | nwarps 2->4: n=8 **+7.3%**（R237）；rows_per_block 2->1: n=8 **+40%**（R244）；换 MMQ: **-22% ~ -28%**（R234 policy 臂） | R237/R244/R234 |
| ⇒ 结论 | **526 GB/s 是该内核结构（2 行分块摊薄 y 重读）的固有值，只能换内核** | R237+R244 |

## 2. 做什么

在 sm_70 上，对 **Q8_0** 且 **ne11 <= 8**（decode / DFlash2 验证批）用一个新内核替换 `mul_mat_vec_q`：

- **融合反量化 + `mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f32`**（sm_70 只有 m8n8k4；m16n8k16 被 ptxas 拒）。
- 参考实现（本地只读）: `v100-refs/ninfer-v100/src/ops/linear/w8/w8_volta_mma_gemm.cuh`。
  **W8G32 == Q8_0**（int8 码 + 每 32 个元素一个 f16 scale）。
  反量化技巧（**2 op / 2 元素，无 int->float 转换**）: `0x6400|u == 1024+u`，XOR 0x80，`__hsub2` 1152.0，`__hmul2` scale。
- 布局：A = 权重（`block_q8_0`，34 B/32 权重，行主序，与今天完全一致，**不需要改量化格式**）；B = 激活（今天 MMVQ 路径已经把它们量化成 `block_q8_1`，**同样适用**该技巧）⇒ **不需要新的量化 kernel**。
- 设计要点沿用 v100-skinny 的结论：**QP 切 N、A-stationary、主循环不进 smem、全程 1 个 barrier**；避免 K 切分（v1 / B_ring 两版都死在 K 切分上）。

## 3. 落点（改动范围）

| 文件 | 改动 |
|---|---|
| `ggml/src/ggml-cuda/mmvq-w8-volta.cuh` | **新增**（约 300-400 行；CUDA 源是按 glob 收集的 ⇒ **不需要改 CMake**） |
| `ggml/src/ggml-cuda/mmvq.cu` | dispatch 一处：env `GGML_CUDA_SM70_MMA_Q8` 存在 **且** `type == Q8_0` **且** `ne11 <= 8` **且** sm_70 时走新内核；否则原样 |
| `ggml/src/ggml-cuda/common.cuh` | 复用 `fp16_mma_hardware_available` 与 `__CUDA_ARCH__ == 700` 判定（尽量不加新宏） |

**默认零行为改变**（env 不设 = 今天的行为；`test-backend-ops test` 与 `harness` 都必须仍然逐位一致）。

## 4. 验收（按账本纪律）

1. **env 关**：与 B3 完全一致 —— greedy sha256 = `f3edac19…`、ms/轮 50.2-50.3（R240 已建立该口径）。
2. **env 开**：数值路径会变（求和顺序不同）⇒ **必须为该配置重新确立门值**（同臂两次一致），不得沿用 `f3edac19…`；AL 必须不掉。
3. **算子级**：`test-backend-ops perf -o MUL_MAT -b CUDA0` 的 q8_0 n=8 **目标 <= 90 µs**（>= 650 GB/s；现在 111.66 µs）。
4. **整轮**：同源 A/B（env 开/关，各 >= 2 臂，不同 PORT）**ms/轮 至少改善 2%**（约 -1 ms）才算成功；否则按"比基准差 / 不够本"就地标灰。
5. 正确性另加：`test-backend-ops test -o MUL_MAT -p type_a=q8_0`（n=1..8）与 CPU 参考在容差内。

## 5. 代价与风险

- **工作量**：新内核 300-400 行 + 调试。按本项目节奏（每次 nvcc 重编 mmvq 相关 TU 约 5-10 min）估 **2-5 小时**，且**结果不确定**（mma 版在小 n 上未必赢过高度优化的 MMVQ）。
- **上限**：即使做到 715 GB/s，也只是 **-4.9 ms/轮（约 +10% tg）**；与 150 t/s 需要的 **-13.25 ms** 仍有差距 ⇒ 它**不是唯一一招**，必须与"主机侧 / AR / draft"的项叠加。
- **风险控制**：env 门控 + 默认关 + 只在 Q8_0/sm_70/ne11<=8 生效 ⇒ 不影响 Q4_K_M 生产路径与任何其他后端。

## 6. 需要用户拍板

**是否开工？（同意 / 不同意 / 先做别的）** 若同意，我会按红线先 commit 当前本地树（B3 那笔改动还没提交），并保留 `.orig` 与 env 门控。
