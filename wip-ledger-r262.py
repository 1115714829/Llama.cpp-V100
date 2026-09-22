#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Insert rounds R262-R264 into BASELINE-LEDGER.md and the matching nodes into PLAN-GRAPH.md."""
import io
import sys

LEDGER = r"F:\vllm+llama.cpp\1cat-vllm-v100-study\BASELINE-LEDGER.md"
GRAPH = r"F:\vllm+llama.cpp\1cat-vllm-v100-study\PLAN-GRAPH.md"

LEDGER_ANCHOR = "### R258 "

LEDGER_ENTRY = """### R262 ★ **MMVQ-WIDE（VDR 2 -> 8，只对 Q8_0/ncols=8）: 算子级反而慢 17% ⇒ 不采用**（env 门控 + 已还原）

- 动机：R259 量到每条 dp4a 要付 7.2 条线程指令，其中约 5 条是 scale 换算。把 vec_dot 加宽到"一线程吃完整 32 值块"后，
  8 条 dp4a 只付一次 scale。
- 实现：`vecdotq.cuh` 加 `vec_dot_q8_0_q8_1_wide`（VDR=8），`mmvq.cu` 加 `wide` 模板参数 + env `GGML_CUDA_MMVQ_WIDE`，
  只实例化 Q8_0/ncols_dst=8；默认不设 env 时**逐位等价**。
- 同库同会话算子级（`test-backend-ops perf -o MUL_MAT -p 'q8_0.*4096.*14336'`；正确性 `test -p q8_0` 两臂各 **50/50 通过**；
  `strings libggml-cuda.so | grep -c GGML_CUDA_MMVQ_WIDE` = 1 证明代码进了被加载的库）：

| n | 1 | 2 | 3 | 4 | 5 | **8** |
|---|---:|---:|---:|---:|---:|---:|
| 基线 µs | 82.39 | 85.21 | 88.42 | 89.03 | 92.75 | **112.40** |
| 宽版 µs | 82.64 | 87.62 | 88.51 | 90.68 | 94.68 | **131.86（+17.3%）** |

- ⇒ **指令数少了、时间反而多了** ⇒ 机制不是"指令数"，而是 **vdr=8 让每线程同时持有 `v[8]+u[8]` 的寄存器压力**（accumulator 之外多 16 个活寄存器）。
- ⇒ 与 R237(nwarps)、R244(rpb) 合起来：**MMVQ 在 Volta 上已经是该结构的局部最优**；解码侧（n≤8）没有便宜的 kernel 空间。
- 源码已还原（`mmvq.cu`/`vecdotq.cuh` 回 `.orig-wide`）。

### R263 ★★★ **长上下文预填充的"KV 每 ubatch 反量化"假设: 代码是真的，但代价不是主要的 ⇒ 证伪**（同源同库，只改 `--cache-type-k/v`）

- 代码事实（`fattn-common.cuh:1026-1088`）：只要选中 TILE/MMA_F16 内核（**n_q>1 必选**），就 `to_fp16(K_data, K_f16, ggml_nelements(K))`
  —— **K 是整条 KV cache 视图**，且这段在每个 ubatch 都跑 ⇒ 形式上 O(n_kv) 每次 = **O(N²/ub)** 总量。
- 零改码判别实验（3 卡 `-ts 1/1/1`、`-ub 2048`、`-r 3`、`GGML_CUDA_P2P=1`，同库 `libggml-cuda e7c082cc`）：

| 深度 | q8_0 KV | f16 KV（**无任何转换**） | 差 |
|---|---:|---:|---:|
| pp8192 | 1964.67 ± 2.24 | 1980.87 ± 0.45 | +0.8% |
| pp32768 | 1920.27 ± 0.80 | 1938.83 ± 0.47 | +1.0% |
| **pp131072** | **1271.21 ± 1.43** | **879.36 ± 0.06** | **−30.8%** |

- ⇒ **f16 KV 完全不做转换，却在 131K 慢 31%** ⇒ 那条 O(N²/ub) 的反量化**不是长上下文的墙**；
  长上下文是 **注意力本身读 KV 的字节数**在说话（q8_0 = 1.06 B/元素 vs f16 = 2 B/元素），但**也不是纯带宽**（否则 f16 不会慢这么多，见 R264 的真实占比）。
- ⚠️ 顺带更正一处**口径混用**：本账本记的 "32K prefill 2162.44"（R239）是 **`llama-bench` 默认 = f16 KV**；
  而 harness/生产跑 **q8_0 KV**（实测 pp32768 = **1920.27**）。以后引用要写清 KV 类型。

### R264 ★★★ **预填充逐算子表（带名字，单卡，`-ub 2048`）: 84% 在注意力 + LM head，全部 GEMM 只占约 5%**

方法：`test-export-graph-ops -m Q8_0.gguf -o /tmp/ops-ub2048.txt -ub 2048`（91 算子）+
`test-backend-ops perf --test-file`（89 个被计时，合计 **470.3 ms**）。⚠️ 名字与耗时在**同一行**（早先按行号配对的表是错位的，已作废）。

| µs | 占比 | 算子 |
|---:|---:|---|
| **368757** | **78.4%** | **`FLASH_ATTN_EXT(name=node_222, ne=[256,24,2048])`，KV 视图 262144** |
| 66518 | 14.1% | `MUL_MAT(name=result_output, ne=[248320,2048])`（LM head 对**全部 2048 个位置**算 logits） |
| 5191 | 1.1% | `FLASH_ATTN_EXT(ne=[256,24,1])`（n_q=1） |
| 4766 / 4737 | 1.0% / 1.0% | `MUL_MAT ffn_gate / ffn_out` |
| 3644 | 0.8% | `GATED_DELTA_NET` |
| 3419 / 2792 / 1753 / 1707 | 0.7/0.6/0.4/0.4% | `Qcur_full / node_13 / z / linear_attn_out`（其余全是 <0.5%） |

- ⇒ **一个算子（长 KV 的 FA）吃掉 78.4%**；把 89 个算子里所有 MUL_MAT 加起来约 5%。
- ⇒ 折算：n_kv=262144 时该 FA 单层 369 ms；256K 预填充要跑 121 个 ubatch，Σ∝n_kv ⇒ 16 层合计约 **357 s**，
  与实测 256K 预填充 **672 s** 同量级 ⇒ **这就是"256K 预填充 370 t/s / TTFT 672 s"的主因**（不是 GEMM，不是 KV 反量化）。
- ⇒ 效率：该 FA 的 FLOPs = 2*2*256*2048*262144 = 550 GFLOP / 0.369 s = **1.49 TFLOPS**（V100 FP16 张量核 125 TFLOPS 的 **1.2%**）
  ⇒ 与 R203「FA 11.7 GB/s = 病态」一致；**这是本项目目前最大的单一 kernel 缺口**。
- ⇒ **下一手（已写进 PLAN-GRAPH 的金色节点 FA-D256）**：D=256 / GQA=6 / Volta 的长 KV 预填充 FA。
  jusko 的两条恰好门控在我们的形状上：**JS4「2-CTA 紧凑核 +13.11%，门控 `gqa_ratio==6 && D==256`」** 与
  **JS2 `fattn-q8-volta.cuh`（smem 内 q8_0 反量化 -> mma.m8n8k4，101k 2.674->1.419 ms）**；
  本地副本 `v100-refs/jusko-llama-volta-qwen3flash/ggml/src/ggml-cuda/fattn-q8-volta.cuh` 可直接对照。

"""

GRAPH_ANCHOR = "  classDef goal fill:#ffe6cc,stroke:#d79b00,stroke-width:3px"

GRAPH_NODES = """  KVCT["x R263 **KV 类型 A/B: "每 ubatch 反量化整条 KV" 假设证伪**<br/>代码确实 O(n_kv)/ubatch（`fattn-common.cuh:1026-1088` to_fp16 整条 K/V）<br/>但 f16 KV（**零转换**）在 131K **慢 31%**: 1271.21 -> 879.36 t/s<br/>8K/32K 两者相同（+0.8%/+1.0%）⇒ 转换不是墙<br/>⇒ 长上下文是**注意力读 KV 的字节数**在说话（也不是纯带宽）<br/>⚠️ 口径: 账本旧值 pp32768=2162.44 是 `llama-bench` 默认 **f16 KV**；q8_0 KV 实测 1920.27"]:::no
  FA78["★★★ R264 **预填充真正的墙（带名字的逐算子表，单卡 -ub 2048，合计 470.3 ms）**<br/>**`FLASH_ATTN_EXT(D=256,24头,n_q=2048,n_kv=262144) = 368757 µs = 78.4%**<br/>LM head(`result_output` 248320x2048) 66518 µs = 14.1%<br/>全部 MUL_MAT 加起来约 **5%**（ffn_gate 4766 / ffn_out 4737 / Qcur_full 3419 / node_13 2792 ...）<br/>FLOPs 550 GFLOP / 0.369 s = **1.49 TFLOPS = FP16 峰值 125 的 1.2%**<br/>折算 256K 预填充: 16 层约 357 s（实测 672 s）⇒ **370 t/s 的主因**<br/>⇒ 与 R203「FA 11.7 GB/s 病态」同一件事；这是**最大单一 kernel 缺口**"]:::hot
  FAD256["★ **FA-D256（下一手主攻，落在预填充 = V100 真正支持的硬件支路）**<br/>目标: 长 KV / D=256 / GQA=6 / n_q>1 的 `FLASH_ATTN_EXT`<br/>参照（本地已存，门控正好是我们的形状）:<br/>**JS4** 2-CTA 紧凑核 **+13.11%**，门控 `gqa_ratio==6 && D==256`<br/>**JS2** `fattn-q8-volta.cuh`（smem 内 q8_0 反量化 -> mma.m8n8k4; 101k 2.674->1.419 ms）<br/>**JS3** D256 只差 6 个常量（combine 128/nstages 2 vs 64/1）<br/>先量: `[FAK]` 探针确认我们现在走哪个内核 + 该形状的 tile 配置<br/>注意 X16: 照抄 jusko 常量在 pp32768 是 -1.19% ⇒ 必须**按形状分别验证**"]:::next
"""

GRAPH_EDGES = """  NCU8 --> FA78
  FA78 ==>|84% 在注意力+LM head| FAD256
  JS4 ==>|门控 = 我们的形状| FAD256
  JS2 ==>|smem 反量化参考实现| FAD256
  KVCT -.->|否证了便宜的零改码解| FA78
  R264 -->|口径: f16 KV 2162 vs q8_0 1920| KVCT
  FAD256 ==>|预填充缺口 = 第二大缺口| G
  OPS --> FA78
  MMVQW -.->|R262 实测: +17.3% 慢 ⇒ 已否证| NCU8
"""


def insert(path, anchor, block, guard, encoding_ok=True):
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    if guard in text:
        print("  ALREADY " + path.rsplit("\\", 1)[-1])
        return
    if text.count(anchor) != 1:
        sys.exit("FAIL anchor count %d in %s" % (text.count(anchor), path))
    nl = "\r\n" if "\r\n" in text else "\n"
    text = text.replace(anchor, block.replace("\n", nl) + anchor, 1)
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print("  INSERTED " + path.rsplit("\\", 1)[-1])


# R264 in the graph refers to itself; keep the id simple
GRAPH_NODES = GRAPH_NODES.replace("  R264 -->|", "  FA78 -->|")

insert(LEDGER, LEDGER_ANCHOR, LEDGER_ENTRY, "### R262 ")
insert(GRAPH, GRAPH_ANCHOR, GRAPH_NODES + "\n" + GRAPH_EDGES, "FA78[")
