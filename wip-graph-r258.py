#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add the R258-R261 nodes and edges to PLAN-GRAPH.md (§1 main graph).

Rule (AGENTS.md §0): every thing tried goes on the graph, falsified nodes are greyed not deleted,
and every completed piece of work must re-read the graph for missing nodes AND missing edges.
"""
import io
import sys

GRAPH = r"F:\vllm+llama.cpp\1cat-vllm-v100-study\PLAN-GRAPH.md"
ANCHOR = "  classDef goal fill:#ffe6cc,stroke:#d79b00,stroke-width:3px"

NODES = """  TRAP["★★★ R258 **基础设施陷阱（先读这条再动手）**<br/>`/mnt/3.84t/v100-opt/llama.cpp` 是**陈旧副本**（打补丁不生效）<br/>真开发树 = `/root/llm/test/v100-opt/llama.cpp`（本地 HEAD 与它 md5 一致）<br/>`/mnt/.../build-instr` 的 CMakeCache 指向 /root ⇒ nvcc 编的是 /root 源码<br/>harness 默认 `L=/mnt/3.84t/v100-opt/libdirs/libdir-instr` = **过期库**(64317717/d283439c)<br/>=> 必须显式 `L=/root/libdir-instr`；改动后必须 `strings <lib> | grep -c <标记串>` 验证"]:::hot
  NCU8["★★★ R259 ncu 把 n=8 的 MMVQ 定死: **不是带宽受限**<br/>dram 两边都 63 MB；L2 只 1.7x；**warp 指令 7.44M -> 26.16M (3.5x)**<br/>= **每条 dp4a 付 7.2 条线程指令**（y 载入 + 5 条 scale 换算 + 地址）<br/>n=8 只有 547 GB/s（n=1 是 751 = DRAM 顶），issue 46%<br/>L1 扇出 17x（每 lane 读一行、行间距 15 KB）"]:::hot
  RPB["x R258b `rows_per_block` 2->4/8/16（Volta ncols 5..8）<br/>首轮三臂因 TRAP 全废；机制：y 读取次数 =(row,token) 对数<br/>rpb 变大时 CTA 数同比变小 => **总流量不变** ⇒ 此杠杆结构上无效<br/>(与 R237 nwarps / R244 rpb->1 合起来 = 该内核参数已局部最优)"]:::no
  W8V["x R260 **Volta MMA kernel**（DESIGN-W8VOLTA-MMA 的 decode 版）: 已实现且**数值正确**<br/>置换探针 + fp64 对拍（maxabs 3.5e-2 / ref 110.7 = f16 舍入预期）<br/>673 µs(lane=row) -> 320(8 lane 协作+smem) -> **238**(split-K=8+预取一块) vs 现役 **111 µs**<br/>ncu: lane=row 版 **16.8x L1 扇出**(33.0M sectors)、No-Eligible 92.7%、3.0 warps/scheduler<br/>★★ 判决性: **ptxas 拒绝 `m8n8k4.s8`**（`Illegal matrix shape`；`m16n8k16` 要 sm_80）<br/>= **sm_70 没有 int8 张量核** ⇒ mma 必付 ~2 指令/值的 int8->f16 解码，<br/>而 **dp4a 对权重编码是零解码** ⇒ mma 的理论优势被解码成本吃光<br/>= V100 上 Q8_0 + n<=8 的 mma 路线**架构性不成立**（非调参问题）"]:::no
  MMVQW["★ **MMVQ-WIDE（下一轮 kernel 候选，直接来自 R259 的数字）**<br/>把 `vec_dot_q8_0_q8_1` 加宽 VDR 2 -> 8: 一个线程吃完整 32 值块<br/>=> 8 条 dp4a 只付**一次** scale（7.2 -> 约 2.5 inst/dp4a）<br/>估算 warp 指令 26.16M -> 约 9M；改动小（vecdotq.cuh + mmvq.cu 各 20-30 行）<br/>只对 Q8_0；**会改累加顺序 ⇒ 必须重立门值**；先做算子级 perf（20 s 出数）"]:::next
  SRETRY["x R261 sched 快路径重试 `GGML_SCHED_RETRY_ALLOC`: **ok=0/563**<br/>四臂 ABBA 同源 100.96 / 101.18 / 102.00 / 102.32，门全 `f3edac19…`<br/>计数器 `bic=563 retry=563 **ok=0** bad=0` ⇒ **一次都没成功**<br/>= 单槽 gallocr 的 `needs_realloc` 每轮必真（一轮在 4-6 张节点数不同的图间交替）<br/>= **慢路径是必需的**；便宜修法穷尽 => 只剩多槽/按指纹的 gallocr（= E6V 那条线）"]:::no
"""

EDGES = """  OPS --> NCU8
  NCU8 --> RPB
  NCU8 --> W8V
  HMMA -.->|其 decode 版已否证| W8V
  NCU8 ==>|数字直接指向的唯一方向| MMVQW
  W8V -.->|同一根因: 没有 int8 mma| MMVQW
  MMVQW ==>|先算子级 perf 再上机| PASSCOST
  GALLOCR --> SRETRY
  SRETRY -.->|只剩多槽这一条| E6V
  SRETRY -->|慢路径确认为必需| GALLOCR
  TRAP --> RPB
  TRAP --> W8V
  TRAP -.->|所有上机实验的前提| G
  W8V -.->|RAM 实测 7.2 inst/dp4a 的另一个解释面| PASSCOST
"""


def main():
    with io.open(GRAPH, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    if "TRAP[" in text:
        sys.exit("ALREADY_INSERTED")
    if text.count(ANCHOR) != 1:
        sys.exit("FAIL anchor count = %d" % text.count(ANCHOR))
    nl = "\r\n" if "\r\n" in text else "\n"
    block = NODES.replace("\n", nl) + nl + EDGES.replace("\n", nl) + nl + ANCHOR
    text = text.replace(ANCHOR, block, 1)
    with io.open(GRAPH, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print("INSERTED nodes+edges")


if __name__ == "__main__":
    main()
