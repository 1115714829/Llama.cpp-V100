#!/usr/bin/env python3
"""Insert the R258-R261 rounds into BASELINE-LEDGER.md (newest first, before R256/R257)."""
import io
import sys

LEDGER = r"F:\vllm+llama.cpp\1cat-vllm-v100-study\BASELINE-LEDGER.md"
ANCHOR = "### R256 / R257 "

ENTRY = """### R258 ★★★ **基础设施陷阱（必须先记住）: `/mnt/3.84t/v100-opt/llama.cpp` 是一份**陈旧副本**，真正的开发树是 `/root/llm/test/v100-opt/llama.cpp`；且 harness 的 `L` 默认值指向**过期的 libdir**

- 触发：R258 的 `rows_per_block` 实验把补丁打到 `/mnt/3.84t/v100-opt/llama.cpp/ggml/src/ggml-cuda/mmvq.cu`，构建"成功"（`BUILD_RC=0`、`Built target ggml-cuda`），但**实测数字与控制臂逐位相同**。
- 根因（三重）：
  1. `/mnt/3.84t/v100-opt/llama.cpp/build-instr/CMakeCache.txt` 里 `CMAKE_HOME_DIRECTORY=/root/llm/test/v100-opt/llama.cpp`，
     `CMAKE_CACHEFILE_DIR=/root/llm/test/v100-opt/llama.cpp/build-instr` ⇒ **那棵构建树编译的是 /root 的源码、对象也落在 /root**。
     nvcc 命令行实测 `-c /root/llm/test/v100-opt/llama.cpp/ggml/src/ggml-cuda/mmvq.cu` ⇒ **打 /mnt 的源码不生效**。
  2. 两棵树的内容**不一致**：`mmvq.cu` 本地/`/root` 树 = `19c984c953d365fa0848a2c389239abc`（76201 B），`/mnt` 树 = `c51dc2a1…`（75866 B）。
  3. `/root/p60-ab-harness.sh` 的默认 `L=/mnt/3.84t/v100-opt/libdirs/libdir-instr`，其 `libggml-cuda.so.0.24.0 = 64317717…`、
     `libggml-base.so.0.24.0 = d283439c…` —— **都不是当前基准**（当前基准 `libdir-instr` = `/root/libdir-instr`：
     `libggml-cuda 0fb36620…`、`libggml-base a8de7c48…`，见 R257）。⇒ **不显式传 `L=` 的臂全部作废**。
- 规矩（新增，写进 AGENTS）：① 只在 **`/root/llm/test/v100-opt/llama.cpp`** 改码；② 构建后必须 `md5sum` 核对 `.o/.so` 的 mtime 与内容，
  并用 `strings <lib> | grep -c <新标记串>` 证明代码进了被加载的那个库（本次 `strings libggml-base.so | grep -c SCHED_RETRY` = 2 才证明生效）；
  ③ 跑臂一律显式 `L=/root/libdir-instr`。

### R259 ★★★ **ncu 把 n=8 的 MMVQ 瓶颈定死了: 不是带宽，是"每条 dp4a 7.2 条指令"+ LSU 队列**（`test-backend-ops perf` q8_0 m=4096 k=14336，单卡）

| 指标 | n=1 | n=8 |
|---|---:|---:|
| `dram__bytes.sum` | 62.91 MB | 63.19 MB（**相同**） |
| `lts__t_sectors.sum`（L2） | 65 MB | 111 MB（1.7x，不是 8x） |
| `l1tex__t_sectors.sum`（L1） | 505 MB | 1076 MB |
| `smsp__inst_executed.sum`（warp 指令） | 7.44 M | **26.16 M（3.5x）** |
| `gpu__time_duration` | 83.74 µs | 115.26 µs |
| issue（相对 490 G/s 上限） | 89 G/s（18%） | 227 G/s（46%） |

- ⇒ **每条 dp4a 要付 7.2 条线程指令**（837 M thread-inst / 117 M dp4a）；`dram` 两边一样 ⇒ **n=8 的额外代价不是权重流量，是逐 (row,token) 调用的固定开销**（y 载入 + 5 条 scale 换算 + 地址）。
- ⇒ **预填充侧同理**：`l1tex` 1076 MB / 63 MB = **17x 扇出**（每 lane 读一行、行间相距 15 KB）。
- ⇒ 这条数据同时**解释了 R237/R244 的两个否定**：改 nwarps / rows_per_block 都在"每 call 开销"这一层，没碰到真正的结构。

### R258b ★★ **`rows_per_block` 2 -> 4/8/16（Volta, ncols 5..8）: 算子级零变化 ⇒ 否证（补丁已还原）**

- 补丁（`wip-` 脚本: `kernel-lab/rpb-patch.py`）把 `calc_rows_per_block` 的 `case 5..8` 改成 Volta 专用 env 常量。
  因 R258 的陷阱，**第一次的三个变体全部作废**（改的是 /mnt 副本）。
- 事后用 ncu 一起把机制也算清楚了：**提高 rows_per_block 不减少 y 流量** —— y 的读取次数 = (row,token) 配对数，
  rpb 变大时 CTA 数同比变小，**总流量不变**（这解释了"为什么这个杠杆在结构上不可能有效"）。
- ⇒ 与 R237/R244 合并成一条**结构性结论**：MMVQ 的"每 call 固定开销"**只能靠换 kernel 结构降低**（见 R260）。

### R260 ★★★ **Volta MMA kernel（`DESIGN-W8VOLTA-MMA.md` 的 decode 版）: 已实现、结果正确，但 238 µs vs 现役 111 µs ⇒ 否证（机制性，不是"没做完"）**

- 独立实验台（`1cat-vllm-v100-study/kernel-lab/w8volta-bench.cu`，`nvcc -arch=sm_70`，约 2 分钟一轮迭代，不碰 llama.cpp 构建）：
  `mma.sync.m8n8k4.row.col.f32.f16.f16.f32` + `0x6400|u` 魔数解码（1 xor + 1 LOP3 + hsub2 + hmul2 每 4 值）+ 自写参考实现（fp64）对拍。
- 正确性：置换探针（PROBE=1）逐点核对 A/B 的 k 配对 ⇒ **maxabs 3.5e-2（ref 尺度 110.7），与设计预期一致**（f16 舍入，需重立门值）。
- 时间线（m=4096 k=14336 n=8）：**673 µs（lane=row）→ 320 µs（8 lane 协作 + smem 暂存）→ 238 µs（split-K=8 + 预取一块）**；现役 MMVQ = **111 µs**。
- ncu 证据：lane=row 版 `l1tex` = **33.0 M sectors = 1056 MB（16.8x 扇出）**、`No Eligible 92.7%`、`3.01 active warps/scheduler`；
  协作版 `l1tex` 降到 **10.5 M**，但**指令数升到 16.9 M**（协作装载+回读的代价），仍是 `No Eligible 86.7%`。
- **判决性机制（ptxas 实测，非推断）**：`mma.sync.aligned.m8n8k4.row.col.s32.s8.s8.s32`
  ⇒ `ptxas error: Illegal matrix shape '.m8n8k4' for instruction 'mma'`；`m16n8k16.s8` ⇒ `requires .target sm_80 or higher`。
  ⇒ **sm_70 没有 int8 张量核** ⇒ 走 mma 就必须付 int8->f16 解码（约 2 指令/值，1024 值/CTA-k-block = 每 warp-k-block 约 88 条），
  而 **dp4a 路径对权重编码是零解码**。这就是为什么"mma 每字节只解码一次、复用到 8 个 token"的理论优势**被解码成本吃光**。
- ⇒ **结论：在 V100 上，对 Q8_0 + n<=8 这个形状，现役 dp4a MMVQ 已经接近该架构的最优解；mma 路线在 sm_70 上不成立**（不是调参问题）。
  复现：`cd /mnt/3.84t/v100-opt/kernel-lab && nvcc -arch=sm_70 -O3 -o w8volta-bench w8volta-bench.cu && ./w8volta-bench 4096 14336 8 200`。
- 剩下的**唯一**未试的 kernel 方向（下一轮候选，成本低）：**把 MMVQ 自己的 vec_dot 加宽（VDR 2 -> 8）**，
  即一个线程吃完整 32 值块（8 条 dp4a 只付一次 scale），把 7.2 inst/dp4a 压到约 2.5（估算 26.16 M -> 约 9 M warp 指令）。
  代价：`vecdotq.cuh` + `mmvq.cu` 各约 20-30 行、只对 Q8_0、会改累加顺序（必须重立门值）。**这是 R259 数据的直接指向。**

### R261 ★★ **sched 快路径重试（`GGML_SCHED_RETRY_ALLOC`）: 实测 `ok=0/563` ⇒ 撤销 R253 的"便宜修法"，只剩多槽 gallocr**

- 改动：`ggml-backend.cpp` 里在 `backend_ids_changed` 为真时**先试一次** `ggml_gallocr_alloc_graph()`（并逐节点校验
  `ggml_backend_buffer_get_type(node->buffer) == sched->bufts[node_backend_ids[i]]`，不通过就回退），env 门控 + 计数器。
- 四臂 ABBA 同源（同一套库，唯一变量 env；`L=/root/libdir-retry`，其余 = B3 口径）：
  `c1 100.96 / r1 101.18 / r2 102.00 / c2 102.32`，**四臂 greedy sha256 全部 = `f3edac19…`**（门未破），ms/轮 都约 50。
  计数器：**`bic=563 retry=563 ok=0 bad=0`** ⇒ **一次都没成功**。
- ⇒ R253 的"99.3% 是 `backend_ids_changed`、短路导致 alloc 根本没被调用"**计数正确但推论不成立**：
  单槽 gallocr 里 `needs_realloc` 每轮都为真（一轮在 4-6 张**节点数不同**的图之间交替），慢路径（sync x3 + reserve）**是必需的**。
- ⇒ 结论：要拿这约 2-5 ms/call，只能做**多槽/按图指纹的 gallocr**（R253 修法② = E6 五次 abort 的那条线，触及容器与 arena 生命周期）。
  便宜修法**已穷尽**；本改动不采用（源码已还原为纯净 `967af7b5…`，重建后 `libggml-base.so` 需回到 `a8de7c48…`）。

"""


def main():
    with io.open(LEDGER, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    if "### R258 " in text:
        sys.exit("ALREADY_INSERTED")
    if text.count(ANCHOR) != 1:
        sys.exit("FAIL anchor count = %d" % text.count(ANCHOR))
    nl = "\r\n" if "\r\n" in text else "\n"
    text = text.replace(ANCHOR, ENTRY.replace("\n", nl) + ANCHOR, 1)
    with io.open(LEDGER, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print("INSERTED")


if __name__ == "__main__":
    main()
