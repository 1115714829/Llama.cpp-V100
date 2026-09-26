# STATE · 当前真相（主代理维护，≤ 1 页）

更新：2026-09-26 11:20 · 阶段：**0（基线）**

## 已定决策

- **D1 测试路线**：先跑 256K 标准测试（L3）定 REF 与 BASE，再用单卡真实形状（L1）做日常迭代，每个里程碑回到 L3 验收（LEDGER L0002）。

## 基线（阶段 0 填写）

| 指标 | REF（1cat-vLLM） | BASE（llama.cpp） | 差距 |
|---|---|---|---|
| TTFT s | 待 S0-05 | 待 S0-04 | |
| TPOT ms | | | |
| pp t/s | | | |
| AL / 每轮 ms | — | | |
| 每卡显存峰值 MiB | | | |
| L3 噪声带 | | | |

## BASE 候选

- 源码：`llama.cpp` 分支 `feat/p3-decomp`，HEAD `e7482aada` + 6 个未提交文件（这 6 个文件已抽查与服务器构建树逐字节一致）。
- 二进制：`/root/libdir-gb`（构建于 2026-09-26 09:52，`libggml-cuda` md5 `d28f8810…`）。
- S0-02 全树对账通过后：主代理 commit + tag `sm70-base0`；执行者把库冻结到 `/mnt/3.84t/sm70/libs/base0/`。

## 服务器（S0-01，2026-09-26 11:00）

6×V100 空闲（GPU4 142 MiB、GPU5 396 MiB 为非计算占用）；`vllm-1cat` inactive，`llmscope`、`new-api` active；无 tmux、无锁；`/` 剩 38 GB，`/mnt/3.84t` 剩 2.5 TB。

## 工作队列（S0-07 后填写）

（空）

## 待用户决定

1. S0-05：是否授权执行者启停 `vllm-1cat` 重测 REF。
