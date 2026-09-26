---
name: staged-baseline
description: Run any performance A/B in this llama.cpp V100 project the project way (same-source arms, greedy sha256 correctness gate, >=2 ABBA arms with dispersion), then record the staged result in BASELINE-LEDGER.md and adopt it as the next round's comparison baseline only if it beats the current one. Use before/after any change that is supposed to make the V100 build faster, when deciding adopt-or-reject, and whenever asked what the current baseline or the gap to 150 T/s is.
---

# staged-baseline — 阶段成果账 + 同源 A/B 纪律

> 用户 2026-09-21 定的规矩：**每拿到一个阶段成果就记录下来，并作为下一轮扩展的对比初值；比基准差的一律不采用。**

## 何时使用
- 任何「应该更快」的代码 / 配置改动，上机前后。
- 用户问「这版比上一版好吗」「当前基准是多少」「离 150 T/s 还有多远」。
- 一组实验跑完，准备下结论、写文档、更新 PLAN-GRAPH 时。

## 唯一的账本
`1cat-vllm-v100-study/BASELINE-LEDGER.md`
- **采用链** B0 -> B1 -> B2 -> ...：每行就是当轮的**对比初值**（配置 + tg p1/p2/p3 + AL + ms/轮 p1/p2/p3 + 均值 + 相对上一阶段 + 判定）。
- **灰色名单**：被否证的候选 + **为什么被否**（不得删除 —— 「为什么被否」本身就是信息）。
- 同一份规矩必须同时出现在 `AGENTS.md` §1（每次会话自动加载）与 `PLAN-GRAPH.md`（节点接线）里，三处互相指路。

## 报数铁律
1. **tg 与 ms/轮 必须并列**；`ms/轮 = AL / tg * 1000`。
2. 三个 prompt 全报（p1/p2/p3），并给**均值 ms/轮** —— 判「采用 / 不采用」只看这个量。
3. AL 从响应 JSON 的 `timings` 取（`rounds = draft_n / n_max`），**不要 grep 服务器日志**。
4. `MEDIAN_TG` 是三 prompt 的**中位数**，AL 不同时会骗人（B2 实例：中位数落在几乎没动的 p1 上，只显示 +1.1%，而均值 ms/轮实际 -6.5%）。

## 同源 A/B 协议（缺一不可）
1. **同源**：两臂用**同一套库**。最强形式 = **env 门控**（同一 build、同一套 .so，只改 `GGML_*` 环境变量）—— 此时四库 md5 天然相同，A/B 不可能失效。
   若必须两套库：四个 md5 全记（`libggml-cuda.so` / **`libggml-base.so`** / **`libllama.so`** / `libllama-common.so`），
   再加二进制标记串 `strings <lib> | grep -c <新串>` —— 只看源码 grep 会骗人。
2. **每配置 >=2 臂，ABBA 交错**（控制 / 特性 / 特性 / 控制）抵消漂移；**对照两臂差 < 0.1%** 才认这组数据。
3. **正确性门**：所有臂的 greedy sha256 必须等于门值
   `f3edac19446ef641447f8391c71cb1b25a74055d662e9887890fa4a98602ca34`。**门破 = 不采用**（无论多快）。
4. **口径**：诊断用 `NODROP=1`（热加载 ~16 s）；要对外报数才用 `drop_caches` 官方口径（加载 255 s/臂）。两者不得混在一次比较里。
5. **独占机器**：测量期间不跑别的测量、不编译、不 `drop_caches`；`/tmp/LLAMA_BUILD_LOCK` 存在就不许跑；**每臂不同 `PORT`**；
   每臂结束先看 `STATUS=OK`（出现 `LAUNCH_FAILED` 则整组作废）。
6. 命令模板与 B2 的完整复现行在 `BASELINE-LEDGER.md` 里。

## 判读纪律（踩过的坑，别再犯）
- **不得从「省了 X ms 主机」推收益**（E7：删掉约 1 s 主机工作，墙钟零变化）。任何主机侧改动必须**同臂量边际**后才投入。
- **穿透率不是 1**：本机实测 25%~55%，所以「主机少了 14 ms/轮」不等于「轮时少 14 ms」。
- **计数器增减不是墙钟代理**（direct/replay 计数变了 != 变快）。
- 阶段成果**只按 ms/轮 判**，不按机制是否「讲得通」判。
- 报数时若某项没测，写「未测」，不要用推算值顶替实测值。

## 流程（照做）
1. 先写**先验**（假设 + 预期幅度 + 如果为负说明什么），再上机 —— 不要先跑后编故事。
2. 跑 ABBA >= 4 臂。
3. 每臂收集：`MEDIAN_TG`、p1/p2/p3 的 tg 与 AL、greedy sha256、`[META]` / `[RT]` / `spec timing` 末行、四库 md5。
4. 用 `scripts/ab_summary.py <tag>...` 算 ms/轮（三 prompt + 均值）。
5. 判：均值 ms/轮优于当前 Bn -> **采用**；否则进灰色名单并写明理由。
6. **采用时**：更新 `BASELINE-LEDGER.md`（新 Bn+1 行 + 证据段 + 复现命令），同步 `AGENTS.md` §1 一行与 `PLAN-GRAPH.md` 节点。
7. 汇报：tg 与 ms/轮并列 + 相对上一阶段的变化 + 未决项（例如某个 prompt 没动、口径是 NODROP）。

## 例子（B1 -> B2 的真实一次）
- B1（9 个自建 commit，NCCL TP3 + P2P）：tg 99.83 / 80.59 / 115.78，ms/轮 55.60 / 52.36 / 55.10（均 **54.35**）
- B2（+ FGC 整调用单图捕获，`GGML_META_FULLGRAPH=1`）：tg **100.95 / 88.30 / 129.12**，ms/轮 **54.98 / 47.79 / 49.41**（均 **50.73**）
- 四臂同源（唯一变量 env）、两特性臂差 0.02%、四臂 sha256 门未破 => **采用 B2，下一轮初值 = 50.73 ms/轮**。
- 诚实保留：p1 几乎没动（待查）；口径是 NODROP，官方口径待补。
