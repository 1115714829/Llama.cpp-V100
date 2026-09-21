# PLAN-GRAPH — 任务依赖图（新会话先读这一份）

> 生成于 2026-09-21 Round 140。**这份文件的目的：用最少 token 让接手者知道「做什么、按什么顺序、为什么」**。
> 详细证据在 `GAP-ANALYSIS-2026-09-21.md` / `SESSION-2026-09-20-measurements.md`；本文件只放图与状态。

## 0. 硬事实（不要重新论证）

```
基线        tg 99.97 / 99.18 t/s, AL 5.55, 55.5 ms/轮, sha256 f3edac19...  (两臂离散 0.8%)
目标        >=180 t/s, ~20 ms/轮                      差距约 1.8x
每轮细账    meta 主机循环 29.4 ms (53%) = dev 19.7 + ar 6.1 + prologue 3.3
            dev 里 13.97% 调用走 direct(两臂逐位相同) 吃掉 dev 的 59%
可加性      4 臂实测: 轮时 ~= 主机 + GPU串行 (55.1=20.6+34.5; 77.0=4.9+72.1)
meta 税     15.8 ms/轮  (由 layer-split 这条独立路径证实: alloc 15x, enqueue 4x)
```

## 1. 节点表

| id | 内容 | 状态 | 成本 | 预期增益 | 证据 |
|---|---|---|---|---|---|
| **N0** | **解决 R1 矛盾**：`LLAMA_ROUND_TIMING_SYNC` A/B 判定主机/GPU 是否真重叠 | **RUNNING** `/tmp/sync-ab.txt` | 零代码 20min | 决定整张「已证伪」清单是否可信 | GAP §11.3 |
| **N1** | **P0-1: q8_0 KV 张量核注意力**（smem 反量化 -> f16 -> mma.sync.m8n8k4） | **TODO = 前置项** | 大（336 行 kernel 按 T=8 改造） | 外测 q8 子核 2.674->1.419ms@101k、6.879->3.363ms@260k；四件套 TG +32.19% | `v100-refs/jusko.../fattn-q8-volta.cuh` |
| **N2** | 只读诊断：打印 `D==256 && Q->ne[1]>1` 时选中的 FA kernel 与 `K->ne[1]` | TODO | **极小**（1 行 + 一次跑测） | 证实 verify 是否走 TILE + 每轮整条 KV 反量化 | GAP §7.1 |
| **N3** | P1-3 D256 FA 常量 A/B（ncols=32/64 两行） | **BUILT, 待测** `/root/libdir-fa-v2` | 已花完构建 | prefill | GAP §9 P1-3 |
| **N4** | P1-4 GDN x4 预填充 | TODO | 中（+167 行） | +2~2.3% PP | jusko `gated_delta_net.cu:172` |
| **N5** | P2-6 RMS_NORM+SCALE 融合 | TODO | 小 | 去 ~480 次 launch（host-bound 下纯赚） | jusko `norm.cu:556` |
| **N6** | 主机侧 metadata 缓存 + 状态指纹失效 | TODO | 中 | 直接打 53% 主机循环 | v100-skinny `gpu_model_runner.py:531-536,675-688` |
| **N7** | P2-selector: DFlash2 selector 上 GPU | TODO | 小（~80 行） | 4.3 ms/轮 = 7.7%；RNG 须 counter-based | ninfer `dflash2_selector_volta.cu` |
| **N8** | Long-ctx: GQA read-once（grid.z: B*H_Q -> B*H_kv） | TODO | 中 | 外测 128K 30.04 -> 200K 49.58 t/s | sglang `tilelang_fa_v100/_kernels_paged_decode.py:1-7` |
| **N9** | Long-ctx: prefill 尾块精确 split-KV（阈值表 Q<=2048 时 64/32/16/8/4/2） | TODO | 小 | 外测 Q=64/K=245760 **4.306 vs 40.675 ms (9.45x)** | sglang `_paged_adapter.py:95-115` |
| X1 | P0-2 layer split | **REFUTED** | - | 77.0 vs 55.1 ms/轮（慢 40%） | GAP §10 |
| X2 | P3-7 KV 压缩 | **REFUTED** | - | KV 只占每轮流量约 1% | GAP §9 |
| X3 | R1 push AR / A4 深切分 / TP2,4,6 / MoE / NCCL 调参 | **REFUTED — 但见 N0** | - | - | 见 §3 |

## 2. 依赖边（**这是本文件的核心**）

```
N0 ──> 决定 X3 整组是否可信
       (若可加性成立 => R1 的『省 4.8ms 主机零改善』是异常，X3 需重审)

N1 ──> 决定『KV 用 q8_0 还是 f16』   [链1]
       理由: 我们那条『KV 保持 q8_0(32K 中性, 64K f16 仅 +4%)』是在【没有 q8_0 TC 内核】的世界里测的;
              jusko 同 session: 无 TC 内核时 q8_0 比 f16 慢 3.8%(要付反量化税)

N1 ──> 决定『长上下文 KV 带宽 105 GB/s vs roofline 800』这个判断 [链1b]

N1 ──> 令 N3 的常量在 decode 上变得相关  [链4]
       理由: 当前 verify 走 TILE, 故 N3 只影响 prefill

N1 ──> 决定『n_max=7 最优』 [链3]
       理由: jusko 整套解码加速(q8 TC kernel/Q5_X4/Q6_W4R4)全是为 T=4 写的

『消灭 meta 税』 ──> 决定『TP3 最优 / 加卡不买带宽』 [链2]
       理由: 那些结论是在 meta 税随卡数增长的前提下测的(TP6 enqueue 高达 62 ms/轮)
       一旦消灭, 加卡的收益函数就变 —— 1cat 的 4 卡 + 整轮 fullgraph 正是那个 regime

N2 ──> 为 N1 提供动机证据（便宜的先做）
```

## 3. 条件结论登记册（**引用前必须看这里**）

| 结论 | 它依赖的前提 | 前提未满足时的正确说法 |
|---|---|---|
| KV dtype 保持 q8_0（64K f16 仅 +4%） | N1 | 「在**没有 q8_0 TC 内核**的条件下如此」 |
| 长上下文有效 KV 带宽只有 105 GB/s、远低于 roofline | N1 / N8 | 「可能是**冗余流量**（GQA 6 个 Q 头各读一遍）而非低效」 |
| n_max=7 最优 | 当前内核的 T 特化 | 「在当前内核条件下最优」 |
| TP3 最优 / 加卡不买带宽 | 当前 meta 税 | 「在当前主机税下如此」 |
| D256 FA 常量（N3）影响 prefill | —— | 目前仅 prefill；N1 后也影响 decode |
| 「V100 上 HMMA 天花板 1.36x」 | —— | 是 **Q8_0 在自己格式内的余量**，不是换格式的收益；换 4-5bpw 是 1.70-2.11x，但端到端**仅 +10~13%**。且**不适用于注意力** |

## 4. 关键路径（我推荐的下一个动作序列）

```
1. N0  读 /tmp/sync-ab.txt        <- 已完成/正在跑, 零成本, 决定 X3 可信度
2. N2  一行只读诊断                <- 最便宜, 为 N1 提供动机
3. N1  q8_0 KV 张量核注意力        <- 前置项, 做完后 链1/1b/3/4 全部解锁
4. N8+N9 长上下文两件             <- 收益最大的一块(9.45x / 30->49.6 t/s)
5. N6+N7+N5 主机侧三件            <- 打 53% 主机循环
```

## 5. 作业纪律（血的教训，必守）

### 5.1 ★ 实验排程事故与已固化的对策（Round 140）

**事故**：主线启动 `sync-ab.sh` 时，FA 子代理的第一轮 `llama-bench` 已在几秒前发射（其脚本含 `pkill`），
结果**主线的 sy0 臂被 KILL 作废**；且该臂的 29 GB 模型加载与主线的 sy1 臂前 3 分钟重叠 => **sy1 也可能受污染**。

**救回的原因**：`sync-ab.sh` 用的是 **A-B-B-A 交错**（baseline / SYNC / SYNC / baseline）。
前两臂一废一污，但**后两臂（sy1b + sy0b）都落在污染窗口之后，构成干净的一对** => 结论仍然可得。
=> **纪律：正式 A/B 一律用 A-B-B-A 交错，不要 A-B-A-B 或单对。** 它是对「中途污染」的冗余保护。

**已固化的对策**（子代理提出并被采纳，比『靠人记得等』健壮）：
```
1) 实验脚本【不主动 pkill 任何东西】：检测到 llama-bench / llama-server 或别的实验脚本
   (sync-ab.sh / layer-ab.sh / p60-ab-harness.sh) 就 MACHINE_BUSY_ABORT 退出。
2) 只回收【自己端口】的 llama-server；遇到别人的 server 直接 abort。
3) 启动任何实验前先查 /tmp/*.txt 里有没有别的任务在跑（SYNC_AB_DONE / LAYER_AB_DONE 等）。
```

### 5.2 其余纪律

```
- 测量必须独占机器。启动任何实验前先看 /tmp/*.txt 里有没有别的任务在跑。
  已踩过两次: layer-ab 与 fa-ab 相撞; sync-ab 与 fa-ab 相撞(第二次在发生前拦住)。
- 端口守卫: 每臂 pkill -9 -x llama-server 并等端口可连; 每臂用不同 PORT。
- 构建/跑测一律写成 .sh 上传后执行, 不要用 bash -c '长串'(引号会被 Windows argv 吃掉)。
- 远端命令里不要出现双引号/$(...)/#。
- 每次报告同时给 AL 与 ms/轮; 跨 split 模式只能比 ms/轮(sha256 会变)。
- 禁止把『条件结论』当『已定论』引用 —— 查 §3。
```