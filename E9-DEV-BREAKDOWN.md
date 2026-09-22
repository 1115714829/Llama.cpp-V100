# E9: dev 拆分 (R192) - 三卡对称, 86% 便宜 + 11.7% 长尾

## 实验 (两臂同源, tag dev1/dev2, 端口 8281/8282, NPRED=512 NODROP, GGML_META_HOST_TIMING=1)
BUILD_RC=0 | ERR_COUNT=0 | 115 Built target | MARK_DEV_HIST=1; 两臂 sha256 = f3edac19... 一致;
MEDIAN_TG = 97.81 / 97.09 (与基线同档); 服务 inactive/active/active 未变; GPU 全空。

## 读数 (末次 [META], 两臂一致)
    calls=832 sub/call=47.6 ar/call=46.6 | total=10.495 loop=9.249 dev=7.034
      dev0=2.441 dev1=2.290 dev2=2.292 ar=2.211 ms/call | prologue=1.247 (11.9%)
      dev_hist_us n=118884 min=6 med=24 max=16367
      <20=50823  20-60=51037  60-150=3061  >150=13963 | nodes/sub=40.0

## 结论
1. 三卡完全对称 (2.441 / 2.290 / 2.292) => dev 不是掉队卡或负载不均, 是每卡等量的主机启动开销;
2. 每次 compute 中位 24 us, n/call = 142.9 (与 sub/call x 3 = 142.8 吻合);
3. 双峰分布: <20us 43% + 20-60us 43% = 86% 便宜; >150us 占 11.7% (13963 次);
4. 反解总量 (832 calls x 7.034 ms = 5.85 s): 三个小桶约 2.7 s => 长尾吃掉约 3.1 s (~53%), 均值约 224 us/次
   (max 16.4 ms, 疑为图捕获的一次性开销);
5. 与 [GRAPH] 的 12.6% direct 精确对应 (11.7% vs 12.6%) => 原先对 direct 单价的估计 (281 us) 基本正确 (实测约 224 us)。

## 与 E7 的调和 (关键)
E7 删掉约 20% 的 direct (3539 次 x 224 us 约 0.79 s, 占臂时约 5.5%) 却零收益 =>
长尾时间被 GPU 执行重叠掉了, 这正是 E8 里那 45% 未穿透部分的来源。
=> 靠 replay 治长尾不可靠; 减固定次数 (减少子图数 / 合并切分) 才直接打在 <60us 那 86% 上。

## 折算 (按 E8 杠杆 1 ms/call 约等于 1.8 ms/轮)
| 成分 | ms/call | 折合 ms/轮 | 性质 |
|---|---:|---:|---|
| 长尾 >150us (11.7%) | ~3.1 | ~5.6 | 可被 GPU 隐藏 (E7 证明) => 不可靠 |
| 主体 <60us (86%, 143 次/call) | ~2.4 | ~4.4 | 固定次数 => 减子图数直接命中 |
| 中间 60-150us | ~0.3 | ~0.5 | 同主体 |
| 合计 dev | 7.03 | ~12.7 | |
| ar | 2.21 | ~4.0 | E1 图内/设备侧 AR (46.5 次 x 47.6us) |
| prologue | 1.25 | ~2.25 | 重建 (N6a/N6b; 内容哈希路线已证不可行) |
| 总计 | 10.49 | ~18.9 | = 穿透到关键路径上的总量 (与 E8 的 19.2 吻合) |

## 下一步 (优先级已由本实验确定)
1. 减少子图数 / 合并切分 (打 <60us 主体的 2.4 ms/call) - 优先于治长尾;
2. E1 图内/设备侧 AR (打 ar 的 2.21 ms/call) - 46.5 次主机启动可整体消失;
3. prologue 1.25 ms/call (N6a/N6b)。


## ★ 追加 (R193): 切图规则已读通 => 优先级 1 与 2 是同一件事的两面

代码事实（ggml-backend-meta.cpp）:
  2270: const bool new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
  2275: const int i_delayed = get_i_delayed(i);   // 已有的"延迟 AllReduce"机制, 目前只在 MoE 分支启用
  2292: i = i_delayed;  ... n_subgraphs++; i_start = i + 1;
  2553: if (n_backends > 1 && i < n_subgraphs - 1) { ... comm_allreduce(comm_ctx, nodes.data()); }

=> 子图数 = PARTIAL 轴节点数 + 1 = 46.5 + 1, 且**每个边界恰好一次 allreduce** (与实测 ar/call 46.6 逐位吻合)。
=> 上游**已经有**"延迟 AllReduce"的机制 (get_i_delayed, 用于 MoE), 只是未在一般情形启用。

### 由此得到的实现路径 (优先级 1 与 2 合并为一件事)
把 get_i_delayed 的延迟推广到一般情形, 并把被延迟的多次 allreduce **合并成一次分组调用**
(NCCL group 语义天然支持多张量: 把 K 个逻辑张量 x n_backends 一起提交):
  - ar 的 46.5 次主机启动 -> 少数几次 (打 2.21 ms/call => 约 4.0 ms/轮);
  - 子图数同步下降 => 图启动次数下降 (打 <60us 主体的 2.4 ms/call => 约 4.4 ms/轮);
  - 两者叠加约 8.4 ms/轮, 且**不依赖任何"治长尾"** (E7 已证那条被 GPU 隐藏, 不可靠)。


## ★★ 追加 (R194): comm 接口语义已确认 => AR 只能靠"减次数", 不能靠"每次更便宜"

事实 (ggml-cuda.cu:1000-1060, ggml-backend.h:210):
  typedef bool (*ggml_backend_comm_allreduce_tensor_t)(void * comm_ctx, struct ggml_tensor ** tensors);
  ggml_backend_cuda_comm_allreduce_nccl():
    const int64_t ne = ggml_nelements(tensors[0]);
    const size_t n_backends = comm_ctx->backends.size();
    for (i < n_backends) { GGML_ASSERT(tensors[i]); GGML_ASSERT(ggml_nelements(tensors[i]) == ne); }
    if (n_backends == 3 && ne < 131072) { ncclGroupStart(); 3x ncclAllReduce(ncclFloat); ncclGroupEnd(); }
    else { to_bf16 暂存到 tmp[i]; ncclGroupStart(); 3x ncclAllReduce(ncclBfloat16); ncclGroupEnd(); }

=> tensors 数组的语义是 **一个逻辑张量 x n_backends 个设备切片**, 不是多个逻辑张量;
   ncclGroupStart/End **已经**用于把 3 卡的那一次归约合成一组。
=> 单次 AR 的 47.6 us = 3 次 ncclAllReduce 主机调用 + group 开销 => **不减次数就压不动**。
   本模型每轮张量: 5120 x 8 = 40960 元素 < 131072 => 走 FP32 直通分支 (无 bf16 暂存开销)。

### 可行形态 (优先级 2 的正确实现)
把 K 个 PARTIAL 输出的归约**合并成一次调用**: 各卡的 K 个切片拼进一段连续缓冲, 再一次 ncclAllReduce,
再把结果散回。数学等价 (求和逐元素且各张量互不影响)。
  - 利好: bf16 分支本来就有 tmp[i] 暂存 + 格式转换的基础设施, 可复用;
  - 代价: 需要暂存拷贝 (可被带宽吸收: 5120x8x4B = 160 KB/张量 x 46.5 = 7.4 MB/轮, 微不足道),
    且 comm 接口要能表达 K x n_backends (签名扩展或 K 次调用放进同一个 group);
  - 前提: 归约必须在任何消费者读取前完成 => 与"延迟 AllReduce"(get_i_delayed) 配合使用;
  - 预期: 若把 46.5 次压到 4-6 次, 按 E8 杠杆 (1 ms/call => 1.8 ms/轮) 可拿下 ar 的 1.8-1.9 ms/call
    => 约 3.2-3.4 ms/轮; 同时子图数从 47.5 降到 5-7 => 图启动次数骤降, 再拿 <60us 主体的一大块。

### 风险与前提 (实施前必须确认)
1. 延迟 allreduce 要求被延迟区间内的张量在归约前不被消费 (数据依赖必须成立); 代码注释已指出与
   "zero-sized tensor slices" 的交互: 若某卡是零切片, 延迟期间必须给相关节点清 GGML_TENSOR_FLAG_COMPUTE
   (2290 行附近已有此处理, 可复用);
2. 分组 allreduce 需要 comm 层支持多张量提交 (现有签名是每设备一张量); 若 1cat 的 NCCL 封装已支持 group,
   则该改动量可控;
3. 必须保持正确性门 (greedy sha256 f3edac19...) 与同源 A/B。

## 现场
- 补丁存档 wip-meta-dev-breakdown.patch; 服务器 libdir-instr 为该构建 (libggml-base.so md5 d283439ca8f84811223b5f58206eaaba);
- 两臂 STATUS=OK, ARM_RC=0, 无 LAUNCH_FAILED; GPU 全空, 无锁, 无残留进程。