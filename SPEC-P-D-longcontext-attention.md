# SPEC P-D：长上下文 decode 的 attention 内核（256K 场景）

> 目的：256K 上下文下 decode 的 attention 附加成本约 **28 ms/token**（实测斜率 0.113 µs/KV-token × 256K），
> 而 KV 带宽 roofline 仅约 **5.7 ms/token**（每卡 ~4.6 GB/token）=> **约 5× 余量**。
> 目标：**256K decode 34.13 t/s → ≥60 t/s**（约 2×）。

## 1. 已确认的事实（本轮实测，勿再重复验证）
- **不是并行度问题**：探针 `GGML_CUDA_FA_SPLIT_FLOOR` 在 `-d 131072` 下 0/64/256/1024 = **26.85 / 24.59 / 22.71 / 22.43 t/s（单调变差）**
  => 上游"一波切分"启发式在本形状就是最优（`fattn-common.cuh:1126-1199`，切分度由 occupancy/wave 决定，不随 n_kv 增长）✓。
  A4 已证伪"更深切分"路线；探针归档于 `patches/0004`。
- **不是带宽总量问题**：256K 每卡 KV 流量约 4.6 GB/token（4 个 KV 头按 2+1+1 切）。
- **派发路径**：M=1 → **VEC**（`fattn.cu:645`，natively 读 q8_0，无反量化）；M=2..8 → **TILE**（`:648`，**强制把整段 KV 反量化成 f16**）；M≥9 → MMA。
  => 长上下文 decode 的 M=1 走 VEC，其**有效带宽仅约 105 GB/s**（54 ms/token 对 4.6 GB）；M=8（投机）走 TILE 且多付一次整段反量化。
- 1cat 的参考实现：`FLASH_ATTN_V100` 的 **partition split-KV**（`flash-attention-v100/kernel/flash_decode_paged.cu:1038,1061-1076`；
  partition size 256/512/**1024（seq_len≥32768）**，`flash_attn_interface.py:19-20,599-608`）+ 归约 kernel；D256 另有专用内核与 smem K ping-pong 补丁。

## 2. 建议的实施顺序（每步独立可验证）
1. **先量化当前内核的瓶颈类型**（不改代码）：用 `llama-bench -d {8192,32768,131072,262144}`（同一 build）得到
   "每 KV-token 的 ms"曲线；再用 `test-backend-ops perf -o FLASH_ATTN_EXT`（`build-nccl/bin/`，注意其用例最长 kv=1024，只能作为微基准辅助）。
   判据：确认是"每 token 时间的线性项过大"（吞吐问题）而非"固定项"。
2. **小步改造 VEC 内核的数据通路**（低风险，先做这个）：把 KV 加载向量化（当前按 `nb11/nb21` 逐元素 stride 访问），
   并检查 `nthreads_KQ`（D=256/q8_0 下为 32）是否让每个 KV 位置的 dot 只由少数线程负责 => 提高 ILP。
   **门**：同配置 greedy sha256 **逐位一致**（纯数据通路改造不应改变数值）+ 128K/256K decode 时间下降。
3. **若第 2 步不足**：才考虑写"partition split-KV + 显式归约"的 D256/GQA=6/q8_0 专用 decode 内核（1cat 路线，工作量大，需另行批准）。

## 3. 验收门（硬性）
- 256K decode **≥60 t/s**（现状 34.13；M=1 普通解码现状 18.86 t/s）；
- greedy sha256 逐位一致（第 2 步应一致；第 3 步若改变求和顺序则不要求与旧值相同，但必须自洽且 AL 不劣化）；
- 长上下文标尺复测（`/root/lc-spec.sh`，3 prompt、种子固定、报 tg/AL/中位数）；
- 8K 主口径不得回归（98.1–98.9 t/s / AL 5.55 / 57.6 ms/轮）；
- 测量独占机器、正式数字带 drop_caches、构建三查 + 四库 md5 + 标记串。

## 4. 风险与止损
- VEC 内核是**所有长上下文 M=1/2 解码的公共路径** => 任何改动都必须过逐位 sha256 门；
- 若第 2 步（向量化 + ILP）拿不到 ≥15% 改善，则本项收益主要在第 3 步（专用内核）=> 需要重新评估投入，或把优先级让给 P-A/P-B。
