# fa-v100-verified.md — FlashAttention 线：独立核实过的事实（t5b）

> 目的：把 1cat 的 FA-V100 与 llama.cpp `fattn.cu` 的对照**只记录我亲自核实过、带行号**的部分，
> 与用户提供的 `FlashAttention-V100-reference.md`（宣传口径）分开，避免把宣传当事实。
> 日期 2026-09-20，base b11053。

## 1. 已核实：1cat 侧

| 事实 | 证据 | 备注 |
|---|---|---|
| 有 `FlashV100Traits` 模板，按 `BLOCK_M`/`BLOCK_N` 参数化 | `flash-attention-v100/kernel/flash_v100_traits.cuh:7`、`:36-37` | |
| **每线程取 8 个元素**（half 时 = 128-bit 宽加载） | `flash_v100_traits.cuh:42` `static constexpr int kGmemElemsPerLoad = 8;` | 与参考文档 "128-bit vectorized access" 一致 |
| Gmem 加载按 `(tidx % kGmemThreadsPerRow) * kGmemElemsPerLoad` 分列 | `flash_v100_traits.cuh:55-58` | |
| **decode 路径用 256 线程**（不是 512） | `flash_decode_paged.cu:41`、`flash_decode_turboquant.cu:13`：`kThreadsPerBlock = 256` | **与参考文档"16 warp / 512 thread"不符** —— 该说法要么指 prefill traits，要么不准；**以代码为准** |
| FP8 KV 转换单独成 kernel，256 线程 | `fp8_kv_bridge.cu:16` | |

> 未核实 / 不要引用：参考文档里的 TFLOP/s 数字、"16 warp / 512 thread"、
> `h3/` 稀疏与 Page4 的具体收益倍数 —— 这些我没有独立证据。

## 2. 已核实：llama.cpp `fattn.cu` 的 Volta 相关结构

- MMA 派发阶梯：`ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1`（`:133`）按 `Q->ne[1]` 选 M-tile
  **8 / 16 / 32 / 64**，每个再除以 `ncols2`；`switch_ncols2`（`:170`）按 GQA 比选 `ncols2` ∈ {8,4,2,1}。
- **M=8 那一档被 Turing 门控**：`:147`
  ```cpp
  if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) { ... case<..., 8/ncols2, ncols2> ... }
  ```
  -> 小 M（decode 方向）的 MMA 档在 Volta 上**故意不派发**。
- 另一处 Turing 特判：`:160` `... || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) || ...`
- 顶层分流（Volta 判定）在 `ggml_cuda_get_best_fattn_kernel`（L520-690 区间，
  `volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72`）。

**与已知实验的关系**：`QWEN.md` §0.5 记录的那次 C1（强行让 decode 走 MMA_F16）**实测崩溃**，
原因是 arch 700 下该 MMA 配置没有可用实例化。上面 `fattn.cu:147` 的 Turing 门控正是这个现象的来源，
**独立印证了 C1 的崩溃原因**（不是环境问题，是代码本来就没给 Volta 这条 M=8 路）。

## 3. 由此得到的结论

1. **FA 线的"缺口"是明确的**：llama.cpp 在 Volta 上把 decode 的小 M-tile MMA 挡掉了（`:147`），
   而 1cat 的全部主张就是"别退回 SIMT，想办法喂满 Volta 的 TC"。
2. **但不要把 C1 再做一遍**：那不是"改一行阈值"，需要**为 arch 700 实例化**对应 MMA case
   （编译期 `if constexpr`），并验证寄存器/共享内存是否可行 —— 属于中等改动，风险高于候选②。
3. **FA 的收益天花板本就有限**：目标模型 64 层里只有 16 层走 `fattn.cu`（48 层是 GDN），
   即使 FA 快 2x，端到端也只动那 1/4 部分。**所以 FA 应排在 MMQ/交叉点之后**。

## 4. 待办（FA 线）
- [ ] 核 `switch_ncols1` 里 8/16/32/64 各档在 Volta 上的**实际可用性**（哪些是 `if constexpr`
      真的会为 arch 700 实例化、哪些被 runtime 门控挡住），确定"放开小 M"到底缺哪几个实例化。
- [ ] 评估 KV 宽加载：llama.cpp 的 FA KV 读取当前是多少位；是否有 1cat 那种 128-bit 合并加载的余地。
      （参考文档的 `kGmemElemsPerLoad=8` 在 1cat 侧已核实，但 llama.cpp 侧的对应实现**尚未核实**。）
