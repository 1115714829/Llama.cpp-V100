# SM70-ATTN 移植评估（R268）

> 对象：`v100-refs/sm70-attn`（1Cat Split-D N32 D256 + SplitKV3，prefill-only）
> 对齐：R264/R266/R267 —— 预填充墙 = 长 KV `FLASH_ATTN_EXT`（真实形状约 370 ms）
> 状态：评估完成。实现属新子系统，按红线需你批准后再改码。

## 1. 结论

值得移植，但不是零成本 drop-in。内核对症（Volta + D=256 + 长 KV 预填 + SplitKV3 + f32 输出）；**硬缺口是生产 KV = q8_0，而插件只收 F16/Q4_0**。声明 176k prefill +39.9% 为 C 级，须我们形状同源 A/B 重测。

## 2. 形状对照

| 维度 | 我们 | sm70-attn 门控 | 判定 |
|---|---|---|---|
| 架构 | V100 sm_70 | `cc == VOLTA` | 命中 |
| D / GQA | 256 / 6 | ne[0]==256，GQA 整除 | 命中 |
| 批 | 预填 n_q=2048 | Q->ne[1] >= min_q（默认 256） | 预填命中；decode/MTP 走 stock |
| mask | 有 | 必须有 mask | 命中 |
| KV | **q8_0** | **仅 F16 / Q4_0** | **硬缺口** |
| 回滚 | — | `LLAMA_SM70_D256=0` | 好 |

## 3. 推荐落地路径

**路径 A（先做）**：q8_0 → 沿用现役 to_fp16 额外槽 → sm70 核吃 f16 K/V。
- R267：to_fp16 估 &lt;1 ms，不是墙；不必先写 q8 原生内核。
- 门控扩 `kv_ok` 放行 q8_0，alloc_size 与 need_f16_K/V 谓词对齐。

**路径 B（后做）**：JS2 `fattn-q8-volta.cuh` 核内 smem q8_0→f16，省镜像显存。

## 4. 集成点

1. `ggml_cuda_flash_attn_ext` / `get_best_fattn_kernel` 前挂钩（对齐 `sm70-hook.patch`）。
2. 拷入 `fattn-sm70-d256.cu` + `fattn-sm70-d256-kernel.cuh`（+ CuTe/cutlass sm70 头）。
3. 只在 `/root/llm/test/v100-opt/llama.cpp` 改；构建后 `strings` 标记串校验。
4. 默认 env 不设 = 行为可关；A/B 显式 `L=/root/libdir-instr`。

## 5. 验收

- 算子级：真实形状 FLASH_ATTN_EXT 370 ms → 目标 &lt;250 ms（&lt;15% 则停）。
- 端到端：pp32768 / pp131072，q8_0 KV，≥2 臂报 ±；B3 解码不回退。
- 新核会改累加序 → **greedy sha256 必须重立门值**，不得沿用 f3edac19。

## 6. 实施顺序（批准后）

A0 零改码基线已齐（R267）→ A1 搬代码+门控+q8→f16 槽 → A2 算子 A/B → A3 端到端 pp → A4（可选）核内 q8。
