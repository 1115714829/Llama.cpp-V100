#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R265: record the post-restore library md5 honestly."""
import io

LEDGER = r"F:\vllm+llama.cpp\1cat-vllm-v100-study\BASELINE-LEDGER.md"
ANCHOR = "### R262 "

ENTRY = """### R265 ★ **状态复位（服务器）: 源码回到纯净，但 libggml-cuda 的 md5 与 R257 记录值不同（需下一轮知情）**

- 已还原：`mmvq.cu = 19c984c953d365fa0848a2c389239abc`（= 本地/HEAD）、`vecdotq.cuh = 275e009da0d98d02eb3467387efe6e9c`、
  `GGML_CUDA_MMVQ_WIDE` 标记串在库里计数 **0**、`ggml-backend.cpp = 967af7b5…`、`libggml-base.so = a8de7c48…`（= R257 值，可复现）。
- ⚠️ **但重建后的 `libggml-cuda.so.0.24.0 = ad317f8b757a4540f533bd7ec59b47d2`，不是 R257 记的 `0fb36620…`**。
  源码 md5 一致、算子级行为一致（`q8_0 m=4096 k=14336`: n=1 **82.81** / n=5 **95.20** / **n=8 112.37** µs，与 R236 的 82.10/111.66 同噪声带），
  说明差异来自**重编译本身**（R258 那次被打断的全量 CUDA 重建把若干 `.cu` 对象重编过，之后链接出的 .so 与 01:45 那次不再逐字节相同）。
- ⇒ 规矩：**引用"基线库 md5"时必须写清是哪一次构建**；行为等价要用算子级/门值复核，不能只看 md5 相等。

"""

with io.open(LEDGER, "r", encoding="utf-8", newline="") as f:
    text = f.read()
if "### R265 " in text:
    print("ALREADY")
else:
    nl = "\r\n" if "\r\n" in text else "\n"
    text = text.replace(ANCHOR, ENTRY.replace("\n", nl) + ANCHOR, 1)
    with io.open(LEDGER, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print("INSERTED R265")
