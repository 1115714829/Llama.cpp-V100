#!/bin/bash
# Fast gguf metadata check (no model load): read header, grep for arch/MTP/NextN keys.
set -uo pipefail
MODEL=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
echo "=== gguf metadata keys (first 2MB strings) ==="
head -c 2000000 "$MODEL" | strings -n 4 | grep -iE "architecture|nextn|mtp|predict|layer|block_count|head_count|head_size|embd|attn" | head -50
echo "=== count of nextn/mtp/predict in header ==="
head -c 2000000 "$MODEL" | strings -n 4 | grep -icE "nextn|mtp|predict"
echo GGUF_DONE
