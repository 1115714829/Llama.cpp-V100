#!/bin/bash
# Quick same-source A/B discriminator: pristine vs C4, single GPU, no MTP.
set -uo pipefail
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
export CUDA_VISIBLE_DEVICES=0

echo "=== GPU state ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo ""

for v in pristine c4; do
  B=/root/bin-b11053-$v-bench
  echo "########## variant=$v  bin=$B ##########"
  "$B" --version 2>&1 | grep -iE "version"
  ls -la "$B"
  echo "--- pp512 / tg128 (3 reps) ---"
  "$B" -m "$M" -p 512 -n 128 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 -r 3 2>&1 | tail -5
  echo ""
done
echo P3_DONE
