#!/bin/bash
set -uo pipefail
BENCH=/root/llm/test/v100-opt/llama.cpp/build/bin/llama-bench
MODEL=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
echo "=== kill 256k bench (free GPU2/5) ==="
pkill -f "llama-bench" 2>/dev/null
sleep 6
echo "=== GPU 2/5 after kill ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 5
export CUDA_VISIBLE_DEVICES=2,5
echo "=== ORIG build, 512 ctx (pp512/tg128) ==="
$BENCH -m "$MODEL" -p 512 -n 128 -b 512 -ub 512 -ngl -1 -fa 1 \
  --tensor-split 1/1 --main-gpu 0 2>&1 | grep -E "pp512|tg128|test |error"
echo "=== ORIG build, 2048 ctx (pp2048/tg128) ==="
$BENCH -m "$MODEL" -p 2048 -n 128 -b 512 -ub 512 -ngl -1 -fa 1 \
  --tensor-split 1/1 --main-gpu 0 2>&1 | grep -E "pp2048|tg128|test |error"
echo NORM_DONE
