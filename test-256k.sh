#!/bin/bash
set -uo pipefail
cd /root/llm/test/v100-opt/llama.cpp
MODEL=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
echo "=== 256k context (262144) direct run, FULL stderr (context creation fails fast) ==="
CUDA_VISIBLE_DEVICES=2,5 ./build/bin/llama-bench -m "$MODEL" -p 262144 -n 128 --tensor-split 1,1 2>&1 | grep -viE "^\s*$" | tail -45
echo "256K_DONE"
