#!/bin/bash
set -uo pipefail
cd /root/llm/test/v100-opt/llama.cpp
MODEL=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
echo "=== test 1: context 512 (quick) on GPU2+5, tensor-split 1,1 ==="
CUDA_VISIBLE_DEVICES=2,5 ./build/bin/llama-bench -m "$MODEL" -p 512 -n 128 --tensor-split 1,1 2>&1 | tail -30
echo "=== GPU mem after test 1 (model should be freed) ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 5
echo TEST1_DONE
