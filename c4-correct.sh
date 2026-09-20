#!/bin/bash
set -uo pipefail
echo "CORR_START"
cd /root/llm/test/v100-opt/llama.cpp
echo "--- llama-cli -c 512 (nwarps=2 build) ---"
CUDA_VISIBLE_DEVICES=2 timeout 120 ./build/bin/llama-cli \
  -m /root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf \
  -p "The capital of France is" -n 16 --temp 0 -c 512 --no-display-prompt 2>&1 | tail -5
echo "EXIT=${PIPESTATUS[0]}"
echo "CORR_DONE"
