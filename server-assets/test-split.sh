#!/bin/bash
# Check the tensor-split syntax: does the model split across GPU2+5?
set -uo pipefail
cd /root/llm/test/v100-opt/llama.cpp
MODEL=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
echo "=== A: --tensor-split 1/1 (slashes) ==="
CUDA_VISIBLE_DEVICES=2,5 ./build/bin/llama-bench -m "$MODEL" -p 512 -n 32 --tensor-split 1/1 > /tmp/splitA.log 2>&1 &
BPID=$!
peak=0
for i in $(seq 1 30); do
  m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 2 | tr -d ' MiB')
  m5=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 5 | tr -d ' MiB')
  echo "  t=$((i))s GPU2=${m2} GPU5=${m5}"
  if ! kill -0 $BPID 2>/dev/null; then break; fi
  sleep 0.5
done
echo "  A result:"; grep -E "tg32|error" /tmp/splitA.log | tail -3
echo "=== B: --tensor-split 1,1 (commas) ==="
CUDA_VISIBLE_DEVICES=2,5 ./build/bin/llama-bench -m "$MODEL" -p 512 -n 32 --tensor-split 1,1 > /tmp/splitB.log 2>&1 &
BPID=$!
for i in $(seq 1 30); do
  m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 2 | tr -d ' MiB')
  m5=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 5 | tr -d ' MiB')
  echo "  t=$((i))s GPU2=${m2} GPU5=${m5}"
  if ! kill -0 $BPID 2>/dev/null; then break; fi
  sleep 0.5
done
echo "  B result:"; grep -E "tg32|error" /tmp/splitB.log | tail -3
echo SPLIT_DONE
