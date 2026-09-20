#!/bin/bash
# Run the 256k context + sample GPU2/5 memory to catch the OOM + required size.
set -uo pipefail
cd /root/llm/test/v100-opt/llama.cpp
MODEL=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
# Launch the bench in the background.
CUDA_VISIBLE_DEVICES=2,5 ./build/bin/llama-bench -m "$MODEL" -p 262144 -n 128 --tensor-split 1,1 > /tmp/bench256.log 2>&1 &
BPID=$!
# Sample GPU2 + GPU5 memory every 0.5s for ~20s (context creation is fast).
for i in $(seq 1 40); do
  m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 2)
  m5=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 5)
  echo "t=$((i))s  GPU2=${m2}MiB  GPU5=${m5}MiB"
  if ! kill -0 $BPID 2>/dev/null; then echo "bench exited at t=$((i))s"; break; fi
  sleep 0.5
done
echo "=== bench output (error) ==="
grep -iE "error|failed|out of memory|oom|alloc|KV|cache" /tmp/bench256.log | head -20
echo "=== last 8 lines ==="
tail -8 /tmp/bench256.log
echo MEM_DONE
