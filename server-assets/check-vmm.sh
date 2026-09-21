#!/bin/bash
set -uo pipefail
cd /root/llm/test/v100-opt/llama.cpp
MODEL=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
echo "=== llama-bench options: vmm / kv / memory / fit ==="
./build/bin/llama-bench --help 2>&1 | grep -iE "vmm|kv|memory|fit|offload|cache-type" | head -20
echo "=== VMM env var / flag in source ==="
grep -rn "NO_VMM\|vmm_enabled\|GGML_VMM\|cudaVmm" ggml/src/ggml-cuda/ggml-cuda.cu ggml/src/ggml-cuda/common.cuh 2>/dev/null | head -12
echo "=== test 128k (131072) context, memory trace (1s samples) ==="
CUDA_VISIBLE_DEVICES=2,5 ./build/bin/llama-bench -m "$MODEL" -p 131072 -n 128 --tensor-split 1/1 > /tmp/bench128.log 2>&1 &
BPID=$!
peak2=0; peak5=0
for i in $(seq 1 90); do
  m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 2 | tr -d ' MiB')
  m5=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 5 | tr -d ' MiB')
  if [ "$m2" -gt "$peak2" ] 2>/dev/null; then peak2=$m2; fi
  if [ "$m5" -gt "$peak5" ] 2>/dev/null; then peak5=$m5; fi
  echo "  t=$((i))s GPU2=${m2} GPU5=${m5}"
  if ! kill -0 $BPID 2>/dev/null; then echo "  bench exited at t=$((i))s"; break; fi
  sleep 1
done
echo "  peak GPU2=${peak2} GPU5=${peak5}"
echo "  128k result:"
grep -E "pp131072|tg128|CUDA error|failed" /tmp/bench128.log | tail -6
echo VMM_DONE
