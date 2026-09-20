#!/bin/bash
set -uo pipefail
cd /root/llm/test/v100-opt/llama.cpp
echo "=== mmvq C4 present? ==="
grep -c "MMVQ_PARAMETERS_VOLTA" ggml/src/ggml-cuda/mmvq.cu
echo "=== bench tg128 + GPU util sampling (bg) ==="
CUDA_VISIBLE_DEVICES=2 ./build/bin/llama-bench -m /root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf -p 0 -n 128 > /tmp/bench.log 2>&1 &
BENCH_PID=$!
# sample util + mem every 0.5s; the tg runs after the model load
peak=0
for i in $(seq 1 90); do
  line=$(nvidia-smi -i 2 --query-gpu=utilization.gpu,memory.used --format=csv,noheader)
  u=$(echo "$line" | awk -F', ' '{print $1}')
  echo "t=$((i))s  util=${u}%  mem=$(echo "$line" | awk -F', ' '{print $2}')"
  if [ "$u" -gt "$peak" ] 2>/dev/null; then peak=$u; fi
  sleep 0.5
done
wait $BENCH_PID
echo "=== peak util during run: ${peak}% ==="
echo "=== bench result ==="
tail -3 /tmp/bench.log
echo GPUUTIL_DONE
