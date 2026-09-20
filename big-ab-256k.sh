#!/bin/bash
# 256k same-source interleaved A/B: pristine vs C4, MTP off (llama-bench has no spec decode),
# 2 GPUs inside one NUMA node (0,1) so the peer path is NV2 rather than SYS.
set -uo pipefail
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
LOG=/tmp/big-ab-256k.log
export CUDA_VISIBLE_DEVICES=0,1

: > "$LOG"
{
  echo "=== 256k same-source A/B  start=$(date -Is) ==="
  echo "config: -p 262144 -n 128 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 --tensor-split 1/1 --main-gpu 0"
  echo "gpus: $CUDA_VISIBLE_DEVICES (same NUMA node 0)"
  echo "--- lib validity (md5 of the lib that carries the change) ---"
  md5sum /root/libdir-pristine/libggml-cuda.so.0.24.0 /root/libdir-c4/libggml-cuda.so.0.24.0
  echo "--- gpu before ---"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
} >> "$LOG"

for round in 1 2; do
  for v in pristine c4; do
    {
      echo ""
      echo "=== round=$round variant=$v  start=$(date -Is) ==="
    } >> "$LOG"
    LD_LIBRARY_PATH=/root/libdir-$v /root/libdir-$v/llama-bench \
      -m "$M" -p 262144 -n 128 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 \
      --tensor-split 1/1 --main-gpu 0 -r 1 >> "$LOG" 2>&1
    {
      echo "--- end round=$round variant=$v  $(date -Is) ---"
      grep -F -e "pp262144" -e "tg128" "$LOG" | tail -2
    } >> "$LOG"
  done
done

{
  echo ""
  echo "=== ALL DONE $(date -Is) ==="
  grep -F -e "pp262144" -e "tg128" "$LOG"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
} >> "$LOG"
