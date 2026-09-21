#!/bin/bash
echo "=== GPU2 mem/util ==="
nvidia-smi -i 2 --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=== llama-cli procs ==="
pgrep -af llama-cli
pkill -f llama-cli
sleep 2
echo "=== after kill ==="
nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader
echo CLEANUP_DONE
