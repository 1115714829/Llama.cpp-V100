#!/bin/bash
set -uo pipefail
echo "=== 128k bench: status + result ==="
if pgrep -f "llama-bench" >/dev/null 2>&1; then echo "RUNNING -> killing"; else echo "not running"; fi
echo "--- 128k log (pp/tg result) ---"
grep -E "pp131072|tg128|error|CUDA" /tmp/bench128.log | tail -6
echo "=== kill any llama-bench (free GPU2/5) ==="
pkill -f "llama-bench" 2>/dev/null
sleep 3
echo "=== GPU 2/5 after cleanup ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 5
echo CLEAN_DONE
