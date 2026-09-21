#!/bin/bash
set -uo pipefail
echo "=== 256k orig progress (tail) ==="
tail -5 /tmp/stress-orig.out
echo "=== GPU 2/5 memory ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 5
echo "=== is bench running? ==="
if pgrep -f "llama-bench" >/dev/null 2>&1; then echo "RUNNING"; else echo "DONE"; fi
echo PROG_DONE
