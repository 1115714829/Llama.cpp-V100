#!/bin/bash
set -uo pipefail
echo "=== C4 256k test (journal, last 5 min) ==="
journalctl -u llama-server-c4 --since "5 min ago" --no-pager 2>/dev/null | grep -iE "prompt eval time|eval time|total time|draft acceptance|n_gen|prompt processing" | tail -8
echo "=== C4 curl running? ==="
if pgrep -f "curl.*8082" >/dev/null 2>&1; then echo "C4_CURL_RUNNING"; else echo "C4_CURL_DONE"; fi
echo "=== GPU 2/5 ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 2
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 5
echo C4C_DONE
