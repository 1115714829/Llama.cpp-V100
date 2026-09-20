#!/bin/bash
set -uo pipefail
echo "=== 256k orig prefill progress (journal) ==="
journalctl -u llama-server-test --since "3 min ago" --no-pager 2>/dev/null | grep -iE "prompt eval|eval time|total time|task |slot |error" | tail -8
echo "=== GPU 2/5 (KV growing) ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 2
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 5
echo "=== curl done? ==="
if [ -s /tmp/resp256k.json ]; then echo "RESPONSE_SAVED"; tail -c 200 /tmp/resp256k.json; else echo "still running"; fi
echo C256_DONE
