#!/bin/bash
set -uo pipefail
echo "=== curl process (current test running?) ==="
pgrep -af "curl.*8081" 2>/dev/null | head -2
if pgrep -f "curl.*8081" >/dev/null 2>&1; then echo "CURL_RUNNING"; else echo "CURL_DONE"; fi
echo "=== journal (last 12 lines, current test) ==="
journalctl -u llama-server-test -n 12 --no-pager 2>/dev/null | tail -12
echo "=== GPU 2/5 ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 2
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 5
echo CCT_DONE
