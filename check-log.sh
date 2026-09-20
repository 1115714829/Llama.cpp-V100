#!/bin/bash
set -uo pipefail
echo "=== full server log (last 45) ==="
journalctl -u llama-server -n 45 --no-pager 2>/dev/null | tail -45
echo "=== service now ==="
systemctl is-active llama-server
echo "=== GPU 2/5 mem ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 5
echo LOG_DONE
