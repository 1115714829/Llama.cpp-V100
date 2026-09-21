#!/bin/bash
set -uo pipefail
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 6
  P=$(ss -ltn 2>/dev/null | grep -c 8080)
  if [ "$P" -gt 0 ]; then echo "PORT_8080_UP after $((i*6))s"; break; fi
done
echo "=== service ==="
systemctl is-active llama-server
echo "=== GPU 2/5 ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 2
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 5
echo "=== port 8080 ==="
ss -ltn 2>/dev/null | grep 8080 | head -2
echo "=== server log tail (journal) ==="
journalctl -u llama-server -n 8 --no-pager 2>/dev/null | tail -8
echo SERVER_DONE
