#!/bin/bash
set -uo pipefail
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  sleep 6
  P=$(ss -ltn 2>/dev/null | grep -c 8081)
  if [ "$P" -gt 0 ]; then echo "PORT_8081_UP after $((i*6))s"; break; fi
done
echo "=== service ==="
systemctl is-active llama-server-test
echo "=== GPU 2/5 ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 5
echo "=== journal (model load / errors, last 25) ==="
journalctl -u llama-server-test -n 25 --no-pager 2>/dev/null | grep -iE "load|model|ctx|error|warn|aborted|ready|listening|slot|tensor" | tail -25
echo TEST_CHECK_DONE
