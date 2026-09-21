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
echo "=== journal (MTP/spec/model load, last 30) ==="
journalctl -u llama-server-test -n 30 --no-pager 2>/dev/null | grep -iE "spec|mtp|draft|load_model|model loaded|listening|nextn|error|aborted" | tail -15
echo MTP_CHECK_DONE
