#!/bin/bash
set -uo pipefail
echo "=== curl port 8080 (health) ==="
curl -s -m 8 http://127.0.0.1:8080/health 2>&1 | head -3
echo ""
echo "=== curl /v1/models ==="
curl -s -m 8 http://127.0.0.1:8080/v1/models 2>&1 | head -3
echo ""
echo "=== service + GPU ==="
systemctl is-active llama-server
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 5
echo "=== last 5 log lines ==="
journalctl -u llama-server -n 5 --no-pager 2>/dev/null | tail -5
echo RESP_DONE
