#!/bin/bash
set -uo pipefail
KEY=$(head -1 /etc/llama-server/api-keys 2>/dev/null | tr -d '[:space:]')
# Build the JSON request (prompt + n_predict=128).
python3 - > /tmp/req256k.json <<'PY'
import json
prompt=open("/tmp/prompt256k.txt").read()
print(json.dumps({"prompt":prompt,"n_predict":128,"stream":False}))
PY
echo "request json: $(wc -c < /tmp/req256k.json) bytes"
echo "=== launch 256k TTFT test (orig, port 8081) in background ==="
nohup bash -c "curl -s -m 3600 http://127.0.0.1:8081/completion -H 'Content-Type: application/json' -H 'Authorization: Bearer $KEY' -d @/tmp/req256k.json -o /tmp/resp256k.json -w 'http=%{http_code} total=%{time_total}s TTFT=%{time_starttransfer}s\n'" > /tmp/bench-256k-orig.log 2>&1 &
echo "launched 256k orig test PID=$! log=/tmp/bench-256k-orig.log"
echo "monitor: journalctl -u llama-server-test -f"
echo LAUNCH_256K_DONE
