#!/bin/bash
set -uo pipefail
# Read the first API key from the api-key file.
KEY=$(head -1 /etc/llama-server/api-keys 2>/dev/null | tr -d '[:space:]')
if [ -z "$KEY" ]; then echo "NO_API_KEY"; exit 1; fi
echo "=== send test request (short prompt, n_predict=128) to port 8081 ==="
curl -s -m 120 http://127.0.0.1:8081/completion \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -d '{"prompt":"Tell me a short story about a cat.","n_predict":128,"stream":false}' \
  -o /tmp/resp.json -w "http=%{http_code} time_total=%{time_total}s TTFT=%{time_starttransfer}s\n" 2>&1
echo "=== response head ==="
head -c 200 /tmp/resp.json 2>/dev/null
echo ""
echo "=== timing from journal (prompt eval / eval / total) ==="
journalctl -u llama-server-test --since "2 min ago" --no-pager 2>/dev/null | grep -iE "prompt eval time|eval time|total time|draft acceptance|n_tokens|task " | tail -12
echo SPEED_DONE
