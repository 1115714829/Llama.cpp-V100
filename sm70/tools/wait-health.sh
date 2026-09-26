#!/bin/bash
# wait-health.sh <port> [timeout_s] - 5 s loop until http://127.0.0.1:<port>/health answers 200.
PORT=${1:?usage: wait-health.sh <port> [timeout_s]}
TMO=${2:-1200}
w=0
while :; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 "http://127.0.0.1:$PORT/health")
  if [ "$code" = "200" ]; then echo "HEALTH_OK port=$PORT after=${w}s"; exit 0; fi
  sleep 5
  w=$((w + 5))
  if [ "$w" -ge "$TMO" ]; then echo "HEALTH_TIMEOUT port=$PORT after=${w}s last_code=$code"; exit 1; fi
done
