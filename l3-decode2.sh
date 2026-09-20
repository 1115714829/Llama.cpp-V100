#!/bin/bash
# L3 decode (corrected): chat-templated 128-token generation on the production model, both variants.
# Fix: wait on /health instead of grepping a log string (the old wait never matched -> wasted 420 s/run).
set -uo pipefail
KEY=$(head -1 /etc/llama-server/api-keys)
LOG=/tmp/l3dec2.log
: > "$LOG"

echo "=== L3 decode2 start $(date -Is) ===" | tee -a "$LOG"
for u in llama-server-l3-pristine llama-server-l3-c4; do
  PORT=$(grep -oE 'port [0-9]+' "/etc/systemd/system/$u.service" | awk '{print $2}')
  echo "" | tee -a "$LOG"
  echo "########## $u port $PORT ##########" | tee -a "$LOG"
  systemctl start "$u"
  n=0
  ok=0
  while [ "$n" -lt 300 ]; do
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
    sleep 5; n=$((n+5))
  done
  echo "health ok=$ok after ${n}s" | tee -a "$LOG"
  if [ "$ok" = "1" ]; then
    curl -s -m 900 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
      -d '{"messages":[{"role":"user","content":"Write a short paragraph about the sea."}],"n_predict":128,"temperature":0.7,"top_p":0.8,"top_k":20,"repeat_penalty":1.05,"cache_prompt":false}' \
      "http://127.0.0.1:$PORT/v1/chat/completions" > "/tmp/l3dec2-$u.json" 2>&1
    echo "resp_bytes=$(wc -c < "/tmp/l3dec2-$u.json")" | tee -a "$LOG"
    journalctl -u "$u" --no-pager -o cat -n 8000 2>/dev/null \
      | grep -E "prompt eval time|eval time =|draft acceptance" | tail -3 | tee -a "$LOG"
  fi
  systemctl stop "$u"
  sleep 6
done
echo "=== done $(date -Is) ===" | tee -a "$LOG"
echo L3DEC2_DONE
