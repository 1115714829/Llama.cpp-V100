#!/bin/bash
# t7: L3 acceptance at 256k on the production model, 3 cards in one NUMA node, WITH MTP (as production).
# Records eval t/s AND draft acceptance (tg is acceptance-dominated).
set -uo pipefail
KEY=$(head -1 /etc/llama-server/api-keys)
LOG=/tmp/l3.log
: > "$LOG"

echo "=== L3 start $(date -Is) ===" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$LOG"

# switch both L3 units from --spec-type none to MTP4 (matching the production config)
for u in llama-server-l3-pristine llama-server-l3-c4; do
  sed -i 's/--spec-type none/--spec-type draft-mtp --spec-draft-n-max 4/' "/etc/systemd/system/$u.service"
done
systemctl daemon-reload
grep -h "spec-type" /etc/systemd/system/llama-server-l3-*.service | tee -a "$LOG"

for u in llama-server-l3-pristine llama-server-l3-c4; do
  PORT=$(grep -oE 'port [0-9]+' "/etc/systemd/system/$u.service" | awk '{print $2}')
  echo "" | tee -a "$LOG"
  echo "########## $u (port $PORT) $(date -Is) ##########" | tee -a "$LOG"
  systemctl start "$u"
  n=0
  while [ "$n" -lt 420 ]; do
    journalctl -u "$u" --no-pager -o cat -n 8000 2>/dev/null | grep -q "model loaded" && break
    sleep 15; n=$((n+15))
  done
  echo "load wait ${n}s" | tee -a "$LOG"
  journalctl -u "$u" --no-pager -o cat -n 8000 2>/dev/null | grep -m1 "backend offload failed" | tee -a "$LOG" || true

  T0=$(date +%s)
  curl -s -m 2400 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d "{\"prompt\":$(python3 -c 'import json;print(json.dumps(open("/tmp/prompt256k.txt").read()))'),\"n_predict\":128,\"temperature\":0.7,\"top_p\":0.8,\"top_k\":20,\"repeat_penalty\":1.05,\"cache_prompt\":false}" \
    "http://127.0.0.1:$PORT/completion" > "/tmp/l3-$u-resp.json" 2>&1
  T1=$(date +%s)
  echo "request wall=$((T1-T0))s resp_bytes=$(wc -c < "/tmp/l3-$u-resp.json")" | tee -a "$LOG"
  journalctl -u "$u" --no-pager -o cat -n 8000 2>/dev/null \
    | grep -E "prompt eval time|eval time =|draft acceptance|total time =" | tail -5 | tee -a "$LOG"
  systemctl stop "$u"
  sleep 6
done
echo "=== L3 done $(date -Is) ===" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$LOG"
echo L3_DONE
