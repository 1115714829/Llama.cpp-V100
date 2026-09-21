#!/bin/bash
# L3 decode, MTP OFF (clean kernel comparison) + fixed seed, production model, both variants.
# With MTP off, tg no longer depends on the draft acceptance rate, so a difference is attributable to the kernel.
set -uo pipefail
KEY=$(head -1 /etc/llama-server/api-keys)
LOG=/tmp/l3dec3.log
: > "$LOG"

echo "=== L3 decode3 (MTP off, seed 42) start $(date -Is) ===" | tee -a "$LOG"
# switch both L3 units to MTP off
for u in llama-server-l3-pristine llama-server-l3-c4; do
  sed -i 's/--spec-type draft-mtp --spec-draft-n-max 4/--spec-type none/' "/etc/systemd/system/$u.service"
done
systemctl daemon-reload
grep -h "spec-type" /etc/systemd/system/llama-server-l3-*.service | tee -a "$LOG"

for u in llama-server-l3-pristine llama-server-l3-c4; do
  PORT=$(grep -oE 'port [0-9]+' "/etc/systemd/system/$u.service" | awk '{print $2}')
  echo "" | tee -a "$LOG"
  echo "########## $u port $PORT ##########" | tee -a "$LOG"
  systemctl start "$u"
  n=0; ok=0
  while [ "$n" -lt 300 ]; do
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
    sleep 5; n=$((n+5))
  done
  echo "health ok=$ok after ${n}s" | tee -a "$LOG"
  if [ "$ok" = "1" ]; then
    # three repeats to see the spread (MTP off should be far more repeatable than with MTP)
    for rep in 1 2 3; do
      curl -s -m 900 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"Write a short paragraph about the sea."}],"n_predict":128,"temperature":0.7,"top_p":0.8,"top_k":20,"repeat_penalty":1.05,"seed":42,"cache_prompt":false}' \
        "http://127.0.0.1:$PORT/v1/chat/completions" > "/tmp/l3dec3-$u-$rep.json" 2>&1
      echo "rep$rep bytes=$(wc -c < "/tmp/l3dec3-$u-$rep.json")" | tee -a "$LOG"
      journalctl -u "$u" --no-pager -o cat -n 8000 2>/dev/null \
        | grep -E "^\s*[0-9.]+ I slot print_timing.*eval time =" | tail -1 | tee -a "$LOG"
    done
  fi
  systemctl stop "$u"
  sleep 6
done
echo "=== done $(date -Is) ===" | tee -a "$LOG"
echo L3DEC3_DONE
