#!/bin/bash
# Corrected L3 DECODE measurement: use /v1/chat/completions so the server applies the chat template
# (a raw /completion of repeated plain text makes the instruct model emit EOS immediately).
# Decode on this hybrid model is ~context-independent (48/64 layers are GDN), so a modest context
# gives the same tg as 256k at a fraction of the cost. MTP on, and we report acceptance.
set -uo pipefail
KEY=$(head -1 /etc/llama-server/api-keys)
LOG=/tmp/l3-dec.log
: > "$LOG"

echo "=== L3 decode (chat template) start $(date -Is) ===" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$LOG"

for u in llama-server-l3-pristine llama-server-l3-c4; do
  PORT=$(grep -oE 'port [0-9]+' "/etc/systemd/system/$u.service" | awk '{print $2}')
  echo "" | tee -a "$LOG"
  echo "########## $u (port $PORT) ##########" | tee -a "$LOG"
  systemctl start "$u"
  n=0
  while [ "$n" -lt 420 ]; do
    journalctl -u "$u" --no-pager -o cat -n 8000 2>/dev/null | grep -q "model loaded" && break
    sleep 15; n=$((n+15))
  done
  echo "load wait ${n}s" | tee -a "$LOG"

  # build a chat-templated request with a ~250-token user message (cheap prefill)
  python3 - "$PORT" "$KEY" <<'PY' > "/tmp/l3dec-$u.json" 2>"/tmp/l3dec-$u.err"
import json, sys, urllib.request
port, key = sys.argv[1], sys.argv[2]
body = {
    "messages": [
        {"role": "user", "content": "Write a short paragraph about the sea."},
    ],
    "n_predict": 128,
    "temperature": 0.7, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05,
    "cache_prompt": False,
}
req = urllib.request.Request(
    "http://127.0.0.1:%s/v1/chat/completions" % port,
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
)
with urllib.request.urlopen(req, timeout=1800) as r:
    out = json.load(r)
print(json.dumps(out))
PY
  echo "resp bytes=$(wc -c < "/tmp/l3dec-$u.json") err=$(head -c 200 "/tmp/l3dec-$u.err")" | tee -a "$LOG"
  python3 -c "
import json,sys
d=json.load(open('/tmp/l3dec-'+sys.argv[1]+'.json'))
u=d.get('usage',{})
print('usage:', u)
ch=d.get('choices',[{}])[0]
print('finish_reason:', ch.get('finish_reason'))
print('content_len:', len(ch.get('message',{}).get('content') or ''))
" "$u" | tee -a "$LOG"

  journalctl -u "$u" --no-pager -o cat -n 8000 2>/dev/null \
    | grep -E "prompt eval time|eval time =|draft acceptance|total time =" | tail -5 | tee -a "$LOG"
  systemctl stop "$u"
  sleep 6
done
echo "=== done $(date -Is) ===" | tee -a "$LOG"
echo L3DEC_DONE
