#!/bin/bash
set -uo pipefail
KEY=$(head -1 /etc/llama-server/api-keys 2>/dev/null | tr -d '[:space:]')
# base paragraph = 20 tokens (confirmed). N repeats to reach ~262000 tokens.
N=13100
python3 - "$N" > /tmp/prompt256k.txt <<'PY'
import sys
N=int(sys.argv[1])
base="The quick brown fox jumps over the lazy dog near the river bank on a calm sunny morning. "
sys.stdout.write(base*N)
PY
# token count of the full prompt (the /tokenize endpoint returns a list of ids; count it)
FTOK=$(python3 - "$KEY" <<'PY'
import sys,json,urllib.request
key=sys.argv[1]
content=open("/tmp/prompt256k.txt").read()
data=json.dumps({"content":content}).encode()
req=urllib.request.Request("http://127.0.0.1:8081/tokenize",data=data,headers={"Content-Type":"application/json","Authorization":"Bearer "+key})
r=urllib.request.urlopen(req,timeout=120)
d=json.load(r)
print(len(d) if isinstance(d,list) else d.get("tokens","?"))
PY
)
echo "full prompt tokens: ${FTOK:-?}"
echo "prompt file: /tmp/prompt256k.txt ($(wc -c < /tmp/prompt256k.txt) bytes)"
echo PREP_DONE
