#!/bin/bash
# Diagnose why the L3 completion produced only 1 token.
set -uo pipefail
f=/tmp/l3-llama-server-l3-pristine-resp.json
echo "file: $f  bytes=$(wc -c < "$f" 2>/dev/null)"
echo "=== head ==="
head -c 300 "$f"
echo ""
echo "=== tail ==="
tail -c 700 "$f"
echo ""
echo "=== parsed fields ==="
python3 - <<'PY'
import json
p = '/tmp/l3-llama-server-l3-pristine-resp.json'
try:
    d = json.load(open(p))
except Exception as e:
    print("JSON parse failed:", e)
    raise SystemExit
print("keys:", sorted(d.keys()))
for k in ('content','stop_type','stopping_word','truncated','tokens_predicted','tokens_evaluated','timings','error'):
    v = d.get(k, '<absent>')
    if k == 'content' and isinstance(v, str):
        print(k, "len=", len(v), "head=", repr(v[:160]), "tail=", repr(v[-120:]))
    else:
        print(k, "=", v)
PY
echo INSPECT_DONE
