#!/bin/bash
# Long-context speculative yardstick (round 40). Same discipline as p60-ab-harness.sh:
# one long identical prefix + 3 fixed short instructions, fixed seed, reports tg / AL / dispersion.
# env: TAG KV CTX NPRED CARDS L PORT
set -uo pipefail
TAG=${TAG:-lc}; KV=${KV:-q8_0}; CTX=${CTX:-32768}; NPRED=${NPRED:-192}
CARDS=${CARDS:-0,1,2}; L=${L:-/root/libdir-instr}; PORT=${PORT:-8140}; FA=${FA:-on}; EXTRA=${EXTRA:-}
NSPLIT=$(echo "$CARDS" | awk -F, '{s="1"; for(i=2;i<=NF;i++) s=s",1"; print s}')
OUT=/tmp/lc-spec-$TAG.txt; SLOG=/tmp/lc-spec-$TAG-server.log
KEY=$(head -1 /etc/llama-server/api-keys)
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
SPEC="--model-draft $D --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-ngl 999"
: > "$OUT"
echo "TAG=$TAG KV=$KV CTX=$CTX NPRED=$NPRED CARDS=$CARDS SPLIT=$NSPLIT FA=$FA EXTRA=$EXTRA L=$L" | tee -a "$OUT"
python3 - "$CTX" <<'PY'
import sys
ctx = int(sys.argv[1])
src = open('/root/lc-passage.txt', 'r', errors='replace').read()
need = int(ctx * 2.8)
out = []
n = 0
while n < need:
    out.append(src)
    n += len(src)
open('/root/lc-prefix.txt', 'w').write(''.join(out)[:need])
print('prefix_chars', need, file=sys.stderr)
PY
sync; echo 3 > /proc/sys/vm/drop_caches
env CUDA_VISIBLE_DEVICES="$CARDS" LD_LIBRARY_PATH="$L" LLAMA_ROUND_TIMING=1 LLAMA_SPEC_TIMING=1 GGML_CUDA_AR_TIMING=1 \
  nohup "$L/llama-server" --model "$M" $SPEC --split-mode tensor --tensor-split "$NSPLIT" \
    --ctx-size "$CTX" -ngl 999 --flash-attn "$FA" --cache-type-k "$KV" --cache-type-v "$KV" --parallel 1 $EXTRA \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 \
    --host 127.0.0.1 --port "$PORT" --api-key-file /etc/llama-server/api-keys > "$SLOG" 2>&1 &
pid=$!
n=0; ok=0
while [ "$n" -lt 900 ]; do
  if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
  kill -0 "$pid" 2>/dev/null || break
  sleep 15; n=$((n+15))
done
echo "health ok=$ok after ${n}s" | tee -a "$OUT"
if [ "$ok" != "1" ]; then
  grep -aE "GGML_ASSERT|error|failed to|abort|not supported|no memory" "$SLOG" | tail -6 | tee -a "$OUT"
  kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null
  echo "LC_SPEC_DONE TAG=$TAG STATUS=LAUNCH_FAILED"; exit 1
fi
VALS=""
i=0
for q in 'Summarize the most important functions in this code and what they are for.' 'List the main data structures and the invariants that connect them.' 'Point out any place where the code could fail at runtime and explain why.'; do
  i=$((i+1))
  python3 - "$q" "$NPRED" <<'PY'
import json, sys
q = sys.argv[1]; n = int(sys.argv[2])
p = open('/root/lc-prefix.txt').read() + chr(10) + chr(10) + q
body = {'messages': [{'role': 'user', 'content': p}], 'n_predict': n, 'temperature': 1.0, 'top_p': 0.95,
        'top_k': 20, 'min_p': 0.0, 'presence_penalty': 0.0, 'repetition_penalty': 1.0, 'seed': 42, 'cache_prompt': False}
json.dump(body, open('/tmp/lc-body.json', 'w'))
PY
  t0=$(date +%s)
  curl -s -m 3600 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d @/tmp/lc-body.json "http://127.0.0.1:$PORT/v1/chat/completions" > "/tmp/lc-$TAG-p$i.json" 2>&1
  t1=$(date +%s)
  tps=$(grep -a -oE "[0-9]+\.[0-9]+ tokens per second" "$SLOG" | tail -1 | awk '{print $1}')
  al=$(grep -a -oE "mean len = +[0-9.]+" "$SLOG" | tail -1 | awk '{print $4}')
  [ -z "$tps" ] && tps=0
  echo "prompt$i: tg=$tps tok/s  AL=$al  wall=$((t1-t0))s" | tee -a "$OUT"
  VALS="$VALS $tps"
done
MED=$(echo $VALS | tr ' ' '\n' | grep -v '^$' | sort -g | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')
echo "MEDIAN_TG=$MED" | tee -a "$OUT"
python3 - "$TAG" <<'PY' | tee -a "$OUT"
import json, sys, hashlib
tag = sys.argv[1]
for i in (1, 2, 3):
    try:
        d = json.load(open('/tmp/lc-%s-p%d.json' % (tag, i)))
        t = d.get('timings', {})
        print('p%d timings: prompt_n=%s prompt_ms=%s predicted_n=%s predicted_ms=%s' % (i, t.get('prompt_n'), t.get('prompt_ms'), t.get('predicted_n'), t.get('predicted_ms')))
    except Exception as e:
        print('p%d PARSE_FAIL %s' % (i, e))
PY
echo '--- round breakdown (target) ---' | tee -a "$OUT"
grep -a -F '[RT] perf:' "$SLOG" | tail -1 | tee -a "$OUT"
echo '--- allreduce ---' | tee -a "$OUT"
grep -a -F '[AR]' "$SLOG" | tail -1 | tee -a "$OUT"
echo '--- spec phases ---' | tee -a "$OUT"
grep -a 'spec timing' "$SLOG" | tail -1 | tee -a "$OUT"
kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null
echo "LC_SPEC_DONE TAG=$TAG"