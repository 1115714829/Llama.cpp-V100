#!/bin/bash
# p60-ab-harness2.sh -- same fixed workload as p60-ab-harness.sh, plus:
#   LDEXTRA=<dir>  appended to LD_LIBRARY_PATH (e.g. the ac922 NCCL bundle dir)
#   prints the [RT] target decode+sync lines and the NCCL/AllReduce startup lines,
#   so a single run carries its own evidence for the all-reduce implementation used.
#
# Usage:
#   CARDS="0,1,2" SPLIT=tensor TAG=nccl L=/root/libdir-nccl LDEXTRA=/path/to/nccl/lib P2P=1 bash p60-ab-harness2.sh
set -uo pipefail

CARDS=${CARDS:-0,1,2}
SPLIT=${SPLIT:-tensor}
TAG=${TAG:-base}
NPRED=${NPRED:-512}
PORT=${PORT:-8120}
L=${L:-/root/libdir-rt}
LDEXTRA=${LDEXTRA:-}
M=${M:-/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf}
D=${D:-/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf}
SPEC=${SPEC:-"--model-draft $D --spec-type draft-dflash --spec-draft-n-max 7"}
KEY=$(head -1 /etc/llama-server/api-keys)
OUT=/tmp/p60-$TAG.log
SLOG=/tmp/p60-$TAG-server.log
: > "$OUT"

IFS=',' read -ra DEVS <<< "$CARDS"
NDEV=${#DEVS[@]}
TS=""
for i in $(seq 1 "$NDEV"); do TS="${TS}1,"; done
TS="${TS%,}"

{
echo "=== p60-ab-harness2 =============================================="
echo "TAG=$TAG"
echo "config: cards=$CARDS (n=$NDEV)  split=$SPLIT  tensor-split=$TS"
echo "        spec='$SPEC'  n_predict=$NPRED  ctx=8192  libdir=$L"
echo "        LDPATH_EXTRA='$LDEXTRA'"
echo "libs:   $(md5sum "$L/libllama.so.0.4.1" 2>/dev/null | awk '{print $1}') libllama.so.0.4.1"
echo "        $(md5sum "$L/libggml-cuda.so.0.24.0" 2>/dev/null | awk '{print $1}') libggml-cuda.so.0.24.0"
echo "        $(md5sum "$L/libllama-common.so.0.4.1" 2>/dev/null | awk '{print $1}') libllama-common.so.0.4.1"
echo "oracle: target $(stat -c %s "$M" 2>/dev/null) B  draft $(stat -c %s "$D" 2>/dev/null) B"
} | tee -a "$OUT"

if [ "${NODROP:-0}" = "1" ]; then
  echo "NODROP=1: skipping drop_caches (diagnostic run, model stays page-cache hot)" | tee -a "$OUT"
else
  sync; echo 3 > /proc/sys/vm/drop_caches; sleep 3
fi
echo "gpu mem before launch:" | tee -a "$OUT"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$OUT"

P1="Explain, step by step, how to compute the sum of the first n integers and prove the formula by induction."
P2="A train leaves Station A at 60 km/h. Another leaves Station B, 300 km away, at 40 km/h toward A. They start at the same time. Work through this step by step and explain when and where they meet."
P3="Solve step by step: if 3x + 7 = 22, what is x? Then explain in detail how you checked your answer."
KW='"chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}'

if [ "${P2P:-0}" = "1" ]; then
  export GGML_CUDA_P2P=1
  echo "extra env: GGML_CUDA_P2P=1" | tee -a "$OUT"
fi
if [ -n "${ARENV:-}" ]; then
  for kv in ${ARENV//,/ }; do export "GGML_CUDA_AR_$kv"; done
  echo "extra env: GGML_CUDA_AR_* = $ARENV" | tee -a "$OUT"
fi
if [ -n "${ARENV2:-}" ]; then
  for kv in ${ARENV2//,/ }; do export "$kv"; done
  echo "extra env: $ARENV2" | tee -a "$OUT"
fi

DRAFT_ARGS="--spec-draft-ngl ${NGLD:-999}"
if [ -n "${DEVD:-}" ]; then
  DRAFT_ARGS="$DRAFT_ARGS --spec-draft-device $DEVD"
  echo "draft placement: device=$DEVD ngl=${NGLD:-999}" | tee -a "$OUT"
fi

if [ "${NOGRAPH:-0}" = "1" ]; then
  export GGML_CUDA_DISABLE_GRAPHS=1
  echo "extra env: GGML_CUDA_DISABLE_GRAPHS=1" | tee -a "$OUT"
fi

env CUDA_VISIBLE_DEVICES="$CARDS" LD_LIBRARY_PATH="$L${LDEXTRA:+:$LDEXTRA}" LLAMA_ROUND_TIMING=1 LLAMA_SPEC_TIMING=1 \
  nohup "$L/llama-server" --model "$M" $SPEC $DRAFT_ARGS --split-mode "$SPLIT" --tensor-split "$TS" \
    --ctx-size 8192 -ngl 999 \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
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
  echo "--- launch failure evidence ---" | tee -a "$OUT"
  grep -aE "GGML_ASSERT|error|failed to|abort|not supported|no memory|NCCL" "$SLOG" | tail -12 | tee -a "$OUT"
  kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null
  echo "P60_DONE TAG=$TAG STATUS=LAUNCH_FAILED"
  exit 1
fi

# all-reduce implementation actually in use (one-time startup line from ggml-cuda.cu)
echo "--- allreduce init lines from server log ---" | tee -a "$OUT"
grep -aiE "NCCL|AllReduce|allreduce" "$SLOG" | head -6 | tee -a "$OUT"

VALS=""
i=0
for p in "$P1" "$P2" "$P3"; do
  i=$((i+1))
  body="{\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"n_predict\":$NPRED,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"presence_penalty\":0.0,\"repetition_penalty\":1.0,\"seed\":42,\"cache_prompt\":false,$KW}"
  curl -s -m 1800 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d "$body" "http://127.0.0.1:$PORT/v1/chat/completions" > "/tmp/p60-$TAG-p$i.json" 2>&1
  tps=$(grep -a -oE "[0-9]+\.[0-9]+ tokens per second" "$SLOG" | tail -1 | awk '{print $1}')
  al=$(grep -a -oE "mean len = +[0-9.]+" "$SLOG" | tail -1 | awk '{print $4}')
  [ -z "$tps" ] && tps=0
  echo "prompt$i: tg=$tps tok/s  AL=$al" | tee -a "$OUT"
  VALS="$VALS $tps"
done
MED=$(echo $VALS | tr ' ' '\n' | grep -v '^$' | sort -g | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')

gb="{\"messages\":[{\"role\":\"user\",\"content\":\"$P3\"}],\"n_predict\":128,\"temperature\":0.0,\"top_p\":1.0,\"top_k\":0,\"min_p\":0.0,\"presence_penalty\":0.0,\"repetition_penalty\":1.0,\"seed\":42,\"cache_prompt\":false}"
curl -s -m 900 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "$gb" "http://127.0.0.1:$PORT/v1/chat/completions" > "/tmp/p60-$TAG-greedy.json" 2>&1
python3 - "$TAG" <<'PY' | tee -a "$OUT"
import json,sys,hashlib
tag=sys.argv[1]
try:
    d=json.load(open(f"/tmp/p60-{tag}-greedy.json"))
    c=d["choices"][0]["message"]["content"] or ""
    rc=d["choices"][0]["message"].get("reasoning_content") or ""
    full=rc+"\x00"+c
    print(f"greedy: content_len={len(c)} reasoning_len={len(rc)} sha256={hashlib.sha256(full.encode()).hexdigest()}")
    open(f"/tmp/p60-{tag}-greedy.txt","w").write(full)
except Exception as e:
    print(f"greedy: PARSE_FAIL {e}")
PY

echo "" | tee -a "$OUT"
echo "MEDIAN_TG=$MED" | tee -a "$OUT"
echo "--- round breakdown (target host side) ---" | tee -a "$OUT"
grep -a -F "[RT] perf:" "$SLOG" | tail -1 | tee -a "$OUT"
echo "--- target decode+sync per round ---" | tee -a "$OUT"
grep -a -F "[RT] target decode+sync" "$SLOG" | tail -3 | tee -a "$OUT"
echo "--- draft phases ---" | tee -a "$OUT"
grep -a "spec timing" "$SLOG" | tail -1 | tee -a "$OUT"

kill "$pid" 2>/dev/null; sleep 4; kill -9 "$pid" 2>/dev/null
echo "P60_DONE TAG=$TAG STATUS=OK"
