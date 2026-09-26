#!/bin/bash
# bench-256k.sh - the L3 standard load (sm70/BENCH.md). The server must already be healthy.
# env: TAG (required), ENGINE (llama|vllm), PORT (8090 llama / 8000 vllm), REPS (3), GEN (128), NMAX (7),
#      PROMPT (/root/llm/test/sm70/prompt-256k.txt), OUTDIR (/root/llm/test/sm70/runs/<TAG>)
# The vLLM API key is read from the (read-only) unit file at run time and is never printed.
set -u
TAG=${TAG:?set TAG=<run tag>}
ENGINE=${ENGINE:-llama}
REPS=${REPS:-3}
GEN=${GEN:-128}
NMAX=${NMAX:-7}
PROMPT=${PROMPT:-/root/llm/test/sm70/prompt-256k.txt}
OUTDIR=${OUTDIR:-/root/llm/test/sm70/runs/$TAG}
T=/root/llm/test/sm70/tools
KEY=""
if [ "$ENGINE" = "vllm" ]; then
  PORT=${PORT:-8000}
  MODEL=Qwen3.8-27B-FP8
  KEY=$(grep -o -e '--api-key [^ ]*' /root/llm/systemd/vllm-1cat.service | cut -d' ' -f2)
else
  PORT=${PORT:-8090}
  MODEL=sm70-llama
fi
[ -f "$PROMPT" ] || { echo "BENCH_REFUSED no prompt file $PROMPT"; exit 2; }
mkdir -p "$OUTDIR"
echo "BENCH_BEGIN tag=$TAG engine=$ENGINE port=$PORT reps=$REPS gen=$GEN prompt=$PROMPT sha=$(sha256sum "$PROMPT" | cut -c1-16) date=$(date -Iseconds)" | tee "$OUTDIR/meta.txt"

nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -lms 1000 > "$OUTDIR/gpu.csv" &
SMI=$!
python3 $T/stress_client.py --base-url "http://127.0.0.1:$PORT" --api-key "$KEY" --model "$MODEL" \
  --prompt-file "$PROMPT" --gen "$GEN" --reps "$REPS" --tag "$TAG" --timeout 3600 > "$OUTDIR/stress.txt" 2>&1
kill $SMI 2> /dev/null

python3 $T/summarize.py "$OUTDIR" "$NMAX" | tee "$OUTDIR/summary.md"
echo "BENCH_DONE tag=$TAG outdir=$OUTDIR"
