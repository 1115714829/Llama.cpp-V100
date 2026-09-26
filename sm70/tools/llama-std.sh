#!/bin/bash
# llama-std.sh - the standard llama-server launch (parity table in sm70/BENCH.md). Runs in the foreground: start it with rjob.sh.
# env: LIBS (required), TAG (required), CARDS (0,1,3,4), MODEL (Q8_0), CTX (262144), SPEC (1), DRAFT (F16 GGUF), NMAX (7), PORT (8095)
# A/B switches: the caller exports GGML_* / LLAMA_* variables; they are inherited unchanged and echoed into the log.
set -u
LIBS=${LIBS:?set LIBS=<lib dir>}
TAG=${TAG:?set TAG=<run tag>}
CARDS=${CARDS:-0,1,3,4}
MODEL=${MODEL:-/mnt/3.84t/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf}
DRAFT=${DRAFT:-/mnt/3.84t/llm-models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-F16.gguf}
CTX=${CTX:-262144}
SPEC=${SPEC:-1}
NMAX=${NMAX:-7}
PORT=${PORT:-8095}

if pgrep -f 'llama-serve[r] --model' > /dev/null; then echo "LAUNCH_REFUSED a llama-server is already running"; exit 2; fi
# 8090 = llmscope, 3000 = new-api, 8000 = vllm-1cat, 8080 = production llama-server: never share a port
if (exec 3<> /dev/tcp/127.0.0.1/$PORT) 2> /dev/null; then echo "LAUNCH_REFUSED port $PORT is already in use"; exit 2; fi
case "$MODEL" in /mnt/3.84t/*) ;; *) echo "LAUNCH_REFUSED model must be under /mnt/3.84t"; exit 2 ;; esac
[ -x "$LIBS/llama-server" ] || { echo "LAUNCH_REFUSED no $LIBS/llama-server"; exit 2; }

N=$(echo "$CARDS" | tr ',' '\n' | grep -c .)
if [ "$N" = "1" ]; then
  SPLIT_ARGS=(--split-mode none)
else
  SPLIT_ARGS=(--split-mode tensor --tensor-split "$(yes 1 | head -n "$N" | paste -sd, -)")
fi
SPEC_ARGS=()
if [ "$SPEC" = "1" ]; then
  SPEC_ARGS=(--model-draft "$DRAFT" --spec-type draft-dflash --spec-draft-n-max "$NMAX")
fi

export CUDA_VISIBLE_DEVICES=$CARDS
export LD_LIBRARY_PATH=$LIBS:/usr/local/cuda/lib64
echo "STD_LAUNCH tag=$TAG libs=$LIBS cards=$CARDS model=$MODEL ctx=$CTX spec=$SPEC nmax=$NMAX port=$PORT date=$(date -Iseconds)"
echo "STD_ENV $(env | grep -E '^(GGML_|LLAMA_)' | sort | tr '\n' ' ')"
sed 's/^/STD_MANIFEST /' "$LIBS/BUILD_MANIFEST.txt" 2> /dev/null
exec "$LIBS/llama-server" --model "$MODEL" --alias sm70-llama \
  --ctx-size "$CTX" --parallel 1 --n-gpu-layers 999 "${SPLIT_ARGS[@]}" \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --batch-size 2048 --ubatch-size 2048 \
  "${SPEC_ARGS[@]}" \
  --reasoning on --reasoning-effort xhigh --reasoning-preserve \
  --host 127.0.0.1 --port "$PORT" --metrics
