#!/bin/bash
# MTP experiment: quantify the CPU-sampler fallback cost and test --split-mode layer.
# Runs on GPU 3,4 (same NUMA node). Uses the pristine b11053 libdir.
set -uo pipefail

BIN=/root/libdir-pristine/llama-server
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
CTX=8192
GEN=128
PORT=8093
LOG=/tmp/mtp-exp.log
: > "$LOG"

export CUDA_VISIBLE_DEVICES=3,4
export LD_LIBRARY_PATH=/root/libdir-pristine

# fixed ~2k-token prompt (reuse the head of the 256k prompt file)
head -c 8000 /tmp/prompt256k.txt > /tmp/mtp-prompt.txt

echo "=== MTP experiment start $(date -Is) ===" | tee -a "$LOG"
echo "bin=$BIN" | tee -a "$LOG"
ls -la "$BIN" | tee -a "$LOG"
"$BIN" --version 2>&1 | grep -iE "version" | tee -a "$LOG"
echo "--- gpu ---" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$LOG"
echo "" | tee -a "$LOG"

run_variant() {
  local tag="$1"; shift
  local sm="$1"; shift
  local spec="$1"; shift

  echo "" | tee -a "$LOG"
  echo "############ variant=$tag  split-mode=$sm  spec=$spec ############" | tee -a "$LOG"

  nohup "$BIN" --model "$M" --ctx-size "$CTX" -ngl 999 \
    --split-mode "$sm" --tensor-split 1,1 \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    $spec \
    --temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05 \
    --host 127.0.0.1 --port "$PORT" --api-key-file /etc/llama-server/api-keys \
    > "/tmp/mtp-$tag.log" 2>&1 &
  local pid=$!
  echo "server pid=$pid log=/tmp/mtp-$tag.log" | tee -a "$LOG"

  # wait for load
  local n=0
  while [ "$n" -lt 240 ]; do
    if grep -q "model loaded" "/tmp/mtp-$tag.log" 2>/dev/null; then break; fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "SERVER DIED" | tee -a "$LOG"; break; fi
    sleep 5; n=$((n + 5))
  done
  echo "load wait: ${n}s  model_loaded=$(grep -c 'model loaded' "/tmp/mtp-$tag.log")" | tee -a "$LOG"

  # note whether the CPU-sampler fallback happened
  echo "offload_failed_lines=$(grep -c 'backend offload failed' "/tmp/mtp-$tag.log")" | tee -a "$LOG"
  grep -m1 "not supported with SPLIT_MODE_TENSOR" "/tmp/mtp-$tag.log" | sed 's/^/  /' | tee -a "$LOG" || true

  # request
  KEY=$(head -1 /etc/llama-server/api-keys)
  curl -s -m 900 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d "{\"prompt\":$(python3 -c 'import json,sys;print(json.dumps(open("/tmp/mtp-prompt.txt").read()))'),\"n_predict\":$GEN,\"temperature\":0.7,\"top_p\":0.8,\"top_k\":20,\"repeat_penalty\":1.05,\"cache_prompt\":false}" \
    "http://127.0.0.1:$PORT/completion" > "/tmp/mtp-$tag-resp.json" 2>&1
  echo "resp_bytes=$(wc -c < "/tmp/mtp-$tag-resp.json")" | tee -a "$LOG"

  echo "--- server timing lines ---" | tee -a "$LOG"
  grep -E "prompt eval time|eval time =|draft acceptance|n_gen =|total time =" "/tmp/mtp-$tag.log" | tail -5 | tee -a "$LOG"

  kill "$pid" 2>/dev/null
  sleep 5
  kill -9 "$pid" 2>/dev/null
  sleep 3
}

run_variant A tensor "--spec-type none"
run_variant B tensor "--spec-type draft-mtp --spec-draft-n-max 4"
run_variant C layer  "--spec-type none"
run_variant D layer  "--spec-type draft-mtp --spec-draft-n-max 4"

echo "" | tee -a "$LOG"
echo "=== MTP experiment done $(date -Is) ===" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$LOG"
echo MTP_EXP_DONE
