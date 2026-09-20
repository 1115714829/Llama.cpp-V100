#!/bin/bash
# Speculative-decoding sweep: which spec type / n_max gives the best acceptance on REAL prose?
# Controllable design: fixed seed, same chat-templated prompt, short ctx -> cheap, comparable.
set -uo pipefail
BIN=/root/libdir-final/llama-server
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
KEY=$(head -1 /etc/llama-server/api-keys)
PORT=8095
LOG=/tmp/spec-sweep.log
: > "$LOG"

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/root/libdir-final

echo "=== spec sweep start $(date -Is) ===" | tee -a "$LOG"
"$BIN" --version 2>&1 | grep -i version | tee -a "$LOG"

# tag|spec args
VARIANTS="
none|--spec-type none
mtp4|--spec-type draft-mtp --spec-draft-n-max 4
mtp8|--spec-type draft-mtp --spec-draft-n-max 8
ngram_simple|--spec-type ngram-simple
ngram_map_k|--spec-type ngram-map-k
ngram_mod|--spec-type ngram-mod
"

echo "$VARIANTS" | while IFS='|' read -r tag spec; do
  [ -z "$tag" ] && continue
  echo "" | tee -a "$LOG"
  echo "########## $tag : $spec ##########" | tee -a "$LOG"
  # shellcheck disable=SC2086
  nohup "$BIN" --model "$M" --ctx-size 4096 -ngl 999 \
    --split-mode tensor --tensor-split 1 \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    $spec \
    --temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05 \
    --host 127.0.0.1 --port "$PORT" --api-key-file /etc/llama-server/api-keys \
    > "/tmp/spec-$tag.log" 2>&1 &
  pid=$!
  n=0; ok=0
  while [ "$n" -lt 240 ]; do
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
    sleep 5; n=$((n+5))
  done
  echo "health ok=$ok after ${n}s" | tee -a "$LOG"

  if [ "$ok" = "1" ]; then
    curl -s -m 600 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
      -d '{"messages":[{"role":"user","content":"Explain how a GPU memory hierarchy works, in a few paragraphs of prose."}],"n_predict":256,"temperature":0.7,"top_p":0.8,"top_k":20,"repeat_penalty":1.05,"seed":42,"cache_prompt":false}' \
      "http://127.0.0.1:$PORT/v1/chat/completions" > "/tmp/spec-$tag.json" 2>&1
    echo "resp_bytes=$(wc -c < "/tmp/spec-$tag.json")" | tee -a "$LOG"
    grep -E "prompt eval time|eval time =|draft acceptance|n_gen =" "/tmp/spec-$tag.log" | tail -4 | tee -a "$LOG"
    grep -ciE "not supported|error" "/tmp/spec-$tag.log" | sed 's/^/warn_lines=/' | tee -a "$LOG"
  fi
  kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null; sleep 2
done
echo "" | tee -a "$LOG"
echo "=== spec sweep done $(date -Is) ===" | tee -a "$LOG"
echo SPEC_SWEEP_DONE
