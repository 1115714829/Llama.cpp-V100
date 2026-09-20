#!/bin/bash
# Robustness check: draft-dflash vs draft-mtp across several DIFFERENT prose prompts, fixed seed.
# One server start per variant; several requests each (cheaper than restarting per prompt).
set -uo pipefail
BIN=/root/libdir-final/llama-server
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
KEY=$(head -1 /etc/llama-server/api-keys)
PORT=8098
LOG=/tmp/prompt-robust.log
: > "$LOG"

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/root/libdir-final

ask() { # $1=label  $2=prompt
  curl -s -m 600 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}],\"n_predict\":256,\"temperature\":0.7,\"top_p\":0.8,\"top_k\":20,\"repeat_penalty\":1.05,\"seed\":42,\"cache_prompt\":false}" \
    "http://127.0.0.1:$PORT/v1/chat/completions" > /dev/null 2>&1
  echo -n "[$1] " | tee -a "$LOG"
  grep -E "eval time =" "/tmp/robust-$3.log" | tail -1 | sed 's/.*| *//' | tee -a "$LOG"
  grep -E "draft acceptance" "/tmp/robust-$3.log" | tail -1 | sed 's/.*| *//' | tee -a "$LOG"
}

P1="Explain how a GPU memory hierarchy works, in a few paragraphs of prose."
P2="Write a short story about a lighthouse keeper who finds a message in a bottle."
P3="Summarize the trade-offs between data parallelism and tensor parallelism for LLM inference."

for tag in dflash mtp4; do
  if [ "$tag" = "dflash" ]; then
    SPEC="--spec-type draft-dflash"; DRAFTARG="--model-draft $D"
  else
    SPEC="--spec-type draft-mtp --spec-draft-n-max 4"; DRAFTARG=""
  fi
  echo "" | tee -a "$LOG"
  echo "########## $tag ($SPEC) ##########" | tee -a "$LOG"
  # shellcheck disable=SC2086
  nohup "$BIN" --model "$M" $DRAFTARG $SPEC \
    --ctx-size 4096 -ngl 999 --split-mode none \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05 \
    --host 127.0.0.1 --port "$PORT" --api-key-file /etc/llama-server/api-keys \
    > "/tmp/robust-$tag.log" 2>&1 &
  pid=$!
  n=0; ok=0
  while [ "$n" -lt 240 ]; do
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 5; n=$((n+5))
  done
  echo "health ok=$ok after ${n}s" | tee -a "$LOG"
  if [ "$ok" = "1" ]; then
    ask "$tag p1" "$P1" "$tag"
    ask "$tag p2" "$P2" "$tag"
    ask "$tag p3" "$P3" "$tag"
  fi
  kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null; sleep 2
done
echo PROMPT_ROBUST_DONE
