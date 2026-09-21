#!/bin/bash
# Phase 0: (a) confirm the draft is recognised as DFlash2, (b) feed the full block by sweeping n_max.
# block_size=8 for our draft => n_draft_max = block_size-1 = 7 (common/speculative.cpp:1010).
# n_max=9 is included on purpose to verify the clamping.
set -uo pipefail
BIN=/root/libdir-final/llama-server
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
KEY=$(head -1 /etc/llama-server/api-keys)
PORT=8100
LOG=/tmp/dflash2-nmax.log
: > "$LOG"
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/root/libdir-final

P1="Explain how a GPU memory hierarchy works, in a few paragraphs of prose."
P2="Write a short story about a lighthouse keeper who finds a message in a bottle."
P3="Summarize the trade-offs between data parallelism and tensor parallelism for LLM inference."

echo "=== Phase0 start $(date -Is) ===" | tee -a "$LOG"
echo "draft: $(ls -la "$D" | awk '{print $5, $9}')" | tee -a "$LOG"

run() {
  tag="$1"; shift
  echo "" | tee -a "$LOG"
  echo "########## $tag : $* ##########" | tee -a "$LOG"
  nohup "$BIN" --model "$M" --model-draft "$D" --spec-type draft-dflash "$@" \
    --ctx-size 4096 -ngl 999 --split-mode none \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05 \
    --host 127.0.0.1 --port "$PORT" --api-key-file /etc/llama-server/api-keys \
    > "/tmp/d2-$tag.log" 2>&1 &
  pid=$!
  n=0; ok=0
  while [ "$n" -lt 240 ]; do
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 5; n=$((n+5))
  done
  echo "health ok=$ok after ${n}s" | tee -a "$LOG"

  # (a) DFlash2 recognition + (b) the n_max clamp — the decisive lines
  echo "--- model/DFlash2 identification ---" | tee -a "$LOG"
  grep -E "DFlash2|selector|conv kernel|DFlash with DSpark" "/tmp/d2-$tag.log" | head -6 | tee -a "$LOG"
  echo "--- speculative impl / clamp ---" | tee -a "$LOG"
  grep -E "adding speculative implementation|n_max=|block_size=|requested draft size|clamping|mask_token_id" "/tmp/d2-$tag.log" | head -8 | tee -a "$LOG"

  if [ "$ok" = "1" ]; then
    for p in "$P1" "$P2" "$P3"; do
      curl -s -m 600 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"n_predict\":256,\"temperature\":0.7,\"top_p\":0.8,\"top_k\":20,\"repeat_penalty\":1.05,\"seed\":42,\"cache_prompt\":false}" \
        "http://127.0.0.1:$PORT/v1/chat/completions" > /dev/null 2>&1
      echo -n "[$tag] " | tee -a "$LOG"
      grep -E "eval time =" "/tmp/d2-$tag.log" | tail -1 | sed 's/.*| *//' | tee -a "$LOG"
      grep -E "draft acceptance" "/tmp/d2-$tag.log" | tail -1 | sed 's/.*| *//' | tee -a "$LOG"
    done
  fi
  kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null; sleep 2
}

run nmax3 --spec-draft-n-max 3
run nmax5 --spec-draft-n-max 5
run nmax7 --spec-draft-n-max 7
run nmax9 --spec-draft-n-max 9
echo DF2_NMAX_DONE
