#!/bin/bash
# MTP knob sweep on REAL prose, 3 prompts each (>=3 per lesson), fixed seed.
# Variants: n_max 3/4/5/6, plus p_min and n_min variants at n_max=4.
set -uo pipefail
BIN=/root/libdir-final/llama-server
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
KEY=$(head -1 /etc/llama-server/api-keys)
PORT=8099
LOG=/tmp/mtp-sweep.log
: > "$LOG"
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/root/libdir-final

P1="Explain how a GPU memory hierarchy works, in a few paragraphs of prose."
P2="Write a short story about a lighthouse keeper who finds a message in a bottle."
P3="Summarize the trade-offs between data parallelism and tensor parallelism for LLM inference."

run() { # $1=tag $2..=extra spec args
  tag="$1"; shift
  echo "" | tee -a "$LOG"
  echo "########## $tag : $* ##########" | tee -a "$LOG"
  nohup "$BIN" --model "$M" --spec-type draft-mtp "$@" \
    --ctx-size 4096 -ngl 999 --split-mode none \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05 \
    --host 127.0.0.1 --port "$PORT" --api-key-file /etc/llama-server/api-keys \
    > "/tmp/mtps-$tag.log" 2>&1 &
  pid=$!
  n=0; ok=0
  while [ "$n" -lt 240 ]; do
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 5; n=$((n+5))
  done
  echo "health ok=$ok" | tee -a "$LOG"
  if [ "$ok" = "1" ]; then
    for p in "$P1" "$P2" "$P3"; do
      curl -s -m 600 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"n_predict\":256,\"temperature\":0.7,\"top_p\":0.8,\"top_k\":20,\"repeat_penalty\":1.05,\"seed\":42,\"cache_prompt\":false}" \
        "http://127.0.0.1:$PORT/v1/chat/completions" > /dev/null 2>&1
      echo -n "[$tag] " | tee -a "$LOG"
      grep -E "eval time =" "/tmp/mtps-$tag.log" | tail -1 | sed 's/.*| *//' | tee -a "$LOG"
      grep -E "draft acceptance" "/tmp/mtps-$tag.log" | tail -1 | sed 's/.*| *//' | tee -a "$LOG"
    done
  fi
  kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null; sleep 2
}

run nmax3 --spec-draft-n-max 3
run nmax4 --spec-draft-n-max 4
run nmax5 --spec-draft-n-max 5
run nmax6 --spec-draft-n-max 6
run nmax4_pmin30 --spec-draft-n-max 4 --spec-draft-p-min 0.30
run nmax4_nmin2 --spec-draft-n-max 4 --spec-draft-n-min 2
echo MTP_SWEEP_DONE
