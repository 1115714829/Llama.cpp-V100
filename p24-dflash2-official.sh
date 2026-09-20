#!/bin/bash
# (a) confirm the DFlash2 selector path actually engaged in earlier runs
# (b) replicate the OFFICIAL conditions from z-lab/Qwen3.8-27B-DFlash2-GGUF:
#     --spec-draft-n-max 7, temp 1.0 / top_p 0.95 / top_k 20, reasoning-style prompts, long generation.
set -uo pipefail
BIN=/root/libdir-final/llama-server
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
KEY=$(head -1 /etc/llama-server/api-keys)
PORT=8101
LOG=/tmp/dflash2-official.log
: > "$LOG"
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/root/libdir-final

echo "=== (a) DFlash2 identification lines in earlier runs ===" | tee -a "$LOG"
for f in /tmp/d2-nmax3.log /tmp/d2-nmax7.log; do
  echo "--- $f" | tee -a "$LOG"
  grep -a -iE "DFlash2 conv kernel|selector rank|DFlash2 model|DFlash with DSpark|DFlash2" "$f" 2>/dev/null | head -4 | tee -a "$LOG"
done
echo "" | tee -a "$LOG"

# reasoning-style prompts (closer to the official GSM8K/xhigh-reasoning conditions than short prose)
P1="A train leaves Station A at 60 km/h. Another leaves Station B, 300 km away, at 40 km/h toward A. They start at the same time. Work through this step by step and explain when and where they meet."
P2="Explain, step by step, how to compute the sum of the first n integers and prove the formula by induction."
P3="Solve step by step: if 3x + 7 = 22, what is x? Then explain in detail how you checked your answer."

run() {
  tag="$1"; shift
  echo "########## $tag : $* ##########" | tee -a "$LOG"
  nohup "$BIN" --model "$M" --model-draft "$D" --spec-type draft-dflash "$@" \
    --ctx-size 8192 -ngl 999 --split-mode none \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
    --host 127.0.0.1 --port "$PORT" --api-key-file /etc/llama-server/api-keys \
    > "/tmp/off-$tag.log" 2>&1 &
  pid=$!
  n=0; ok=0
  while [ "$n" -lt 240 ]; do
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 5; n=$((n+5))
  done
  echo "health ok=$ok" | tee -a "$LOG"
  grep -a -iE "DFlash2 conv kernel|selector rank" "/tmp/off-$tag.log" | head -3 | tee -a "$LOG"
  if [ "$ok" = "1" ]; then
    for p in "$P1" "$P2" "$P3"; do
      curl -s -m 900 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"n_predict\":1024,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"seed\":42,\"cache_prompt\":false}" \
        "http://127.0.0.1:$PORT/v1/chat/completions" > /dev/null 2>&1
      echo -n "[$tag] " | tee -a "$LOG"
      grep -E "eval time =" "/tmp/off-$tag.log" | tail -1 | sed 's/.*| *//' | tee -a "$LOG"
      grep -E "draft acceptance" "/tmp/off-$tag.log" | tail -1 | sed 's/.*| *//' | tee -a "$LOG"
    done
  fi
  kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null; sleep 2
}

run nmax3 --spec-draft-n-max 3
run nmax7 --spec-draft-n-max 7
echo OFFICIAL_DONE
