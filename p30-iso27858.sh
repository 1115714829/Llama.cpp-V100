#!/bin/bash
# Isolation: is the ~4% DFlash2 acceptance a *target-model* effect, or a broken CPU selector?
# Same binary/libdir/cards/sampling/seed/prompts as p29, but the target is Q2_K_XL (which the same
# draft scored mean-len 4.92 on, single card, in dflash-in-llamacpp.md 10.2).
#   (a) Q2_K_XL target, 3 cards, TENSOR, dflash n=7 -- if mean len jumps to ~4.9 the port is CORRECT
#       and the production-model 4% is a target-compatibility problem, not a selector bug
#   (b) Q2_K_XL target, 3 cards, TENSOR, mtp n=4    -- reference for the same target/cards
# n_predict 1024 + official sampling, matching the 10.2 protocol that produced 4.92.
set -uo pipefail
L=/root/libdir-27858
BIN=$L/llama-server
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
KEY=$(head -1 /etc/llama-server/api-keys)
PORT=8106
LOG=/tmp/iso27858.log
: > "$LOG"
export CUDA_VISIBLE_DEVICES=0,1,2
export LD_LIBRARY_PATH="$L"

P1="A train leaves Station A at 60 km/h. Another leaves Station B, 300 km away, at 40 km/h toward A. They start at the same time. Work through this step by step and explain when and where they meet."
P2="Explain, step by step, how to compute the sum of the first n integers and prove the formula by induction."
P3="Solve step by step: if 3x + 7 = 22, what is x? Then explain in detail how you checked your answer."

run() {
  tag="$1"; sm="$2"; shift 2
  echo "" | tee -a "$LOG"
  echo "########## $tag : split=$sm $* ##########" | tee -a "$LOG"
  nohup "$BIN" --model "$M" --split-mode "$sm" --tensor-split 1,1,1 \
    --ctx-size 8192 -ngl 999 "$@" \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
    --host 127.0.0.1 --port "$PORT" --api-key-file /etc/llama-server/api-keys \
    > "/tmp/iso27858-$tag.log" 2>&1 &
  pid=$!
  n=0; ok=0
  while [ "$n" -lt 300 ]; do
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 5; n=$((n+5))
  done
  echo "health ok=$ok after ${n}s" | tee -a "$LOG"
  if [ "$ok" != "1" ]; then
    grep -aE "GGML_ASSERT|error|failed to load|abort" "/tmp/iso27858-$tag.log" | tail -4 | tee -a "$LOG"
  else
    for p in "$P1" "$P2" "$P3"; do
      curl -s -m 900 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"n_predict\":1024,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"seed\":42,\"cache_prompt\":false}" \
        "http://127.0.0.1:$PORT/v1/chat/completions" > /dev/null 2>&1
      echo -n "[$tag] " | tee -a "$LOG"
      grep -E "eval time =" "/tmp/iso27858-$tag.log" | tail -1 | sed 's/.*| *//' | tee -a "$LOG"
      grep -E "draft acceptance" "/tmp/iso27858-$tag.log" | tail -1 | sed 's/.*| *//' | tee -a "$LOG"
    done
    grep -aE "capping draft context ubatch|unused tensor selector_" "/tmp/iso27858-$tag.log" | head -3 | tee -a "$LOG"
  fi
  kill "$pid" 2>/dev/null; sleep 4; kill -9 "$pid" 2>/dev/null; sleep 3
}

run q2xl_tensor_dflash7 tensor --model-draft "$D" --spec-type draft-dflash --spec-draft-n-max 7
run q2xl_tensor_mtp4     tensor --spec-type draft-mtp --spec-draft-n-max 4
echo ISO27858_DONE
