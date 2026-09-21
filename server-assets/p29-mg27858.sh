#!/bin/bash
# Phase 1a payoff: does the PR #27858 port make tensor + draft-dflash work on 3 cards, and does it beat tensor + MTP?
#   (a) tensor + mtp     n_max=4   -- baseline, same as the 58.6 t/s measurement (re-measured here for a clean same-session comparison)
#   (b) tensor + dflash  n_max=7   -- official setting; previously SIGABRT
#   (c) tensor + dflash  n_max=4   -- depth-robust setting (community: n=4 wins at 32k depth)
#   (d) layer  + dflash  n_max=7   -- Phase 1b probe: does layer still give ~4% acceptance?
# ALL arms use the SAME libdir (/root/libdir-27858), same cards (0,1,2 = one NUMA node), same ctx,
# same sampling (official temp 1.0 / top-p 0.95 / top-k 20), same seed 42, same 3 prompts, n_predict 512.
set -uo pipefail
L=/root/libdir-27858
BIN=$L/llama-server
M=/root/llm/models/Qwen3.8-27B-TurboFCFusion-gguf/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
KEY=$(head -1 /etc/llama-server/api-keys)
PORT=8105
LOG=/tmp/mg27858.log
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
    > "/tmp/mg27858-$tag.log" 2>&1 &
  pid=$!
  n=0; ok=0
  while [ "$n" -lt 300 ]; do
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 5; n=$((n+5))
  done
  echo "health ok=$ok after ${n}s" | tee -a "$LOG"
  if [ "$ok" != "1" ]; then
    echo "--- failure evidence ---" | tee -a "$LOG"
    grep -aE "GGML_ASSERT|error|failed to load|abort" "/tmp/mg27858-$tag.log" | tail -4 | tee -a "$LOG"
  else
    for p in "$P1" "$P2" "$P3"; do
      curl -s -m 900 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"n_predict\":512,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"seed\":42,\"cache_prompt\":false}" \
        "http://127.0.0.1:$PORT/v1/chat/completions" > /dev/null 2>&1
      echo -n "[$tag] " | tee -a "$LOG"
      grep -E "eval time =" "/tmp/mg27858-$tag.log" | tail -1 | sed 's/.*| *//' | tee -a "$LOG"
      grep -E "draft acceptance" "/tmp/mg27858-$tag.log" | tail -1 | sed 's/.*| *//' | tee -a "$LOG"
    done
    echo "--- selector path evidence ---" | tee -a "$LOG"
    grep -aE "capping draft context ubatch|unused tensor selector_" "/tmp/mg27858-$tag.log" | head -3 | tee -a "$LOG"
  fi
  kill "$pid" 2>/dev/null; sleep 4; kill -9 "$pid" 2>/dev/null; sleep 3
}

echo "=== binary provenance (same libdir for every arm) ===" | tee -a "$LOG"
md5sum "$L/libllama.so.0.4.1" "$L/libggml-cuda.so.0.24.0" | tee -a "$LOG"

run tensor_mtp4     tensor --spec-type draft-mtp --spec-draft-n-max 4
run tensor_dflash7  tensor --model-draft "$D" --spec-type draft-dflash --spec-draft-n-max 7
run tensor_dflash4  tensor --model-draft "$D" --spec-type draft-dflash --spec-draft-n-max 4
run layer_dflash7   layer  --model-draft "$D" --spec-type draft-dflash --spec-draft-n-max 7
echo MG27858_DONE
