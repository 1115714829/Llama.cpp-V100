#!/bin/bash
# Sizing experiment: how does the round cost scale with the verification batch M = n_max + 1?
#   none      -> M = 1  (target-only decode, AL = 1)
#   dflash 1  -> M = 2
#   dflash 3  -> M = 4
#   dflash 7  -> M = 8
# If cost per round is ~flat in M, the forward is bandwidth-bound => Tensor Cores cannot help much.
# If cost grows ~linearly with M, it is compute-bound => HMMA Tensor Cores are the lever.
set -uo pipefail
L=/root/libdir-rt
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
LOG=/tmp/p47.log
: > "$LOG"
export LD_LIBRARY_PATH="$L"
export CUDA_VISIBLE_DEVICES=0,1,2
export LLAMA_ROUND_TIMING=1
KEY=$(head -1 /etc/llama-server/api-keys)
PORT=8117
P="Explain, step by step, how to compute the sum of the first n integers and prove the formula by induction."

run() {
  tag="$1"; shift
  echo "" | tee -a "$LOG"
  echo "########## $tag : $* ##########" | tee -a "$LOG"
  nohup "$L/llama-server" --model "$M" --split-mode tensor --tensor-split 1,1,1 "$@" \
    --ctx-size 8192 -ngl 999 \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --temp 1.0 --top-p 0.95 --top-k 20 \
    --host 127.0.0.1 --port "$PORT" --api-key-file /etc/llama-server/api-keys \
    > "/tmp/p47-$tag.log" 2>&1 &
  pid=$!
  n=0; ok=0
  while [ "$n" -lt 900 ]; do
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 15; n=$((n+15))
  done
  echo "health ok=$ok after ${n}s" | tee -a "$LOG"
  if [ "$ok" = "1" ]; then
    curl -s -m 1800 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
      -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"n_predict\":512,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"seed\":42,\"cache_prompt\":false}" \
      "http://127.0.0.1:$PORT/v1/chat/completions" > /dev/null 2>&1
    echo -n "[$tag] " | tee -a "$LOG"
    grep -a -E "eval time =" "/tmp/p47-$tag.log" | tail -1 | sed 's/.*| *//' | tee -a "$LOG"
    echo -n "[$tag] " | tee -a "$LOG"
    grep -a -E "draft acceptance" "/tmp/p47-$tag.log" | tail -1 | sed 's/.*| *//' | tee -a "$LOG"
    echo -n "[$tag] host-side: " | tee -a "$LOG"
    grep -a -F "[RT] perf:" "/tmp/p47-$tag.log" | tail -1 | sed 's/.*perf: //' | tee -a "$LOG"
  fi
  kill "$pid" 2>/dev/null; sleep 4; kill -9 "$pid" 2>/dev/null; sleep 2
}

run none      --spec-type none
run dflash3   --model-draft "$D" --spec-type draft-dflash --spec-draft-n-max 3
run dflash7   --model-draft "$D" --spec-type draft-dflash --spec-draft-n-max 7
echo P47_DONE
