#!/bin/bash
# Matched-contract comparison on the OFFICIAL target, with 1cat's chat-template kwargs:
#   (a) tensor + dflash n=7   <- 1cat-aligned
#   (b) tensor + mtp     n=4  <- 1cat's pre-1.5.0 config (base model is MTP-trained)
#   (c) layer  + dflash n=7   <- avoids the 23.4 ms/round CPU selector + CPU sampling
# Sampling = official Qwen thinking-mode set + presence/repetition explicit.
set -uo pipefail
L=/root/libdir-rt
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
LOG=/tmp/p49.log
: > "$LOG"
export LD_LIBRARY_PATH="$L"
export CUDA_VISIBLE_DEVICES=0,1,2
export LLAMA_ROUND_TIMING=1
export LLAMA_SPEC_TIMING=1
KEY=$(head -1 /etc/llama-server/api-keys)
PORT=8119

P="Explain, step by step, how to compute the sum of the first n integers and prove the formula by induction."
KW='"chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}'

run() {
  tag="$1"; sm="$2"; shift 2
  echo "" | tee -a "$LOG"
  echo "########## $tag : split=$sm $* ##########" | tee -a "$LOG"
  nohup "$L/llama-server" --model "$M" --split-mode "$sm" --tensor-split 1,1,1 "$@" \
    --ctx-size 8192 -ngl 999 \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 \
    --host 127.0.0.1 --port "$PORT" --api-key-file /etc/llama-server/api-keys \
    > "/tmp/p49-$tag.log" 2>&1 &
  pid=$!
  n=0; ok=0
  while [ "$n" -lt 900 ]; do
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 15; n=$((n+15))
  done
  echo "health ok=$ok after ${n}s" | tee -a "$LOG"
  if [ "$ok" = "1" ]; then
    local body="{\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"n_predict\":512,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"presence_penalty\":0.0,\"repetition_penalty\":1.0,\"seed\":42,\"cache_prompt\":false,$KW}"
    curl -s -m 1800 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
      -d "$body" "http://127.0.0.1:$PORT/v1/chat/completions" > "/tmp/p49-$tag.json" 2>&1
    echo -n "[$tag] resp: " | tee -a "$LOG"
    grep -o '"completion_tokens":[0-9]*' "/tmp/p49-$tag.json" | head -1 | tee -a "$LOG"
    grep -o '"finish_reason":"[a-z]*"' "/tmp/p49-$tag.json" | head -1 | tee -a "$LOG"
    echo -n "[$tag] " | tee -a "$LOG"
    grep -a -E "eval time =" "/tmp/p49-$tag.log" | tail -1 | sed 's/.*| *//' | tee -a "$LOG"
    echo -n "[$tag] " | tee -a "$LOG"
    grep -a -E "draft acceptance" "/tmp/p49-$tag.log" | tail -1 | sed 's/.*| *//' | tee -a "$LOG"
    echo -n "[$tag] draft phases: " | tee -a "$LOG"
    grep -a "spec timing" "/tmp/p49-$tag.log" | tail -1 | sed 's/.*spec timing: //' | tee -a "$LOG"
    echo -n "[$tag] target round: " | tee -a "$LOG"
    grep -a -F "[RT] perf:" "/tmp/p49-$tag.log" | tail -1 | sed 's/.*perf: //' | tee -a "$LOG"
  fi
  kill "$pid" 2>/dev/null; sleep 4; kill -9 "$pid" 2>/dev/null; sleep 2
}

run tensor_dflash7 tensor --model-draft "$D" --spec-type draft-dflash --spec-draft-n-max 7
run tensor_mtp4     tensor --spec-type draft-mtp --spec-draft-n-max 4
run layer_dflash7   layer  --model-draft "$D" --spec-type draft-dflash --spec-draft-n-max 7
echo P49_DONE
