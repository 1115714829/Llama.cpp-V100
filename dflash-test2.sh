#!/bin/bash
# Retry draft-dflash WITHOUT --split-mode tensor (the crash was a meta-backend split-axis assert).
set -uo pipefail
BIN=/root/libdir-final/llama-server
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
KEY=$(head -1 /etc/llama-server/api-keys)
PORT=8097
LOG=/tmp/dflash2.log
: > "$LOG"

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/root/libdir-final

for SM in none layer; do
  echo "" | tee -a "$LOG"
  echo "########## split-mode=$SM ##########" | tee -a "$LOG"
  nohup "$BIN" --model "$M" --model-draft "$D" --spec-type draft-dflash \
    --ctx-size 4096 -ngl 999 --split-mode "$SM" \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05 \
    --host 127.0.0.1 --port "$PORT" --api-key-file /etc/llama-server/api-keys \
    > "/tmp/dflash2-$SM.log" 2>&1 &
  pid=$!
  n=0; ok=0
  while [ "$n" -lt 240 ]; do
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 5; n=$((n+5))
  done
  echo "health ok=$ok after ${n}s alive=$(kill -0 $pid 2>/dev/null && echo yes || echo no)" | tee -a "$LOG"
  if [ "$ok" = "1" ]; then
    curl -s -m 600 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
      -d '{"messages":[{"role":"user","content":"Explain how a GPU memory hierarchy works, in a few paragraphs of prose."}],"n_predict":256,"temperature":0.7,"top_p":0.8,"top_k":20,"repeat_penalty":1.05,"seed":42,"cache_prompt":false}' \
      "http://127.0.0.1:$PORT/v1/chat/completions" > "/tmp/dflash2-$SM.json" 2>&1
    echo "resp_bytes=$(wc -c < "/tmp/dflash2-$SM.json")" | tee -a "$LOG"
    grep -E "prompt eval time|eval time =|draft acceptance|n_gen =" "/tmp/dflash2-$SM.log" | tail -4 | tee -a "$LOG"
  else
    echo "--- failure tail ---" | tee -a "$LOG"
    grep -E "GGML_ASSERT|error|failed|abort" "/tmp/dflash2-$SM.log" | tail -6 | tee -a "$LOG"
  fi
  kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null; sleep 2
done
echo DFLASH2_DONE
