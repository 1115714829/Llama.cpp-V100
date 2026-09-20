#!/bin/bash
# High-value test: llama.cpp has --spec-type draft-dflash, and the box already holds 1cat's DFlash2 GGUF.
# Same controlled setup as the spec sweep (ctx 4096, seed 42, real prose) so numbers are comparable.
set -uo pipefail
BIN=/root/libdir-final/llama-server
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
KEY=$(head -1 /etc/llama-server/api-keys)
PORT=8096
LOG=/tmp/dflash.log
: > "$LOG"

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/root/libdir-final

echo "=== DFlash2 GGUF dir ===" | tee -a "$LOG"
ls -la /root/llm/models/Qwen3.8-27B-DFlash2-GGUF/ 2>&1 | head -20 | tee -a "$LOG"
DRAFT=$(ls /root/llm/models/Qwen3.8-27B-DFlash2-GGUF/*.gguf 2>/dev/null | head -1)
echo "draft candidate: $DRAFT" | tee -a "$LOG"
if [ -z "$DRAFT" ]; then echo "NO_DFLASH_GGUF"; exit 0; fi

echo "=== is draft-dflash accepted as a spec type? ===" | tee -a "$LOG"
"$BIN" --help 2>&1 | grep -A3 -- "--spec-type" | head -6 | tee -a "$LOG"

echo "=== start server: --spec-type draft-dflash --model-draft <dflash gguf> ===" | tee -a "$LOG"
nohup "$BIN" --model "$M" --model-draft "$DRAFT" --spec-type draft-dflash \
  --ctx-size 4096 -ngl 999 --split-mode tensor --tensor-split 1 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
  --temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05 \
  --host 127.0.0.1 --port "$PORT" --api-key-file /etc/llama-server/api-keys \
  > "/tmp/dflash-server.log" 2>&1 &
pid=$!
n=0; ok=0
while [ "$n" -lt 300 ]; do
  if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
  kill -0 "$pid" 2>/dev/null || break
  sleep 5; n=$((n+5))
done
echo "health ok=$ok after ${n}s  pid_alive=$(kill -0 $pid 2>/dev/null && echo yes || echo no)" | tee -a "$LOG"

if [ "$ok" = "1" ]; then
  curl -s -m 600 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Explain how a GPU memory hierarchy works, in a few paragraphs of prose."}],"n_predict":256,"temperature":0.7,"top_p":0.8,"top_k":20,"repeat_penalty":1.05,"seed":42,"cache_prompt":false}' \
    "http://127.0.0.1:$PORT/v1/chat/completions" > /tmp/dflash-resp.json 2>&1
  echo "resp_bytes=$(wc -c < /tmp/dflash-resp.json)" | tee -a "$LOG"
  grep -E "prompt eval time|eval time =|draft acceptance|n_gen =" /tmp/dflash-server.log | tail -4 | tee -a "$LOG"
else
  echo "--- server log tail (why it failed) ---" | tee -a "$LOG"
  tail -25 /tmp/dflash-server.log | tee -a "$LOG"
fi
kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null
echo DFLASH_TEST_DONE
