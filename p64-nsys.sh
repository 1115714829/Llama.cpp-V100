#!/bin/bash
# Ground-truth GPU attribution with nsys: which kernels actually dominate a speculative decode round?
# Uses a smaller context and short generation to keep the report small; NO drop_caches so the model
# stays hot and the load phase (which we do not care about) is short.
set -uo pipefail
L=/root/libdir-rt
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
NSYS=/usr/local/cuda-12.4/bin/nsys
KEY=$(head -1 /etc/llama-server/api-keys)
LOG=/tmp/p64.log
: > "$LOG"
export LD_LIBRARY_PATH="$L"
export CUDA_VISIBLE_DEVICES=0,1,2

echo "=== nsys present? ===" | tee -a "$LOG"
ls -l "$NSYS" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== launch server under nsys (layer, 3 cards, ctx 2048) ===" | tee -a "$LOG"
nohup "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none \
  -o /tmp/nsys-layer --force-overwrite=true \
  "$L/llama-server" --model "$M" --model-draft "$D" \
    --spec-type draft-dflash --spec-draft-n-max 7 \
    --split-mode layer --tensor-split 1,1,1 --ctx-size 2048 -ngl 999 \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --host 127.0.0.1 --port 8121 --api-key-file /etc/llama-server/api-keys \
  > /tmp/p64-server.log 2>&1 &
pid=$!

n=0; ok=0
while [ "$n" -lt 600 ]; do
  if curl -s -m 5 "http://127.0.0.1:8121/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
  kill -0 "$pid" 2>/dev/null || break
  sleep 15; n=$((n+15))
done
echo "health ok=$ok after ${n}s" | tee -a "$LOG"

if [ "$ok" = "1" ]; then
  P="Explain, step by step, how to compute the sum of the first n integers and prove the formula by induction."
  body="{\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"n_predict\":96,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"seed\":42,\"cache_prompt\":false}"
  curl -s -m 900 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d "$body" "http://127.0.0.1:8121/v1/chat/completions" > /dev/null 2>&1
  grep -a -E "eval time =|draft acceptance" /tmp/p64-server.log | tail -2 | tee -a "$LOG"
fi

kill "$pid" 2>/dev/null; sleep 8; kill -9 "$pid" 2>/dev/null; sleep 5
echo "report file:" | tee -a "$LOG"
ls -l /tmp/nsys-layer.nsys-rep 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== top GPU kernels by total time (whole run incl. load) ===" | tee -a "$LOG"
"$NSYS" stats --report cuda_gpu_kern_sum --format table /tmp/nsys-layer.nsys-rep 2>&1 | head -45 | tee -a "$LOG"
echo P64_DONE
