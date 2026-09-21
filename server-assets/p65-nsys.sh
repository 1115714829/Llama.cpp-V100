#!/bin/bash
# nsys attribution, second attempt: stop the profiled process gracefully so the report is written.
set -uo pipefail
L=/root/libdir-rt
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
NSYS=/usr/local/cuda-12.4/bin/nsys
KEY=$(head -1 /etc/llama-server/api-keys)
LOG=/tmp/p65.log
: > "$LOG"
export LD_LIBRARY_PATH="$L"
export CUDA_VISIBLE_DEVICES=0,1,2
rm -f /tmp/nsys-layer.nsys-rep /tmp/nsys-layer.qdrep /tmp/nsys-layer.sqlite

echo "=== launch under nsys (layer, 3 cards, ctx 2048, 192 tokens) ===" | tee -a "$LOG"
nohup "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none \
  -o /tmp/nsys-layer --force-overwrite=true \
  "$L/llama-server" --model "$M" --model-draft "$D" \
    --spec-type draft-dflash --spec-draft-n-max 7 \
    --split-mode layer --tensor-split 1,1,1 --ctx-size 2048 -ngl 999 \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --host 127.0.0.1 --port 8122 --api-key-file /etc/llama-server/api-keys \
  > /tmp/p65-server.log 2>&1 &
pid=$!

n=0; ok=0
while [ "$n" -lt 600 ]; do
  if curl -s -m 5 "http://127.0.0.1:8122/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
  kill -0 "$pid" 2>/dev/null || break
  sleep 15; n=$((n+15))
done
echo "health ok=$ok after ${n}s" | tee -a "$LOG"

if [ "$ok" = "1" ]; then
  P="Explain, step by step, how to compute the sum of the first n integers and prove the formula by induction."
  body="{\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"n_predict\":192,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"seed\":42,\"cache_prompt\":false}"
  curl -s -m 900 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d "$body" "http://127.0.0.1:8122/v1/chat/completions" > /dev/null 2>&1
  grep -a -E "eval time =|draft acceptance" /tmp/p65-server.log | tail -2 | tee -a "$LOG"
fi

echo "=== graceful stop (SIGINT so nsys can finalize) ===" | tee -a "$LOG"
kill -INT "$pid" 2>/dev/null
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  kill -0 "$pid" 2>/dev/null || break
  sleep 5
done
kill -0 "$pid" 2>/dev/null && { echo "(still alive, SIGTERM)"; kill -TERM "$pid" 2>/dev/null; sleep 10; }
kill -0 "$pid" 2>/dev/null && { echo "(still alive, SIGKILL)"; kill -9 "$pid" 2>/dev/null; sleep 5; }

echo "report file:" | tee -a "$LOG"
ls -l /tmp/nsys-layer.nsys-rep 2>&1 | tee -a "$LOG"

if [ -f /tmp/nsys-layer.nsys-rep ]; then
  echo "" | tee -a "$LOG"
  echo "=== top GPU kernels by total time ===" | tee -a "$LOG"
  "$NSYS" stats --report cuda_gpu_kern_sum --format table /tmp/nsys-layer.nsys-rep 2>&1 | head -50 | tee -a "$LOG"
fi
echo P65_DONE
