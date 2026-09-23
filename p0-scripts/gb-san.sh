#!/bin/bash
# Minimal repro under compute-sanitizer memcheck: bucket=1.25 + DFlash2 draft,
# one short greedy request -> the crash becomes a named kernel + invalid address.
set -u
export CUDA_VISIBLE_DEVICES=0,1,2
export LD_LIBRARY_PATH=/root/libdir-gb
export GGML_KV_BUCKET_RATIO=1.25
M=/mnt/3.84t/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
D=/mnt/3.84t/llm-models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
KEY=$(head -1 /etc/llama-server/api-keys)
CS=/usr/local/cuda-12.4/bin/compute-sanitizer
nohup "$CS" --tool memcheck --error-exitcode 9 \
  /root/libdir-gb/llama-server --model "$M" --model-draft "$D" \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --split-mode tensor --tensor-split 1,1,1 --ctx-size 2048 -ngl 999 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --host 127.0.0.1 --port 8331 --api-key "$KEY" \
  > /tmp/gb-san-server.log 2>&1 &
SAN_PID=$!
echo "SAN_PID=$SAN_PID"
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" http://127.0.0.1:8331/health)
  if [ "$code" = "200" ]; then echo "health up after ${i}x5s"; break; fi
  sleep 5
done
curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","n_predict":32,"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"repeat_penalty":1.0}' \
  http://127.0.0.1:8331/completion > /tmp/gb-san-req.json 2>&1
sleep 8
if kill -0 $SAN_PID 2>/dev/null; then echo "SERVER_STILL_ALIVE"; kill $SAN_PID; else echo "SERVER_DIED"; fi
echo ===SAN_REPORT===
grep -aE '=========|Invalid|illegal|ERROR' /tmp/gb-san-server.log | head -20
echo ===STACK_TOP===
grep -a -A6 'Invalid' /tmp/gb-san-server.log | head -30
echo SAN_DONE
