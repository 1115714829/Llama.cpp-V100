#!/bin/bash
# profile-tgt2.sh -- correct nsys run: terminate the APP (not nsys) so nsys finalizes the report.
set -u
L=/root/libdir-nccl
PKGLIB=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/lib
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
KEY=$(head -1 /etc/llama-server/api-keys)
NSYS=/usr/local/cuda-12.4/bin/nsys
PORT=8191
OUT=/tmp/prof2
rm -rf "$OUT".*

echo "=== make sure no stale server ==="
pgrep -af 'llama-server' || echo none

echo "=== launch under nsys ==="
env CUDA_VISIBLE_DEVICES=0,1,2 LD_LIBRARY_PATH="$L:$PKGLIB" GGML_CUDA_P2P=1 \
  "$NSYS" profile -o "$OUT" --force-overwrite true --trace cuda \
  "$L/llama-server" --model "$M" --model-draft "$D" \
    --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-ngl 999 \
    --split-mode tensor --tensor-split 1,1,1 --ctx-size 8192 -ngl 999 \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
    --host 127.0.0.1 --port "$PORT" --api-key-file /etc/llama-server/api-keys \
    > /tmp/prof2-server.log 2>&1 &
NSYS_PID=$!

n=0
while [ "$n" -lt 600 ]; do
  curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
  kill -0 "$NSYS_PID" 2>/dev/null || { echo "nsys died"; tail -5 /tmp/prof2-server.log; exit 1; }
  sleep 15; n=$((n+15))
done
echo "health after ${n}s"

P1="Explain, step by step, how to compute the sum of the first n integers and prove the formula by induction."
body="{\"messages\":[{\"role\":\"user\",\"content\":\"$P1\"}],\"n_predict\":512,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"presence_penalty\":0.0,\"repetition_penalty\":1.0,\"seed\":42,\"cache_prompt\":false,\"chat_template_kwargs\":{\"enable_thinking\":true,\"preserve_thinking\":true,\"reasoning_effort\":\"xhigh\"}}"
echo "=== request ==="
curl -s -m 900 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "$body" "http://127.0.0.1:$PORT/v1/chat/completions" > /tmp/prof2-resp.json
grep -a -oE "[0-9]+\.[0-9]+ tokens per second" /tmp/prof2-server.log | tail -1

echo "=== terminate the APP so nsys finalizes ==="
APP_PID=$(pgrep -f "libdir-nccl/llama-server" | head -1)
echo "app pid=$APP_PID"
kill -TERM "$APP_PID" 2>/dev/null
wait "$NSYS_PID" 2>/dev/null
echo "nsys exit=$?"
sleep 5
ls -l "$OUT".nsys-rep 2>&1

echo "=== kernel summary ==="
"$NSYS" stats --report cuda_gpu_kern_sum "$OUT".nsys-rep 2>&1 | tail -40
echo PROF2_DONE
