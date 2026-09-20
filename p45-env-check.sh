#!/bin/bash
# Does getenv() inside libllama.so actually see LLAMA_ROUND_TIMING when the server is started with it?
set -uo pipefail
L=/root/libdir-rt
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
LOG=/tmp/p45.log
: > "$LOG"
exec >> "$LOG" 2>&1

export LD_LIBRARY_PATH="$L"
export CUDA_VISIBLE_DEVICES=0
export LLAMA_ROUND_TIMING=1

echo "shell env: LLAMA_ROUND_TIMING=[$LLAMA_ROUND_TIMING]"

nohup "$L/llama-server" --model "$M" --ctx-size 2048 -ngl 999 \
  --host 127.0.0.1 --port 8115 > /tmp/p45-server.log 2>&1 &
pid=$!
sleep 3
echo -n "in /proc/$pid/environ: "
xargs -0 -n1 < "/proc/$pid/environ" 2>/dev/null | grep LLAMA_ROUND_TIMING || echo "(absent)"

n=0
while [ "$n" -lt 360 ]; do
  curl -s -m 5 "http://127.0.0.1:8115/health" 2>/dev/null | grep -q '"ok"' && break
  sleep 10; n=$((n+10))
done
echo "health after ${n}s"

curl -s -m 300 -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello."}],"n_predict":48,"temperature":0.0,"seed":1,"cache_prompt":false}' \
  "http://127.0.0.1:8115/v1/chat/completions" > /dev/null 2>&1
kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null

echo ""
echo "=== [RT] lines (rt_enabled tells us whether getenv worked inside libllama) ==="
grep -a -F "[RT]" /tmp/p45-server.log | head -20
echo P45_DONE
