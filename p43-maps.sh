#!/bin/bash
# Ground truth: which libllama.so FILE does the running process actually map?
set -uo pipefail
L=/root/libdir-rt
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
LOG=/tmp/p43.log
: > "$LOG"
export LD_LIBRARY_PATH="$L"
export CUDA_VISIBLE_DEVICES=0
export LLAMA_ROUND_TIMING=1

echo "=== did ANY libllama INFO line appear under llama-bench? (p41 log) ===" | tee -a "$LOG"
grep -a -c -e "llama_model_loader" -e "print_info" -e "load time" /tmp/p41.log | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== launch server (Q2_K_XL, 1 card, ctx 2048) with the env set ===" | tee -a "$LOG"
nohup "$L/llama-server" --model "$M" --ctx-size 2048 -ngl 999 \
  --host 127.0.0.1 --port 8113 > /tmp/p43-server.log 2>&1 &
pid=$!
sleep 20

echo "--- mapped libllama/ggml files of pid $pid ---" | tee -a "$LOG"
grep -a -e "libllama" -e "libggml" "/proc/$pid/maps" 2>/dev/null | awk '{print $6}' | sort -u | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- marker count in each mapped path ---" | tee -a "$LOG"
for f in $(grep -a "libllama" "/proc/$pid/maps" 2>/dev/null | awk '{print $6}' | sort -u) ; do
  echo "$(grep -c -a 'round timing (LLAMA_ROUND_TIMING)' "$f" 2>/dev/null)  $f" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "--- env of pid $pid ---" | tee -a "$LOG"
xargs -0 -n1 < "/proc/$pid/environ" 2>/dev/null | grep LLAMA_ROUND_TIMING | tee -a "$LOG"

n=0
while [ "$n" -lt 300 ]; do
  curl -s -m 5 "http://127.0.0.1:8113/health" 2>/dev/null | grep -q '"ok"' && break
  sleep 10; n=$((n+10))
done
echo "health after ${n}s" | tee -a "$LOG"

curl -s -m 300 -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello and count to five."}],"n_predict":64,"temperature":0.0,"seed":42,"cache_prompt":false}' \
  "http://127.0.0.1:8113/v1/chat/completions" > /dev/null 2>&1

echo "" | tee -a "$LOG"
echo "=== our markers in the server log ===" | tee -a "$LOG"
grep -a -e "LLAMA_ROUND_TIMING = " -e "round timing" /tmp/p43-server.log | head -6
echo "=== sanity: any libllama INFO in server log? ===" | tee -a "$LOG"
grep -a -c -e "llama_model_loader" -e "print_info" -e "graphs reused" /tmp/p43-server.log | tee -a "$LOG"

kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null
echo P43_DONE
