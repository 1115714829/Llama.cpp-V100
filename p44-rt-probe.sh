#!/bin/bash
# Decisive: unconditional fprintf at llama_decode / ctx::decode / process_ubatch / perf_get_data.
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
L=/root/libdir-rt
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
LOG=/tmp/p44.log
: > "$LOG"
exec >> "$LOG" 2>&1

echo "=== rebuild ==="
sed -i 's/\r$//' "$SRC/src/llama-context.cpp"
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j82 --target llama-server" > /tmp/build-p44.log 2>&1 &
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do
  sleep 10; n=$((n+10)); [ "$n" -gt 700 ] && { echo "build timeout"; break; }
done
echo -n "build errors          : "; grep -icE "error:|FAILED" /tmp/build-p44.log
echo -n "llama-context compiled: "; grep -c "llama-context.cpp.o" /tmp/build-p44.log
echo -n "[RT] marker in new lib: "; grep -c -a "perf_get_data" "$SRC/build/bin/libllama.so.0.4.1"

echo ""
echo "=== snapshot + short server run (single card, Q2_K_XL) ==="
rm -rf "$L" && mkdir -p "$L" && cp -a "$SRC/build/bin/." "$L/"
export LD_LIBRARY_PATH="$L"
export CUDA_VISIBLE_DEVICES=0

nohup "$L/llama-server" --model "$M" --ctx-size 2048 -ngl 999 \
  --host 127.0.0.1 --port 8114 > /tmp/p44-server.log 2>&1 &
pid=$!
n=0
while [ "$n" -lt 360 ]; do
  curl -s -m 5 "http://127.0.0.1:8114/health" 2>/dev/null | grep -q '"ok"' && break
  sleep 10; n=$((n+10))
done
echo "health after ${n}s"
echo "mapped libllama of pid $pid:"
grep -a "libllama.so" "/proc/$pid/maps" 2>/dev/null | awk '{print $6}' | sort -u

curl -s -m 300 -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello."}],"n_predict":48,"temperature":0.0,"seed":1,"cache_prompt":false}' \
  "http://127.0.0.1:8114/v1/chat/completions" > /dev/null 2>&1
kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null

echo ""
echo "=== [RT] raw-stderr diagnostic lines ==="
grep -a -F "[RT]" /tmp/p44-server.log | head -20
echo ""
echo "=== sanity: did the request actually generate? ==="
grep -a -E "eval time =|prompt eval time" /tmp/p44-server.log | tail -2
echo P44_DONE
