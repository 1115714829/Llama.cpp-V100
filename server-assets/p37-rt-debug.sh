#!/bin/bash
# Phase 0.2b: rebuild the self-diagnosing instrument and run a short controlled test to find out
# why the round-timing lines did not appear.
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
L=/root/libdir-rt
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
PORT=8111
LOG=/tmp/rt-debug.log
: > "$LOG"
exec >> "$LOG" 2>&1

echo "=== [1] normalize + rebuild ==="
sed -i 's/\r$//' "$SRC/src/llama-context.cpp"
echo -n "LLAMA_ROUND_TIMING refs in source: "; grep -c LLAMA_ROUND_TIMING "$SRC/src/llama-context.cpp"
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j82 --target llama-server" > /tmp/build-rt2.log 2>&1 &
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do
  sleep 10; n=$((n+10)); [ "$n" -gt 600 ] && { echo "build timeout"; break; }
done
echo -n "build errors: "; grep -icE "error:|FAILED" /tmp/build-rt2.log
echo "--- build tail ---"; tail -3 /tmp/build-rt2.log

echo ""
echo "=== [2] re-snapshot ==="
rm -rf "$L" && mkdir -p "$L" && cp -a "$SRC/build/bin/." "$L/"
echo -n "instrument strings in libllama.so: "; grep -c -a llama_round_timing "$L/libllama.so.0.4.1"

echo ""
echo "=== [3] launch with LLAMA_ROUND_TIMING set inline on the command ==="
KEY=$(head -1 /etc/llama-server/api-keys)
export CUDA_VISIBLE_DEVICES=0,1,2
export LD_LIBRARY_PATH="$L"

LLAMA_ROUND_TIMING=1 nohup "$L/llama-server" --model "$M" --model-draft "$D" \
  --split-mode tensor --tensor-split 1,1,1 \
  --spec-type draft-dflash --spec-draft-n-max 7 --ctx-size 4096 -ngl 999 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
  --temp 1.0 --top-p 0.95 --top-k 20 \
  --host 127.0.0.1 --port "$PORT" --api-key-file /etc/llama-server/api-keys \
  > /tmp/rt-debug-server.log 2>&1 &
pid=$!
sleep 5
echo "--- env actually seen by the server process ---"
xargs -0 -n1 < "/proc/$pid/environ" 2>/dev/null | grep LLAMA_ROUND_TIMING || echo "(LLAMA_ROUND_TIMING NOT in environ)"

n=0; ok=0
while [ "$n" -lt 600 ]; do
  if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
  kill -0 "$pid" 2>/dev/null || break
  sleep 10; n=$((n+10))
done
echo "health ok=$ok after ${n}s"

if [ "$ok" = "1" ]; then
  echo "--- one short request (n_predict 256) ---"
  P="Solve step by step: if 3x + 7 = 22, what is x? Explain how you checked it."
  curl -s -m 600 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"n_predict\":256,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"seed\":42,\"cache_prompt\":false}" \
    "http://127.0.0.1:$PORT/v1/chat/completions" > /dev/null 2>&1
  grep -E "eval time =|draft acceptance" /tmp/rt-debug-server.log | tail -2
fi
kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null

echo ""
echo "=== self-report line (proves whether the env was seen) ==="
grep -a "LLAMA_ROUND_TIMING" /tmp/rt-debug-server.log | head -3
echo ""
echo "=== round timing lines ==="
grep -a llama_round_timing /tmp/rt-debug-server.log | head -12
echo ""
echo "=== do other libllama INFO lines appear in this log? ==="
grep -ac -e "capping draft context ubatch" /tmp/rt-debug-server.log
echo RTDEBUG_DONE
