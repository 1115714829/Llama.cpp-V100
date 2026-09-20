#!/bin/bash
# Phase 0.3 (complete): real config round-cost measurement - 3 cards, Q8_0 + DFlash2 n=7.
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
L=/root/libdir-rt
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
LOG=/tmp/p46.log
: > "$LOG"
exec >> "$LOG" 2>&1

echo "=== rebuild ==="
sed -i 's/\r$//' "$SRC/src/llama-context.cpp"
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j82 --target llama-server" > /tmp/build-p46.log 2>&1 &
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do
  sleep 10; n=$((n+10)); [ "$n" -gt 700 ] && { echo "build timeout"; break; }
done
echo -n "build errors: "; grep -icE "error:|FAILED" /tmp/build-p46.log
echo -n "instrument marker in lib: "; grep -c -a "perf: rounds=" "$SRC/build/bin/libllama.so.0.4.1"

echo ""
echo "=== snapshot + REAL config: 3 cards (0/1/2), Q8_0 target + DFlash2 n=7, 512 tokens ==="
rm -rf "$L" && mkdir -p "$L" && cp -a "$SRC/build/bin/." "$L/"
export LD_LIBRARY_PATH="$L"
export CUDA_VISIBLE_DEVICES=0,1,2
export LLAMA_ROUND_TIMING=1
KEY=$(head -1 /etc/llama-server/api-keys)

nohup "$L/llama-server" --model "$M" --model-draft "$D" --split-mode tensor --tensor-split 1,1,1 \
  --spec-type draft-dflash --spec-draft-n-max 7 --ctx-size 8192 -ngl 999 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
  --temp 1.0 --top-p 0.95 --top-k 20 \
  --host 127.0.0.1 --port 8116 --api-key-file /etc/llama-server/api-keys > /tmp/p46-server.log 2>&1 &
pid=$!
n=0
while [ "$n" -lt 900 ]; do
  curl -s -m 5 "http://127.0.0.1:8116/health" 2>/dev/null | grep -q '"ok"' && break
  sleep 15; n=$((n+15))
done
echo "health after ${n}s"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

P="Explain, step by step, how to compute the sum of the first n integers and prove the formula by induction."
curl -s -m 1800 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"n_predict\":512,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"seed\":42,\"cache_prompt\":false}" \
  "http://127.0.0.1:8116/v1/chat/completions" > /dev/null 2>&1
kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null

echo ""
echo "=== throughput / acceptance ==="
grep -a -E "eval time =|draft acceptance" /tmp/p46-server.log | tail -4
echo ""
echo "=== [RT] perf lines (final windows) ==="
grep -a -F "[RT] perf:" /tmp/p46-server.log | tail -6
echo ""
echo "=== graphs reused (perf) ==="
grep -a "graphs reused" /tmp/p46-server.log | tail -2
echo P46_DONE
