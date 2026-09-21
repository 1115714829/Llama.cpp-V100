#!/bin/bash
# Phase 0.3 baseline: official Qwen3.8-27B Q8_0 target + official DFlash2 draft, 3 cards in one NUMA
# node (0/1/2), short context, official sampling, WITH per-round cost instrumentation.
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
L=/root/libdir-rt
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
PORT=8110
LOG=/tmp/base-q8.log
: > "$LOG"
exec >> "$LOG" 2>&1

echo "=== [1] snapshot the instrumented build -> $L ==="
rm -rf "$L" && mkdir -p "$L" && cp -a "$SRC/build/bin/." "$L/"
echo -n "instrument present in libllama.so: "; grep -c -a llama_round_timing "$L/libllama.so.0.4.1"
md5sum "$L/libllama.so.0.4.1" "$L/libggml-cuda.so.0.24.0"
echo -n "target size: "; stat -c %s "$M"

echo ""
echo "=== [2] launch server: Q8_0 + DFlash2 n=7, TP3 on GPU 0/1/2, ctx 8192 ==="
echo "server --version:"
"$L/llama-server" --version 2>&1 | head -3

KEY=$(head -1 /etc/llama-server/api-keys)
export CUDA_VISIBLE_DEVICES=0,1,2
export LD_LIBRARY_PATH="$L"
export LLAMA_ROUND_TIMING=1

nohup "$L/llama-server" --model "$M" --model-draft "$D" --split-mode tensor --tensor-split 1,1,1 \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --ctx-size 8192 -ngl 999 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --host 127.0.0.1 --port "$PORT" --api-key-file /etc/llama-server/api-keys \
  > /tmp/base-q8-server.log 2>&1 &
pid=$!

echo "waiting for health (Q8_0 is 29 GB over NFS; this is network-bound)"
n=0; ok=0
while [ "$n" -lt 2400 ]; do
  if curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
  kill -0 "$pid" 2>/dev/null || break
  sleep 15; n=$((n+15))
  if [ $((n % 300)) -eq 0 ]; then echo "  ...loading ${n}s"; fi
done
echo "health ok=$ok after ${n}s"

if [ "$ok" != "1" ]; then
  echo "--- failure evidence ---"
  grep -aE "GGML_ASSERT|error|failed to load|abort|not supported" /tmp/base-q8-server.log | tail -6
  echo BASEQ8_DONE
  exit 0
fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

P1="A train leaves Station A at 60 km/h. Another leaves Station B, 300 km away, at 40 km/h toward A. They start at the same time. Work through this step by step and explain when and where they meet."
P2="Explain, step by step, how to compute the sum of the first n integers and prove the formula by induction."
P3="Solve step by step: if 3x + 7 = 22, what is x? Then explain in detail how you checked your answer."

i=0
for p in "$P1" "$P2" "$P3"; do
  i=$((i+1))
  echo ""
  echo "########## request $i (n_predict 512, temp 1.0 / top_p 0.95 / top_k 20, seed 42) ##########"
  curl -s -m 1800 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"n_predict\":512,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"seed\":42,\"cache_prompt\":false}" \
    "http://127.0.0.1:$PORT/v1/chat/completions" > /dev/null 2>&1
  grep -E "eval time =" /tmp/base-q8-server.log | tail -1
  grep -E "draft acceptance" /tmp/base-q8-server.log | tail -1
done

echo ""
echo "=== per-round cost instrumentation (LLAMA_ROUND_TIMING=1) ==="
grep -a llama_round_timing /tmp/base-q8-server.log | tail -40
echo ""
echo "=== graphs reused (perf summary) ==="
grep -a "graphs reused" /tmp/base-q8-server.log | tail -4
echo ""
echo "=== draft/selector evidence ==="
grep -aE "capping draft context ubatch|DFlash2|block_size=" /tmp/base-q8-server.log | head -6

kill "$pid" 2>/dev/null; sleep 4; kill -9 "$pid" 2>/dev/null
echo BASEQ8_DONE
