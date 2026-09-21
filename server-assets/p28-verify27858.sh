#!/bin/bash
# Phase 1a verification, step 1: snapshot the port build, then reproduce the exact crash case
# (1 card + --split-mode tensor + draft-dflash) that previously hit
# ggml-backend-meta.cpp handle_per_row: GGML_ASSERT(axis != SPLIT_AXIS_0).
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
BIN=/root/libdir-27858
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
LOG=/tmp/verify27858.log
: > "$LOG"
exec >> "$LOG" 2>&1

echo "=== [1] snapshot build/bin -> $BIN ==="
rm -rf "$BIN" && mkdir -p "$BIN"
cp -a "$SRC/build/bin/." "$BIN/"
echo "llama-server --version:"
LD_LIBRARY_PATH="$BIN" "$BIN/llama-server" --version 2>&1 | head -3
echo "md5 (must differ from libdir-final for libllama.so):"
md5sum "$BIN/libggml-cuda.so.0.24.0" "$BIN/libllama.so.0.4.1" 2>/dev/null
echo "for reference, previous variant:"
md5sum /root/libdir-final/libllama.so.0.4.1 /root/libdir-final/libggml-cuda.so.0.24.0 2>/dev/null

echo ""
echo "=== [2] reproduce: 1 card, --split-mode tensor, draft-dflash (previously SIGABRT) ==="
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH="$BIN" timeout 300 \
  "$BIN/llama-server" --model "$M" --model-draft "$D" --spec-type draft-dflash \
    --spec-draft-n-max 7 --ctx-size 2048 -ngl 999 --split-mode tensor \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --host 127.0.0.1 --port 8104 --api-key-file /etc/llama-server/api-keys \
  > /tmp/verify27858-server.log 2>&1 &
pid=$!
n=0; ok=0
while [ "$n" -lt 300 ]; do
  if curl -s -m 5 "http://127.0.0.1:8104/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
  kill -0 "$pid" 2>/dev/null || break
  sleep 5; n=$((n+5))
done
echo "health ok=$ok after ${n}s   (previously: process aborted during load)"

echo "--- does an actual completion work? ---"
if [ "$ok" = "1" ]; then
  curl -s -m 120 "http://127.0.0.1:8104/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Compute 17*23 and explain the steps."}],"max_tokens":96,"temperature":1.0,"top_p":0.95,"top_k":20,"seed":42,"cache_prompt":false}' \
    | head -c 1200
  echo ""
fi

kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null

echo ""
echo "--- crash / assert lines (expect NONE) ---"
grep -aE "GGML_ASSERT|abort|per-row op does not support|terminate called" /tmp/verify27858-server.log | head -8 || echo "(none)"

echo ""
echo "--- CPU-selector evidence (expect the tensor-parallel path to be taken) ---"
grep -aE "DFlash2|selector|block draft|capping draft context ubatch|split mode|tensor" /tmp/verify27858-server.log \
  | grep -aviE "^ggml_cuda|^load_tensors" | head -25

echo ""
echo "--- speculative stats ---"
grep -aiE "n_draft|n_accept|accept|mean len|mean_len|statistics" /tmp/verify27858-server.log | tail -20
echo VERIFY27858_DONE
