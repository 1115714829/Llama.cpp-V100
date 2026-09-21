#!/bin/bash
# Phase 1a: rebuild with the diagnostic in handle_per_row, then reproduce with 1 card + --split-mode tensor
# (the assert fires even with a single visible device, so we do not need the 3-card production model).
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
G="$SRC/ggml/src/ggml-backend-meta.cpp"
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
LOG=/tmp/diag1a.log
: > "$LOG"

sed -i 's/\r$//' "$G"
echo "diagnostic present? $(grep -c 'per-row op does not support' "$G")" | tee -a "$LOG"

echo "=== build ===" | tee -a "$LOG"
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j8 --target llama-server" > /tmp/build-diag1a.log 2>&1 &
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do
  n=$((n+15)); [ "$n" -ge 430 ] && { echo "still building" | tee -a "$LOG"; break; }
  sleep 15
done
grep -iE "error:|FAILED" /tmp/build-diag1a.log | head -5 | tee -a "$LOG" || true
tail -2 /tmp/build-diag1a.log | tee -a "$LOG"

rm -rf /root/libdir-diag && mkdir -p /root/libdir-diag
cp -a "$SRC/build/bin/." /root/libdir-diag/ 2>/dev/null

echo "" | tee -a "$LOG"
echo "=== reproduce: 1 card, --split-mode tensor, draft-dflash ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/root/libdir-diag timeout 240 \
  /root/libdir-diag/llama-server --model "$M" --model-draft "$D" --spec-type draft-dflash \
    --spec-draft-n-max 7 --ctx-size 2048 -ngl 999 --split-mode tensor \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --host 127.0.0.1 --port 8103 --api-key-file /etc/llama-server/api-keys \
  > /tmp/diag1a-server.log 2>&1 &
pid=$!
n=0; ok=0
while [ "$n" -lt 240 ]; do
  if curl -s -m 5 "http://127.0.0.1:8103/health" 2>/dev/null | grep -q '"ok"'; then ok=1; break; fi
  kill -0 "$pid" 2>/dev/null || break
  sleep 5; n=$((n+5))
done
echo "health ok=$ok after ${n}s" | tee -a "$LOG"
kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null

echo "--- diagnostic / crash lines ---" | tee -a "$LOG"
grep -aE "per-row op does not support|GGML_ASSERT|META_PER_ROW|abort" /tmp/diag1a-server.log | head -8 | tee -a "$LOG"
echo DIAG1A_DONE
