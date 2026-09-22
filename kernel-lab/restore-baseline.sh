#!/bin/bash
# Restore the baseline: revert the MMVQ-WIDE experiment source and rebuild so libggml-cuda.so
# returns to the recorded baseline md5 (0fb36620...), which the next session must be able to assume.
set -u
SRC=/root/llm/test/v100-opt/llama.cpp
CD=$SRC/ggml/src/ggml-cuda
LAB=/mnt/3.84t/v100-opt/kernel-lab
BIN=$SRC/build-instr/bin
LOG=$LAB/restore-baseline.log
MARK=$LAB/restore-baseline.done

rm -f "$MARK"
exec >"$LOG" 2>&1
echo "=== START $(date -Is)"
[ -e /tmp/LLAMA_BUILD_LOCK ] && { echo ABORT_BUILD_LOCK; touch "$MARK"; exit 1; }
pgrep -f 'llama-serve[r] --model' >/dev/null && { echo ABORT_BUSY; touch "$MARK"; exit 1; }
touch /tmp/LLAMA_BUILD_LOCK

for f in mmvq.cu vecdotq.cuh; do
    if [ -f "$CD/$f.orig-wide" ]; then
        cp -f "$CD/$f.orig-wide" "$CD/$f"
        echo "restored $f: $(md5sum "$CD/$f" | cut -d' ' -f1)"
    else
        echo "WARN no .orig-wide for $f"
    fi
done
grep -c -e GGML_CUDA_MMVQ_WIDE "$CD/mmvq.cu"

echo "=== BUILD $(date -Is)"
cmake --build "$SRC/build-instr" --target test-backend-ops -j128 >"$LAB/build-restore3.log" 2>&1
echo "BUILD_RC=$? errors=$(grep -c -e 'error' "$LAB/build-restore3.log")"
tail -3 "$LAB/build-restore3.log"
echo "lib md5 (expect 0fb366208abb9e83293d476b35ab10db):"
md5sum "$BIN/libggml-cuda.so.0.24.0"
echo "wide marker count (expect 0): $(strings "$BIN/libggml-cuda.so.0.24.0" | grep -c -e GGML_CUDA_MMVQ_WIDE)"

rm -f /tmp/LLAMA_BUILD_LOCK
echo "=== DONE $(date -Is)"
touch "$MARK"
