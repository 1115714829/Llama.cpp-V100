#!/bin/bash
# Build the MMVQ-wide variant and run the operator-level comparison (fast, no model load).
set -u

SRC=/root/llm/test/v100-opt/llama.cpp
LAB=/mnt/3.84t/v100-opt/kernel-lab
BIN=$SRC/build-instr/bin
LOG=$LAB/mmvq-wide.log
MARK=$LAB/mmvq-wide.done
SRCDIR=$SRC/ggml/src/ggml-cuda

rm -f "$MARK"
exec >"$LOG" 2>&1
echo "=== START $(date -Is)"
export CUDA_VISIBLE_DEVICES=0

fail() {
    echo "ABORT_$1"
    rm -f /tmp/LLAMA_BUILD_LOCK
    touch "$MARK"
    exit 1
}
pgrep -f 'llama-benc[h] -m' >/dev/null && fail BUSY_BENCH
pgrep -f 'llama-serve[r] --model' >/dev/null && fail BUSY_SERVER
pgrep -f 'cmake --buil[d]' >/dev/null && fail BUSY_CMAKE
[ -e /tmp/LLAMA_BUILD_LOCK ] && fail BUILD_LOCK
touch /tmp/LLAMA_BUILD_LOCK

echo "source state:"
md5sum "$SRCDIR/mmvq.cu" "$SRCDIR/vecdotq.cuh"
grep -c -e GGML_CUDA_MMVQ_WIDE "$SRCDIR/mmvq.cu"
grep -c -e vec_dot_q8_0_q8_1_wide "$SRCDIR/vecdotq.cuh"

echo "=== BUILD $(date -Is)"
cmake --build "$SRC/build-instr" --target test-backend-ops -j128 >"$LAB/build-wide.log" 2>&1
RC=$?
echo "BUILD_RC=$RC errors=$(grep -c -e 'error' "$LAB/build-wide.log")"
grep -e 'Building CUDA' "$LAB/build-wide.log" | head -5
if [ $RC -ne 0 ]; then
    tail -40 "$LAB/build-wide.log"
    fail "BUILD"
fi
# the marker string must be inside the library that gets loaded, not just in the source
STR=$(strings "$BIN/libggml-cuda.so.0.24.0" | grep -c -e GGML_CUDA_MMVQ_WIDE)
echo "LIB_MARKER_WIDE=$STR"
md5sum "$BIN/libggml-cuda.so.0.24.0"

perf() {   # $1 = tag
    echo "=== PERF $1 $(date -Is)"
    "$BIN/test-backend-ops" perf -o MUL_MAT -b CUDA0 -p 'q8_0.*4096.*14336' 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'us/run|MUL_MAT' | sed 's/^[[:space:]]*//'
}

perf base
echo "=== with GGML_CUDA_MMVQ_WIDE=1"
GGML_CUDA_MMVQ_WIDE=1 perf wide

echo "=== CORRECTNESS with wide on (vs CPU reference) $(date -Is)"
GGML_CUDA_MMVQ_WIDE=1 "$BIN/test-backend-ops" test -o MUL_MAT -b CUDA0 -p 'q8_0' 2>&1 \
    | sed 's/\x1b\[[0-9;]*m//g' | tail -12

echo "=== CORRECTNESS with wide off $(date -Is)"
"$BIN/test-backend-ops" test -o MUL_MAT -b CUDA0 -p 'q8_0' 2>&1 \
    | sed 's/\x1b\[[0-9;]*m//g' | tail -6

rm -f /tmp/LLAMA_BUILD_LOCK
echo "=== DONE $(date -Is)"
touch "$MARK"
