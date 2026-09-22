#!/bin/bash
# Does raising MMVQ rows_per_block for Volta (ncols 5..8) unlock the n=8 bandwidth?
# Op level only: test-backend-ops perf, single card, q8_0 m=4096 k=14336.
# Control = the build that is already installed; then rpb = 4, 8, 16; then restore.
set -u

SRC=/mnt/3.84t/v100-opt/llama.cpp
LAB=/mnt/3.84t/v100-opt/kernel-lab
MMVQ=$SRC/ggml/src/ggml-cuda/mmvq.cu
BIN=$SRC/build-instr/bin
LOG=$LAB/rpb-experiment.log
MARK=$LAB/rpb-experiment.done

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
echo "LOCKED /tmp/LLAMA_BUILD_LOCK"

MD5_ORIG=$(md5sum "$MMVQ" | awk '{print $1}')
echo "MD5_ORIG=$MD5_ORIG"
if [ ! -f "$MMVQ.orig-rpb" ]; then
    cp -f "$MMVQ" "$MMVQ.orig-rpb"
fi

perf() {
    echo "=== PERF $1 $(date -Is)"
    $BIN/test-backend-ops perf -o MUL_MAT -b CUDA0 -p 'q8_0.*4096.*14336' 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'us/run' \
        | sed 's/^[[:space:]]*//'
}

build() {
    echo "=== BUILD $1 $(date -Is)"
    cmake --build "$SRC/build-instr" -j128 >"$LAB/build-rpb$1.log" 2>&1
    local rc=$?
    echo "BUILD_RC=$rc errors=$(grep -c -e 'error' "$LAB/build-rpb$1.log")"
    grep -E 'Built target ggml-cuda' "$LAB/build-rpb$1.log" | tail -1
    if [ $rc -ne 0 ]; then
        tail -40 "$LAB/build-rpb$1.log"
        fail "BUILD_$1"
    fi
}

perf control

for V in 4 8 16; do
    python3 "$LAB/rpb-patch.py" "$MMVQ" $V || fail "PATCH_$V"
    build "$V"
    perf "rpb$V"
    echo "=== CORRECTNESS rpb$V $(date -Is)"
    $BIN/test-backend-ops test -o MUL_MAT -b CUDA0 -p 'q8_0.*4096.*14336' 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g' | tail -12
done

python3 "$LAB/rpb-patch.py" "$MMVQ" 0 || fail UNPATCH
if cmp -s "$MMVQ" "$MMVQ.orig-rpb"; then echo "RESTORE_MATCH=yes"; else echo "RESTORE_MATCH=NO"; fi
echo "MD5_AFTER=$(md5sum "$MMVQ" | awk '{print $1}')"
build restore
perf control_restored

rm -f /tmp/LLAMA_BUILD_LOCK
echo "=== DONE $(date -Is)"
touch "$MARK"
