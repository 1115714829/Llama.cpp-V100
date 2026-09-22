#!/bin/bash
# Where does PREFILL time actually go? Per-op table with real shapes (R229 method, but for the
# prefill batch instead of the verify batch): export the graph ops at -ub 2048 and time them.
set -u

BIN=/root/llm/test/v100-opt/llama.cpp/build-instr/bin
LAB=/mnt/3.84t/v100-opt/kernel-lab
M=/mnt/3.84t/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
LOG=$LAB/prefill-ops.log
MARK=$LAB/prefill-ops.done

rm -f "$MARK"
exec >"$LOG" 2>&1
echo "=== START $(date -Is)"
export CUDA_VISIBLE_DEVICES=0

fail() { echo "ABORT_$1"; touch "$MARK"; exit 1; }
pgrep -f 'llama-benc[h] -m' >/dev/null && fail BUSY_BENCH
[ -e /tmp/LLAMA_BUILD_LOCK ] && fail BUILD_LOCK

for UB in 2048 512; do
    echo "=== export -ub $UB $(date -Is)"
    "$BIN/test-export-graph-ops" -m "$M" -o "/tmp/ops-ub$UB.txt" -ub "$UB" 2>&1 | tail -3
    wc -l "/tmp/ops-ub$UB.txt"
    echo "--- distinct op names ---"
    sed 's/(.*//' "/tmp/ops-ub$UB.txt" | sort | uniq -c | sort -rn | head -20
    echo "=== perf -ub $UB $(date -Is)"
    "$BIN/test-backend-ops" perf --test-file "/tmp/ops-ub$UB.txt" -b CUDA0 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'us/run|ms/run' > "/tmp/perf-ub$UB.txt"
    wc -l "/tmp/perf-ub$UB.txt"
done

echo "=== top 25 ops by time (-ub 2048) ==="
sed 's/^[[:space:]]*//' /tmp/perf-ub2048.txt | paste - - 2>/dev/null | head -3
echo "--- full list with the matching case names ---"
grep -B0 -A0 -e 'us/run' /tmp/perf-ub2048.txt | sed 's/^[[:space:]]*//' | sort -t' ' -k1 | tail -25

echo "=== DONE $(date -Is)"
touch "$MARK"
