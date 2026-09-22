#!/bin/bash
# Where does the n=8 MUL_MAT time go, versus n=1? ncu on test-backend-ops (small tensors, so
# ncu's device memory save/restore works here -- it does not work on the 18-29 GB models).
set -u
BIN=/root/llm/test/v100-opt/llama.cpp/build-instr/bin
NCU=/usr/local/cuda-12.4/bin/ncu
export CUDA_VISIBLE_DEVICES=0
export GGML_CUDA_DISABLE_GRAPHS=1

METRICS=dram__bytes.sum,lts__t_sectors.sum,l1tex__t_sectors.sum,smsp__inst_executed.sum,gpu__time_duration.sum

for N in 1 8; do
    echo "===== n=$N ====="
    $NCU --metrics "$METRICS" --kernel-name regex:mul_mat_vec_q \
        --launch-skip 20 --launch-count 1 \
        "$BIN/test-backend-ops" perf -o MUL_MAT -b CUDA0 -p "q8_0.*m=4096,n=$N,k=14336" 2>&1 | tail -25
done

echo "===== n=8 stalls ====="
$NCU --section WarpStateStats --section SchedulerStats --kernel-name regex:mul_mat_vec_q \
    --launch-skip 20 --launch-count 1 \
    "$BIN/test-backend-ops" perf -o MUL_MAT -b CUDA0 -p "q8_0.*m=4096,n=8,k=14336" 2>&1 | tail -45

echo "===== n=1 stalls ====="
$NCU --section WarpStateStats --section SchedulerStats --kernel-name regex:mul_mat_vec_q \
    --launch-skip 20 --launch-count 1 \
    "$BIN/test-backend-ops" perf -o MUL_MAT -b CUDA0 -p "q8_0.*m=4096,n=1,k=14336" 2>&1 | tail -45
echo "=== DONE ==="
