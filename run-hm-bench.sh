#!/bin/bash
# run-hm-bench.sh -- build and run the SM70 HMMA Q8_0 verify-shape GEMM microbenchmark.
set -u
NVCC=/usr/local/cuda-12.4/bin/nvcc
echo "=== build ==="
"$NVCC" -arch=sm_70 -O3 -std=c++17 --ptxas-options=-v -o /root/hm_q8_bench /root/hm_q8_bench.cu 2>&1 | grep -E "error|Error|registers|spill|Function properties|Used" | head -20
ls -l /root/hm_q8_bench 2>&1
echo "=== correctness K=64 ==="
CUDA_VISIBLE_DEVICES=0 /root/hm_q8_bench 64 2>&1 | tail -14
echo "=== correctness K=5120 ==="
CUDA_VISIBLE_DEVICES=0 /root/hm_q8_bench 5120 2>&1 | tail -12
echo HM_BENCH_DONE
