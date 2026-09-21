#!/bin/bash
# microbench.sh -- quantify the M<=8 GEMM headroom: per-op MUL_MAT throughput by weight type.
# Q8_0 goes through MMVQ (re-dequantizes per column); F16 goes through the FP16 MMA path.
set -u
L=/root/libdir-nccl
T=$L/test-backend-ops
echo "=== binary ==="
ls -l "$T"
echo "=== help ==="
env LD_LIBRARY_PATH="$L" "$T" --help 2>&1 | head -40
echo "=== perf: MUL_MAT (background log) ==="
env LD_LIBRARY_PATH="$L" CUDA_VISIBLE_DEVICES=0 "$T" perf -o MUL_MAT -b CUDA0 > /tmp/mb-mulmat.log 2>&1
echo "exit=$?"
echo "=== interesting rows (small token counts / large k) ==="
grep -aE "MUL_MAT|type_a|Backend|dev" /tmp/mb-mulmat.log | head -5
grep -aE "Q8_0|F16" /tmp/mb-mulmat.log | head -60
echo MICROBENCH_DONE
