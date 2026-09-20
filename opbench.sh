#!/bin/bash
# opbench.sh -- op-level cost split for the M=8 verify round.
# Why: ncu cannot profile this workload on this box (ncu-tgt.sh died with
# "Failed to profile scale_f32 / No kernels were profiled" after
# "Backing up device memory in system memory").  Use the in-tree op
# microbenchmark instead.  Step 1 of Goal 8dbd30ea.
set -u
cd /root/llm/test/v100-opt/llama.cpp
export LD_LIBRARY_PATH=/root/libdir-nccl
export CUDA_VISIBLE_DEVICES=0
TBO=./build-nccl/bin/test-backend-ops

echo "=== which ==="
ls -l "$TBO"

echo "=== GATED_DELTA_NET perf ==="
"$TBO" perf -o GATED_DELTA_NET -b CUDA0 > /tmp/opbench-gdn.txt 2>&1
echo "gdn lines: $(wc -l < /tmp/opbench-gdn.txt)"

echo "=== FLASH_ATTN_EXT perf ==="
"$TBO" perf -o FLASH_ATTN_EXT -b CUDA0 > /tmp/opbench-fa.txt 2>&1
echo "fa lines: $(wc -l < /tmp/opbench-fa.txt)"

echo "=== MUL_MAT q8_0 perf (for the incumbent reference) ==="
"$TBO" perf -o MUL_MAT -b CUDA0 > /tmp/opbench-mm.txt 2>&1
echo "mm lines: $(wc -l < /tmp/opbench-mm.txt)"

echo OPBENCH_DONE
