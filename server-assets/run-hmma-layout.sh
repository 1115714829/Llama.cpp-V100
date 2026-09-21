#!/bin/bash
# run-hmma-layout.sh -- build and run the one-hot fragment-layout probe.
set -u
NVCC=/usr/local/cuda-12.4/bin/nvcc
"$NVCC" -arch=sm_70 -O2 -std=c++17 -o /root/hmma_layout /root/hmma_layout.cu 2>&1 | head -20
ls -l /root/hmma_layout 2>&1
CUDA_VISIBLE_DEVICES=0 /root/hmma_layout > /tmp/hmma_layout.out 2>&1
echo "exit=$?"
echo "=== A-probe first 24 lines ==="
sed -n '1,28p' /tmp/hmma_layout.out
echo "=== total lines ==="
wc -l /tmp/hmma_layout.out
echo LAYOUT_RUN_DONE
