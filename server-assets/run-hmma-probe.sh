#!/bin/bash
# run-hmma-probe.sh -- build and run the m8n8k4 layout probe on the V100.
set -u
NVCC=/usr/local/cuda-12.4/bin/nvcc
cd /root || exit 1
echo "=== nvcc version ==="
"$NVCC" --version | tail -2
echo "=== build (arch=sm_70) ==="
"$NVCC" -arch=sm_70 -O2 -std=c++17 -o /root/hmma_probe /root/hmma_probe.cu 2>&1 | head -20
ls -l /root/hmma_probe 2>&1
echo "=== run ==="
CUDA_VISIBLE_DEVICES=0 /root/hmma_probe 2>&1
echo PROBE_BUILD_RUN_DONE
