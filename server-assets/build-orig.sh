#!/bin/bash
# Swap in the original (pre-C4) mmvq.cu + incrementally rebuild llama-bench (the ORIGINAL variant).
set -uo pipefail
cd /root/llm/test/v100-opt/llama.cpp
MMVQ=ggml/src/ggml-cuda/mmvq.cu
echo "=== backup current (C4) mmvq.cu ==="
cp "$MMVQ" /root/mmvq-c4-backup.cu
echo "C4 backup VOLTA refs (expect >0): $(grep -c MMVQ_PARAMETERS_VOLTA /root/mmvq-c4-backup.cu)"
echo "=== replace with original (mmvq-orig.cu) ==="
cp /root/mmvq-orig.cu "$MMVQ"
sed -i 's/\r$//' "$MMVQ"
echo "orig now VOLTA refs (expect 0): $(grep -c MMVQ_PARAMETERS_VOLTA "$MMVQ")"
echo "=== free + GPU state before build ==="
free -g | head -2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "=== build (incremental, target llama-bench) ==="
source /opt/rh/gcc-toolset-12/enable
cmake --build build --config Release -j16 --target llama-bench 2>&1 | tail -18
echo "=== verify ==="
ls -la build/bin/llama-bench
echo BUILD_ORIG_DONE
