#!/bin/bash
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
echo "=== restore C4 mmvq.cu (from backup) ==="
cp /root/mmvq-c4-backup.cu "$SRC/ggml/src/ggml-cuda/mmvq.cu"
echo "C4 restored: $(grep -c MMVQ_PARAMETERS_VOLTA $SRC/ggml/src/ggml-cuda/mmvq.cu) VOLTA refs"
echo "=== rebuild llama-server (cmake incremental, background) ==="
cd "$SRC"
source /opt/rh/gcc-toolset-12/enable
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j8 --target llama-server 2>&1" > /tmp/build-c4.log 2>&1 &
echo "launched C4 rebuild PID=$! log=/tmp/build-c4.log"
echo BUILD_C4_LAUNCHED
