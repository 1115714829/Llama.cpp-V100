#!/bin/bash
# Install the mmvq Volta-crossover change into the server source tree and rebuild.
# Only files that changed: ggml/src/ggml-cuda/mmvq.cu and mmvq.cuh (scp'd from Windows).
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
G="$SRC/ggml/src/ggml-cuda"

echo "=== [1] backup current (C4-only) source files ==="
cp "$G/mmvq.cu"  /root/mmvq-c4-prev.cu
cp "$G/mmvq.cuh" /root/mmvq-c4-prev.cuh
ls -la /root/mmvq-c4-prev.cu /root/mmvq-c4-prev.cuh
echo ""

echo "=== [2] normalize CRLF (files came from a Windows checkout) ==="
sed -i 's/\r$//' "$G/mmvq.cu"
sed -i 's/\r$//' "$G/mmvq.cuh"
echo "mmvq.cu  VOLTA refs: $(grep -c 'GGML_CUDA_CC_VOLTA' "$G/mmvq.cu")"
echo "mmvq.cuh VOLTA refs: $(grep -c 'MMVQ_VOLTA_MAX_BATCH_SIZE_K' "$G/mmvq.cuh")"
echo ""

echo "=== [3] DIFF vs previous (must be ONLY the new Volta block + define + include) ==="
diff -u /root/mmvq-c4-prev.cu "$G/mmvq.cu" || true
echo "--- mmvq.cuh ---"
diff -u /root/mmvq-c4-prev.cuh "$G/mmvq.cuh" || true
echo ""

echo "=== [4] launch incremental build ==="
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j8 --target llama-server llama-bench" > /tmp/build-volta.log 2>&1 &
echo "launched PID=$! log=/tmp/build-volta.log"
sleep 12
tail -4 /tmp/build-volta.log
echo BUILD_VOLTA_LAUNCHED
