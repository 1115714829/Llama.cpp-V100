#!/bin/bash
# Phase P1: stop the production inference services (user-authorized), then produce a
# PRISTINE b11053 binary pair by swapping mmvq.cu in the SAME build dir (identical flags).
set -uo pipefail

SRC=/root/llm/test/v100-opt/llama.cpp
MMVQ=$SRC/ggml/src/ggml-cuda/mmvq.cu

echo "############ [1] state BEFORE (for restore) ############"
for u in vllm-1cat llmscope new-api llama-server llama-server-test llama-server-c4; do
  printf "%-20s %s\n" "$u" "$(systemctl is-active "$u" 2>/dev/null)"
done
echo "--- GPU before ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "--- free -g before ---"
free -g | head -2
echo ""

echo "############ [2] STOP production inference (authorized 2026-09-20) ############"
echo ">>> restore later with: systemctl start vllm-1cat llmscope"
systemctl stop vllm-1cat
systemctl stop llmscope
sleep 5
for u in vllm-1cat llmscope new-api; do
  printf "%-20s %s\n" "$u" "$(systemctl is-active "$u" 2>/dev/null)"
done
echo "--- GPU after ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "--- free -g after ---"
free -g | head -2
echo ""

echo "############ [3] save the current C4 binaries aside ############"
cp -a "$SRC/build/bin/llama-server" /root/bin-b11053-c4-server
cp -a "$SRC/build/bin/llama-bench"  /root/bin-b11053-c4-bench
ls -la /root/bin-b11053-c4-server /root/bin-b11053-c4-bench
echo ""

echo "############ [4] swap in PRISTINE mmvq.cu ############"
cp "$MMVQ" /root/mmvq-c4-backup.cu
echo "saved C4 backup: MMVQ_PARAMETERS_VOLTA refs = $(grep -c MMVQ_PARAMETERS_VOLTA /root/mmvq-c4-backup.cu)"
cp /root/mmvq-orig.cu "$MMVQ"
sed -i 's/\r$//' "$MMVQ"
echo "source now: MMVQ_PARAMETERS_VOLTA refs = $(grep -c MMVQ_PARAMETERS_VOLTA "$MMVQ")  (expect 0)"
echo ""

echo "############ [5] launch INCREMENTAL build of pristine llama-server+llama-bench ############"
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j8 --target llama-server llama-bench" > /tmp/build-pristine.log 2>&1 &
echo "launched PID=$!  log=/tmp/build-pristine.log"
sleep 8
echo "--- first lines of build log ---"
head -5 /tmp/build-pristine.log 2>/dev/null
echo BUILD_PRISTINE_LAUNCHED
