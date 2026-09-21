#!/bin/bash
# Wait for the volta build, snapshot it, verify the block, then sweep the K-quant crossover T.
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
G="$SRC/ggml/src/ggml-cuda"
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
LOG=/tmp/volta-sweep.log
: > "$LOG"

echo "=== [1] wait for build (max 480s) ===" | tee -a "$LOG"
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do
  n=$((n+15)); [ "$n" -ge 480 ] && { echo "still building" | tee -a "$LOG"; break; }
  sleep 15
done
grep -iE "error:|FAILED" /tmp/build-volta.log | head -5 | tee -a "$LOG" || true
echo "(no error lines above = clean)" | tee -a "$LOG"
tail -2 /tmp/build-volta.log | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== [2] verify the change is in the source ===" | tee -a "$LOG"
grep -n "GGML_CUDA_CC_VOLTA" "$G/mmvq.cu" | tee -a "$LOG"
grep -n "MMVQ_VOLTA_TUNE_MAX" "$G/mmvq.cu" | tee -a "$LOG"
grep -n "MMVQ_VOLTA_MAX_BATCH_SIZE_K" "$G/mmvq.cuh" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== [3] snapshot libdir-volta ===" | tee -a "$LOG"
rm -rf /root/libdir-volta && mkdir -p /root/libdir-volta
cp -a "$SRC/build/bin/." /root/libdir-volta/
echo "files: $(ls /root/libdir-volta | wc -l)" | tee -a "$LOG"
md5sum /root/libdir-pristine/libggml-cuda.so.0.24.0 /root/libdir-c4/libggml-cuda.so.0.24.0 /root/libdir-volta/libggml-cuda.so.0.24.0 | tee -a "$LOG"
test -x /root/libdir-volta/llama-bench && echo "bench OK" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== [4] crossover sweep: T = max ne11 still using MMVQ for K-quants ===" | tee -a "$LOG"
echo "T=8 is upstream default. Smaller T => MMQ used for ne11 > T. UB sets ne11 during prefill." | tee -a "$LOG"
export CUDA_VISIBLE_DEVICES=2
for UB in 4 6 8 10 12 16 24 32; do
  for T in 8 6 4 2 1; do
    OUT=$(MMVQ_VOLTA_TUNE_MAX=$T LD_LIBRARY_PATH=/root/libdir-volta /root/libdir-volta/llama-bench \
      -m "$M" -p 512 -n 32 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 -b "$UB" -ub "$UB" -r 3 2>&1)
    ROW=$(echo "$OUT" | grep -F "pp512")
    printf "UB=%-3s T=%-2s %s\n" "$UB" "$T" "$ROW" | tee -a "$LOG"
  done
  echo "" | tee -a "$LOG"
done
echo VOLTA_SWEEP_DONE
