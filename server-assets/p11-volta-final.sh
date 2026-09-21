#!/bin/bash
# Rebuild with the final hard-coded Volta crossover (4) and verify it took effect + no regression.
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
G="$SRC/ggml/src/ggml-cuda"
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
LOG=/tmp/volta-final.log
: > "$LOG"

echo "=== [1] normalize + confirm scaffold is gone ===" | tee -a "$LOG"
sed -i 's/\r$//' "$G/mmvq.cu" "$G/mmvq.cuh"
echo "scaffold refs (expect 0): $(grep -c 'MMVQ_VOLTA_TUNE_MAX' "$G/mmvq.cu")" | tee -a "$LOG"
echo "getenv in mmvq.cu (expect 0): $(grep -c 'getenv' "$G/mmvq.cu")" | tee -a "$LOG"
echo "Volta define (expect 4): $(grep 'MMVQ_VOLTA_MAX_BATCH_SIZE_K' "$G/mmvq.cuh")" | tee -a "$LOG"
echo "Volta block line: $(grep -n 'cc == GGML_CUDA_CC_VOLTA' "$G/mmvq.cu" | head -1)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== [2] rebuild (incremental) ===" | tee -a "$LOG"
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j8 --target llama-server llama-bench" > /tmp/build-volta2.log 2>&1 &
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do
  n=$((n+15)); [ "$n" -ge 400 ] && { echo "still building" | tee -a "$LOG"; break; }
  sleep 15
done
grep -iE "error:|FAILED" /tmp/build-volta2.log | head -5 | tee -a "$LOG" || true
tail -2 /tmp/build-volta2.log | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "=== [3] snapshot libdir-volta2 ===" | tee -a "$LOG"
rm -rf /root/libdir-volta2 && mkdir -p /root/libdir-volta2
cp -a "$SRC/build/bin/." /root/libdir-volta2/
md5sum /root/libdir-pristine/libggml-cuda.so.0.24.0 /root/libdir-c4/libggml-cuda.so.0.24.0 /root/libdir-volta2/libggml-cuda.so.0.24.0 | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "=== [4] VERIFY: hard-coded 4 must reproduce the T=4 numbers ===" | tee -a "$LOG"
export CUDA_VISIBLE_DEVICES=2
for UB in 4 8; do
  ROW=$(LD_LIBRARY_PATH=/root/libdir-volta2 /root/libdir-volta2/llama-bench \
    -m "$M" -p 512 -n 32 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 -b "$UB" -ub "$UB" -r 3 2>&1 | grep -F "pp512")
  printf "no-env UB=%-3s %s\n" "$UB" "$ROW" | tee -a "$LOG"
done
echo "expected: UB=4 ~96.4 (MMVQ); UB=8 ~137.4 (MMQ)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "=== [5] REGRESSION check in the default regime (ne11=512 prefill, ne11=1 decode) ===" | tee -a "$LOG"
for v in pristine volta2; do
  ROWS=$(LD_LIBRARY_PATH=/root/libdir-$v /root/libdir-$v/llama-bench \
    -m "$M" -p 512 -n 128 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 -r 3 2>&1 | grep -E "pp512|tg128")
  echo "--- $v ---" | tee -a "$LOG"
  echo "$ROWS" | tee -a "$LOG"
done
echo VOLTA_FINAL_DONE
