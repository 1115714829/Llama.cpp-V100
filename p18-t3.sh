#!/bin/bash
# t3: build mmq-config-volta.cuh (I=160 for Q2_K/Q4_K) and measure it in the MMQ window.
# Control: -ub 4 -> MMVQ, so I must have no effect there.
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
G="$SRC/ggml/src/ggml-cuda"
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
LOG=/tmp/t3.log
: > "$LOG"

echo "=== [1] normalize + confirm wiring ===" | tee -a "$LOG"
sed -i 's/\r$//' "$G/mmq.cuh" "$G/mmq-config-volta.cuh"
grep -n "mmq-config-volta" "$G/mmq.cuh" | tee -a "$LOG"
grep -n "MMQ_VOLTA_I_K" "$G/mmq-config-volta.cuh" | tee -a "$LOG"
echo ""

echo "=== [2] build ===" | tee -a "$LOG"
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j8 --target llama-server llama-bench" > /tmp/build-t3.log 2>&1 &
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do
  n=$((n+15)); [ "$n" -ge 430 ] && { echo "still building" | tee -a "$LOG"; break; }
  sleep 15
done
echo "--- errors? ---" | tee -a "$LOG"
grep -iE "error:|FAILED" /tmp/build-t3.log | head -8 | tee -a "$LOG" || true
tail -2 /tmp/build-t3.log | tee -a "$LOG"
echo ""

echo "=== [3] snapshot libdir-volta3 ===" | tee -a "$LOG"
rm -rf /root/libdir-volta3 && mkdir -p /root/libdir-volta3
cp -a "$SRC/build/bin/." /root/libdir-volta3/
md5sum /root/libdir-volta2/libggml-cuda.so.0.24.0 /root/libdir-volta3/libggml-cuda.so.0.24.0 | tee -a "$LOG"
echo ""

echo "=== [4] I=128 (volta2) vs I=160 (volta3) across the MMQ window (Q2_K_XL, 1 GPU) ===" | tee -a "$LOG"
export CUDA_VISIBLE_DEVICES=0
for UB in 4 16 24 32 40 48; do
  for v in volta2 volta3; do
    ROW=$(LD_LIBRARY_PATH=/root/libdir-$v /root/libdir-$v/llama-bench \
      -m "$M" -p 512 -n 32 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 -b "$UB" -ub "$UB" -r 3 2>&1 | grep -F "pp512")
    printf "UB=%-3s %-7s %s\n" "$UB" "$v" "$ROW" | tee -a "$LOG"
  done
  echo "" | tee -a "$LOG"
done
echo T3_DONE
