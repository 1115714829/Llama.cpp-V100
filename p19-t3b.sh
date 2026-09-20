#!/bin/bash
# t3b: test the opposite direction - SMALLER I (more blocks => more parallelism).
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
G="$SRC/ggml/src/ggml-cuda"
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
LOG=/tmp/t3b.log
: > "$LOG"

sed -i 's/\r$//' "$G/mmq.cuh" "$G/mmq-config-volta.cuh"
echo "=== constant now ===" | tee -a "$LOG"
grep -n "MMQ_VOLTA_I_K" "$G/mmq-config-volta.cuh" | tee -a "$LOG"

echo "=== build ===" | tee -a "$LOG"
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j8 --target llama-bench" > /tmp/build-t3b.log 2>&1 &
n=0
while pgrep -f "[c]make --build" >/dev/null 2>&1; do
  n=$((n+15)); [ "$n" -ge 430 ] && { echo "still building" | tee -a "$LOG"; break; }
  sleep 15
done
grep -iE "error:|FAILED" /tmp/build-t3b.log | head -6 | tee -a "$LOG" || true
tail -2 /tmp/build-t3b.log | tee -a "$LOG"

rm -rf /root/libdir-volta4 && mkdir -p /root/libdir-volta4
cp -a "$SRC/build/bin/." /root/libdir-volta4/
echo "=== md5 (must differ from volta2) ===" | tee -a "$LOG"
md5sum /root/libdir-volta2/libggml-cuda.so.0.24.0 /root/libdir-volta4/libggml-cuda.so.0.24.0 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== I=128 (volta2, current best) vs I=96 (volta4) ===" | tee -a "$LOG"
export CUDA_VISIBLE_DEVICES=0
for UB in 4 16 24 32 40 48; do
  for v in volta2 volta4; do
    ROW=$(LD_LIBRARY_PATH=/root/libdir-$v /root/libdir-$v/llama-bench \
      -m "$M" -p 512 -n 32 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 -b "$UB" -ub "$UB" -r 3 2>&1 | grep -F "pp512")
    printf "UB=%-3s %-7s %s\n" "$UB" "$v" "$ROW" | tee -a "$LOG"
  done
done
echo T3B_DONE
