#!/bin/bash
cd /root/llm/test/v100-opt/llama.cpp || exit 9
LOG=/root/fak-build.log
T0=$(date +%s)
echo "START=$T0" > /root/fak-build-times.txt
cmake --build build-instr -j128 > $LOG 2>&1
RC=$?
T1=$(date +%s)
echo "END=$T1" >> /root/fak-build-times.txt
echo "WALL=$(expr $T1 - $T0)" >> /root/fak-build-times.txt
cat /root/fak-build-times.txt
echo "BUILD_RC=$RC"
echo "CHECK1_errcount=$(grep -c -e error $LOG)"
echo "CHECK2_built=$(grep -c -e 'Built target' $LOG)"
echo "CHECK2_building=$(grep -c -e 'Building' $LOG)"
echo "CHECK3_marker=$(strings build-instr/bin/libggml-cuda.so | grep -a -c GGML_CUDA_FA_KERNEL_DEBUG)"
if [ "$RC" != "0" ]; then
  echo "----- FIRST 40 ERROR LINES -----"
  grep -n -e error -e Error $LOG | head -40
fi
exit $RC
