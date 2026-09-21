#!/bin/bash
# Controlled A/B with the AR timing probe on: baseline vs device AR (GRID=4).
cd /root/llm/test/v100-opt/llama.cpp || exit 1
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
rm -f build-instr/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/allreduce.cu.o
echo G4_REBUILD_START
cmake --build build-instr --config Release -j128 > /tmp/g4-build.log 2>&1
echo "G4_RC=$?"
echo -n 'error lines: '; grep -a -c -e error /tmp/g4-build.log
tail -1 /tmp/g4-build.log
cp -a build-instr/bin/. /root/libdir-instr/
md5sum /root/libdir-instr/libggml-cuda.so.0.24.0
echo -n 'marker: '; strings /root/libdir-instr/libggml-cuda.so.0.24.0 | grep -a -c GGML_CUDA_AR_DEVICE
cd /root
echo ===ARM-A-baseline-with-timing===
env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=btim ARENV=TIMING=1 NPRED=512 bash /root/p60-ab-harness.sh 2>&1 | grep -a -e prompt -e MEDIAN -e greedy -e ar_us_avg
grep -a ar_us_avg /tmp/p60-btim-server.log | tail -1
echo ===ARM-B-device-AR-GRID4-with-timing===
env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=g4 ARENV=DEVICE=1,TIMING=1 NPRED=512 bash /root/p60-ab-harness.sh 2>&1 | grep -a -e prompt -e MEDIAN -e greedy -e ar_us_avg
grep -a ar_us_avg /tmp/p60-g4-server.log | tail -1
grep -a -e 'device-side push AllReduce enabled' -e 'device-side AllReduce init failed' /tmp/p60-g4-server.log | head -2
echo G4AB_DONE