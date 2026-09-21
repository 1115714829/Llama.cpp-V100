#!/bin/bash
# After the scalar A/B arms: force-rebuild with the float4 kernel and measure the device-AR arm again.
while ! grep -q AROFF2_DONE /tmp/aroff2.log 2>/dev/null; do sleep 20; done
cd /root/llm/test/v100-opt/llama.cpp || exit 1
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
rm -f build-instr/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/allreduce.cu.o
echo VEC_REBUILD_START
cmake --build build-instr --config Release -j128 > /tmp/vec-build.log 2>&1
echo "VEC_BUILD_RC=$?"
echo -n 'error lines: '; grep -a -c -e error /tmp/vec-build.log
tail -1 /tmp/vec-build.log
cp -a build-instr/bin/. /root/libdir-instr/
md5sum /root/libdir-instr/libggml-cuda.so.0.24.0
echo -n 'marker count: '; strings /root/libdir-instr/libggml-cuda.so.0.24.0 | grep -a -c GGML_CUDA_AR_DEVICE
cd /root
env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=aronv ARENV=DEVICE=1 NPRED=512 bash /root/p60-ab-harness.sh 2>&1 | tail -22
grep -a -e 'device-side push AllReduce enabled' /tmp/p60-aronv-server.log 2>/dev/null | head -2
grep -a ar_us_avg /tmp/p60-aronv-server.log 2>/dev/null | tail -1
grep -a 'RT. perf' /tmp/p60-aronv-server.log 2>/dev/null | grep -a -e Qwen3.8-27B | tail -1
echo ARONV_DONE