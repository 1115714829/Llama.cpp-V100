#!/bin/bash
# Stop the invalid A/B, force a real rebuild of the AR integration, verify the marker.
for p in $(ps -eo pid,args | awk '/bash \/root\/ar-ab.sh/ {print $1}'); do kill $p 2>/dev/null; done
for p in $(ps -eo pid,args | awk '/bash \/root\/p60-ab-harness.sh/ {print $1}'); do kill $p 2>/dev/null; done
sleep 2
for p in $(pgrep -x llama-server); do kill $p 2>/dev/null; done
sleep 3
for p in $(pgrep -x llama-server); do kill -9 $p 2>/dev/null; done
cd /root/llm/test/v100-opt/llama.cpp || exit 1
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
rm -f build-instr/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/ggml-cuda.cu.o
rm -f build-instr/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/allreduce.cu.o
echo REBUILD_START
cmake --build build-instr --config Release -j128 > /tmp/ardev-build3.log 2>&1
echo "BUILD3_RC=$?"
echo -n 'error lines: '; grep -a -c -e error /tmp/ardev-build3.log
tail -2 /tmp/ardev-build3.log
cp -a build-instr/bin/. /root/libdir-instr/
md5sum /root/libdir-instr/libggml-cuda.so.0.24.0
echo -n 'marker count: '; strings /root/libdir-instr/libggml-cuda.so.0.24.0 | grep -a -c GGML_CUDA_AR_DEVICE
echo ARDEV3_DONE