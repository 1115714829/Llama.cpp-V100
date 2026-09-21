#!/bin/bash
# Build A2 (index-based recurrent writes) and verify: bitwise-identical greedy sha256 + rebuild/alloc + ms/round.
cd /root/llm/test/v100-opt/llama.cpp || exit 1
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
find build-instr -name 'delta-net-base.cpp.o' -o -name 'llama-graph.cpp.o' -o -name 'llama-memory-recurrent.cpp.o' | xargs -r rm -f
echo A2_BUILD_START
cmake --build build-instr --config Release -j128 > /tmp/a2-build.log 2>&1
echo "A2_RC=$?"
echo -n 'error lines: '; grep -a -c -e error /tmp/a2-build.log
grep -a -c -e 'Building' -e 'Built target' /tmp/a2-build.log
tail -2 /tmp/a2-build.log
cp -a build-instr/bin/. /root/libdir-instr/
md5sum /root/libdir-instr/libggml-cuda.so.0.24.0 /root/libdir-instr/libllama.so.0.4.1
cd /root
env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=a2 NPRED=512 bash /root/p60-ab-harness.sh 2>&1 | grep -a -e prompt -e MEDIAN -e greedy -e 'launch failure'
echo '--- round probes ---'
grep -a 'RT. perf' /tmp/p60-a2-server.log 2>/dev/null | grep -a -e Qwen3.8-27B | tail -2
echo A2_DONE