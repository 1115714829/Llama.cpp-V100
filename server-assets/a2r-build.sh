#!/bin/bash
# A2 re-test: incremental build of build-instr + deploy to /root/libdir-instr
cd /root/llm/test/v100-opt/llama.cpp || exit 9
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
echo PWD=$(pwd)
echo ---objects-before---
find build-instr -name llama-graph.cpp.o -o -name llama-memory-recurrent.cpp.o -o -name delta-net-base.cpp.o | tee /tmp/a2r-objs.txt
xargs -r rm -f < /tmp/a2r-objs.txt
rm -f /tmp/a2r-build.log
cmake --build build-instr --config Release -j128 > /tmp/a2r-build.log 2>&1
BUILD_RC=$?
echo BUILD_RC=$BUILD_RC
if [ $BUILD_RC -ne 0 ]; then
  echo ---last-40---
  tail -40 /tmp/a2r-build.log
  echo A2R_BUILD_FAILED
  exit 1
fi
cp -a build-instr/bin/. /root/libdir-instr/
echo ---CHECKS---
echo CHECK1_error_lines=$(grep -c -e error /tmp/a2r-build.log)
echo CHECK2_built_target=$(grep -c -e 'Built target' /tmp/a2r-build.log)
echo CHECK2_building_lines=$(grep -c -e 'Building' /tmp/a2r-build.log)
echo CHECK3_marker_libllama=$(strings /root/libdir-instr/libllama.so.0.4.1 | grep -a -c GGML_RS_INDEX_WRITE)
md5sum /root/libdir-instr/libllama.so.0.4.1 /root/libdir-instr/libggml-cuda.so.0.24.0 /root/libdir-instr/libllama-common.so.0.4.1 /root/libdir-instr/libggml-base.so.0.24.0
echo ---recompiled-TUs---
grep -o -e 'llama[a-z-]*\.cpp\.o' -e 'delta-net-base\.cpp\.o' /tmp/a2r-build.log | sort -u | head -30
echo A2R_BUILD_OK
