#!/bin/bash
# Restore the server tree to canonical plus the FA split probe only, then rebuild and verify the markers.
cd /root/llm/test/v100-opt/llama.cpp || exit 1
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
find build-instr -name 'llama-graph.cpp.o' -o -name 'delta-net-base.cpp.o' -o -name 'llama-memory-recurrent.cpp.o' -o -name 'allreduce.cu.o' -o -name 'ggml-cuda.cu.o' | xargs -r rm -f
echo RESTORE_BUILD_START
cmake --build build-instr --config Release -j128 > /tmp/restore-build.log 2>&1
echo "RESTORE_RC=$?"
echo -n 'error lines: '; grep -a -c -e error /tmp/restore-build.log
tail -1 /tmp/restore-build.log
cp -a build-instr/bin/. /root/libdir-instr/
md5sum /root/libdir-instr/libggml-cuda.so.0.24.0 /root/libdir-instr/libllama.so.0.4.1
echo -n 'device AR marker (want 0): '; strings /root/libdir-instr/libggml-cuda.so.0.24.0 | grep -a -c GGML_CUDA_AR_DEVICE
echo -n 'rs index marker (want 0): '; strings /root/libdir-instr/libllama.so.0.4.1 | grep -a -c GGML_RS_INDEX_WRITE
echo -n 'fa split probe marker (want 1): '; strings /root/libdir-instr/libggml-cuda.so.0.24.0 | grep -a -c GGML_CUDA_FA_SPLIT_FLOOR
echo RESTORE_DONE