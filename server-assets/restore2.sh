#!/bin/bash
set -u
cd /root/llm/test/v100-opt/llama.cpp || exit 1
echo '--- stray files in src/ ---'
ls -la src/allreduce.cu src/allreduce.cuh src/ggml-cuda.cu 2>&1 | awk '{print $5, $9}' | head -4
for f in allreduce.cu allreduce.cuh ggml-cuda.cu; do
  if [ -f src/$f ]; then mv -f src/$f ggml/src/ggml-cuda/$f; echo moved $f; fi
done
echo -n 'device AR in allreduce.cu (want 0): '; grep -a -c ggml_cuda_ar_device_allreduce ggml/src/ggml-cuda/allreduce.cu
echo -n 'AR hook in ggml-cuda.cu (want 0): '; grep -a -c ggml_backend_cuda_comm_init_device ggml/src/ggml-cuda/ggml-cuda.cu
echo -n 'FA probe (want 2): '; grep -a -c GGML_CUDA_FA_SPLIT_FLOOR ggml/src/ggml-cuda/fattn-common.cuh
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
find build-instr -name 'llama-graph.cpp.o' -o -name 'delta-net-base.cpp.o' -o -name 'llama-memory-recurrent.cpp.o' -o -name 'allreduce.cu.o' -o -name 'ggml-cuda.cu.o' | xargs -r rm -f
echo RESTORE2_BUILD_START
cmake --build build-instr --config Release -j128 > /tmp/restore2-build.log 2>&1
echo "RESTORE2_RC=$?"
echo -n 'error lines: '; grep -a -c -e error /tmp/restore2-build.log
tail -1 /tmp/restore2-build.log
cp -a build-instr/bin/. /root/libdir-instr/
md5sum /root/libdir-instr/libggml-cuda.so.0.24.0 /root/libdir-instr/libllama.so.0.4.1
echo -n 'device AR marker (want 0): '; strings /root/libdir-instr/libggml-cuda.so.0.24.0 | grep -a -c GGML_CUDA_AR_DEVICE
echo -n 'rs index marker (want 0): '; strings /root/libdir-instr/libllama.so.0.4.1 | grep -a -c GGML_RS_INDEX_WRITE
echo -n 'FA probe marker (want 1): '; strings /root/libdir-instr/libggml-cuda.so.0.24.0 | grep -a -c GGML_CUDA_FA_SPLIT_FLOOR
echo RESTORE2_DONE