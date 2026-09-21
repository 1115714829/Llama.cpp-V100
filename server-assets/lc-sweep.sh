#!/bin/bash
# Long-context decode sweep on the canonical build (round 40).
set -uo pipefail
cd /root/llm/test/v100-opt/llama.cpp
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
echo '=== REBUILD canonical (HEAD sources) ==='
cmake --build build-instr --config Release -j128 > /tmp/lc-build.log 2>&1
echo BUILD_RC=$?
tail -2 /tmp/lc-build.log
cp -a build-instr/bin/. /root/libdir-instr/
md5sum /root/libdir-instr/libggml-cuda.so.0.24.0 /root/libdir-instr/libggml-base.so.0.24.0 /root/libdir-instr/libllama.so.0.4.1 /root/libdir-instr/libllama-common.so.0.4.1
grep -c push_flag_stride ggml/src/ggml-cuda/allreduce.cu
export LD_LIBRARY_PATH=/root/libdir-instr
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
LB=/root/llm/test/v100-opt/llama.cpp/build-instr/bin/llama-bench
run() {
  echo "--- ARM $1 type=$2 depth=$3 fa=$4"
  sync; echo 3 > /proc/sys/vm/drop_caches
  $LB -m $M -ngl 999 -sm tensor -ts 1/1/1 -fa $4 -ctk $2 -ctv $2 -p 0 -n 64 -r 2 -d $3 2>&1 | grep -a -e tg -e pp -e error -e failed -e abort | tail -5
}
run q8-d8 q8_0 8192 1
run q8-d32 q8_0 32768 1
run q8-d128 q8_0 131072 1
run f16-d8 f16 8192 1
run f16-d32 f16 32768 1
run f16-d128 f16 131072 1
run q8-d256 q8_0 262144 1
run f16-d256 f16 262144 1
run nofa-q8-d8 q8_0 8192 0
run nofa-q8-d32 q8_0 32768 0
echo LC_SWEEP_DONE