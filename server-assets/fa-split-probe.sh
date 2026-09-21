#!/bin/bash
# Round 41: probe GGML_CUDA_FA_SPLIT_FLOOR (deeper KV split) at long context. Diagnostic: NODROP, llama-bench only.
set -uo pipefail
while ! grep -q CHAIN2_DONE /tmp/lc-chain2.log 2>/dev/null; do sleep 30; done
cd /root/llm/test/v100-opt/llama.cpp
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
echo '=== snapshot canonical libs before rebuilding with the probe ==='
rm -rf /root/libdir-canon; cp -a /root/libdir-instr /root/libdir-canon
md5sum /root/libdir-canon/libggml-cuda.so.0.24.0
echo '=== BUILD with GGML_CUDA_FA_SPLIT_FLOOR probe ==='
cmake --build build-instr --config Release -j128 > /tmp/fs-build.log 2>&1
rc=$?; echo BUILD_RC=$rc
if [ $rc -ne 0 ]; then tail -25 /tmp/fs-build.log; echo FS_DONE STATUS=BUILD_FAIL; exit 1; fi
cp -a build-instr/bin/. /root/libdir-instr/
md5sum /root/libdir-instr/libggml-cuda.so.0.24.0
echo -n 'probe marker count: '; strings /root/libdir-instr/libggml-cuda.so.0.24.0 | grep -a -c GGML_CUDA_FA_SPLIT_FLOOR
export LD_LIBRARY_PATH=/root/libdir-instr
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
LB=/root/llm/test/v100-opt/llama.cpp/build-instr/bin/llama-bench
run() {
  echo "--- ARM floor=$1 depth=$2"
  GGML_CUDA_FA_SPLIT_FLOOR=$1 $LB -m $M -ngl 999 -sm tensor -ts 1/1/1 -fa on -ctk q8_0 -ctv q8_0 -p 0 -n 64 -r 2 -d $2 2>&1 | grep -a -e tg -e error -e failed | tail -4
}
for d in 32768 131072; do
  for f in 0 64 256 1024; do run $f $d; done
done
for f in 0 1024; do run $f 262144; done
echo FS_DONE