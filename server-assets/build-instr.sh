#!/bin/bash
# Build the B1/B2 instrumented tree (build-instr) and snapshot it to /root/libdir-inst
set -e
SRC=/root/llm/test/v100-opt/llama.cpp
NCCLROOT=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
cd $SRC
echo === free before configure ===; free -g | head -2
echo === configure ===; date
cmake -B build-instr -DCMAKE_BUILD_TYPE=Release -DLLAMA_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
  -DCMAKE_INSTALL_RPATH=/root/llm/llama.cpp/lib64 \
  -DLLAMA_BUILD_TESTS=ON \
  -DNCCL_INCLUDE_DIR=$NCCLROOT/include \
  -DNCCL_LIBRARY=$NCCLROOT/lib/libnccl.so.2 2>&1 | tail -8
echo === build ===; date
cmake --build build-instr --config Release -j64 2>&1 | tail -30
echo === snapshot ===; date
mkdir -p /root/libdir-inst
cp -a build-instr/bin/. /root/libdir-instr/
echo === md5 self-check (must DIFFER from libdir-nccl) ===; 
md5sum /root/libdir-instr/libggml-cuda.so.0.24.0 /root/libdir-nccl/libggml-cuda.so.0.24.0
md5sum /root/libdir-instr/libllama-common.so.0.4.1 /root/libdir-nccl/libllama-common.so.0.4.1
echo BUILD_ALL_DONE; date