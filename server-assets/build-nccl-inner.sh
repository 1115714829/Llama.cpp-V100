#!/bin/bash
# Build llama.cpp ggml-cuda with NCCL (from the ac922 venv bundle), matching build/ config.
set -u
SRC=/root/llm/test/v100-opt/llama.cpp
PKG=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime
cd "$SRC" || exit 1
echo "=== src identity (13 modified files on server) ==="
md5sum common/common.cpp common/sampling.cpp common/speculative.cpp common/speculative.h \
  ggml/src/ggml-cuda/mmvq.cu ggml/src/ggml-cuda/mmvq.cuh \
  src/llama-context.cpp src/llama-context.h src/llama-ext.h src/llama-model.cpp src/llama-model.h \
  src/models/dflash.cpp tools/server/server-context.cpp 2>&1
echo "=== nccl bundle ==="
ls -l "$PKG/lib/libnccl.so.2" "$PKG/include/nccl.h"
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
echo "=== configure build-nccl ==="
cmake -B build-nccl -DCMAKE_BUILD_TYPE=Release -DLLAMA_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
  -DCMAKE_INSTALL_RPATH=/root/llm/llama.cpp/lib64 \
  -DNCCL_INCLUDE_DIR="$PKG/include" -DNCCL_LIBRARY="$PKG/lib/libnccl.so.2" 2>&1 | tail -40
echo "=== cache option diff (build vs build-nccl) ==="
grep -E "^(LLAMA_|GGML_|CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|NCCL_)" build/CMakeCache.txt | grep -v "^//" | sort > /tmp/cache-build.txt
grep -E "^(LLAMA_|GGML_|CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|NCCL_)" build-nccl/CMakeCache.txt | grep -v "^//" | sort > /tmp/cache-build-nccl.txt
diff /tmp/cache-build.txt /tmp/cache-build-nccl.txt
echo "(diff exit=$?)"
echo "=== build (this is the long part) ==="
cmake --build build-nccl --config Release -j82 2>&1 | tail -50
echo "=== verify nccl linkage of the built lib ==="
ldd build-nccl/bin/libggml-cuda.so.0.24.0
echo "=== snapshot to /root/libdir-nccl ==="
mkdir -p /root/libdir-nccl
cp -a build-nccl/bin/. /root/libdir-nccl/
md5sum /root/libdir-nccl/libggml-cuda.so.0.24.0 /root/libdir-nccl/libllama.so.0.4.1 /root/libdir-nccl/libllama-common.so.0.4.1
echo "BUILD_NCCL_DONE"
