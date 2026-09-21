#!/bin/bash
# Identify which source commit each llama-server binary was built from (read-only).
set -uo pipefail

echo "=== [1] candidate binaries: version + mtime ==="
for p in \
  /root/llm/llama.cpp/bin/llama-server \
  /root/llm/test/llama.cpp/build/bin/llama-server \
  /root/llm/test/v100-opt/llama.cpp/build/bin/llama-server ; do
  echo "--- $p"
  if [ -x "$p" ]; then
    ls -la "$p"
    "$p" --version 2>&1 | grep -iE "version|commit" | head -3
  else
    echo "(missing)"
  fi
  echo ""
done

echo "=== [2] C4 build CMakeCache highlights ==="
grep -iE "CMAKE_CUDA_ARCHITECTURES|CMAKE_BUILD_TYPE|LLAMA_CUDA|CMAKE_CUDA_COMPILER:" \
  /root/llm/test/v100-opt/llama.cpp/build/CMakeCache.txt 2>/dev/null | head -12
echo ""
echo "=== [3] any build-info / version files ==="
grep -m1 -r "b11053\|1af554f8f" /root/llm/test/v100-opt/llama.cpp/ggml/src/ggml-cuda/../../common/build-info.cpp 2>/dev/null | head -3
echo ""
echo "=== [4] build dir sizes (may be slow on NFS) ==="
du -sh /root/llm/test/v100-opt/llama.cpp/build 2>/dev/null
du -sh /root/llm/test/llama.cpp/build 2>/dev/null
echo ""
echo "=== [5] source tree markers: is v100-opt source == b11053? ==="
grep -m1 -n "LLAMA_BUILD_NUMBER\|GGML_VERSION" /root/llm/test/v100-opt/llama.cpp/CMakeLists.txt 2>/dev/null
grep -m1 -n "0.4.1" /root/llm/test/v100-opt/llama.cpp/ggml/CMakeLists.txt 2>/dev/null
grep -m1 -n "0.4.1" /root/llm/test/v100-opt/llama.cpp/src/CMakeLists.txt 2>/dev/null
echo ""
grep -m1 -n "0.4.0" /root/llm/test/llama.cpp/ggml/CMakeLists.txt 2>/dev/null
echo ""
echo "=== [6] nvcc / gcc availability ==="
/usr/local/cuda-12.4/bin/nvcc --version 2>&1 | tail -2
ls /opt/rh/gcc-toolset-12/root/usr/bin/gcc 2>&1
echo ""
echo "=== [7] cmake version ==="
cmake --version 2>&1 | head -1
echo DONE
