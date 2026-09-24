#!/bin/bash
# p3-decomp T1 build: fattn-sm70-d256.cu only. Same discipline as gb-build.sh.
set -uo pipefail
LOG=/tmp/p3-build.log
MAKELOG=/tmp/p3-build-make.log
exec > "$LOG" 2>&1
echo "BUILD_START $(date)"
if [ -f /tmp/LLAMA_BUILD_LOCK ]; then echo "LOCK_HELD"; echo BUILD_DONE; exit 2; fi
if pgrep -f 'llama-serve[r] --model' >/dev/null; then echo "BUSY_SERVER"; echo BUILD_DONE; exit 2; fi
if pgrep -f 'llama-benc[h] -m' >/dev/null; then echo "BUSY_BENCH"; echo BUILD_DONE; exit 2; fi
if pgrep -f 'cmake --buil[d]' >/dev/null; then echo "BUSY_BUILD"; echo BUILD_DONE; exit 2; fi
touch /tmp/LLAMA_BUILD_LOCK
cd /root/llm/test/v100-opt/llama.cpp || { echo NO_TREE; rm -f /tmp/LLAMA_BUILD_LOCK; echo BUILD_DONE; exit 2; }
if [ ! -f /tmp/p3-fattn-sm70-d256.cu ] || [ ! -f /tmp/p3-fattn-sm70-decomp.cuh ]; then
  echo NO_SRC_PAYLOAD; rm -f /tmp/LLAMA_BUILD_LOCK; echo BUILD_DONE; exit 2
fi
sed 's/\r$//' /tmp/p3-fattn-sm70-d256.cu > ggml/src/ggml-cuda/fattn-sm70-d256.cu
sed 's/\r$//' /tmp/p3-fattn-sm70-decomp.cuh > ggml/src/ggml-cuda/fattn-sm70-decomp.cuh
if [ -f /tmp/p3-fattn79t-prefill.cu ]; then
  sed 's/\r$//' /tmp/p3-fattn79t-prefill.cu > ggml/src/ggml-cuda/fattn79t-prefill.cu
fi
if [ -f /tmp/p3-fattn79t-src.tgz ]; then
  tar xzf /tmp/p3-fattn79t-src.tgz -C ggml/src/ggml-cuda/
fi
if [ -f /tmp/p3-ggml-alloc.c ]; then
  sed 's/\r$//' /tmp/p3-ggml-alloc.c > ggml/src/ggml-alloc.c
fi
if [ -f /tmp/p3-ggml-backend.cpp ]; then
  sed 's/\r$//' /tmp/p3-ggml-backend.cpp > ggml/src/ggml-backend.cpp
fi
if [ -f /tmp/p3-ggml-backend-meta.cpp ]; then
  sed 's/\r$//' /tmp/p3-ggml-backend-meta.cpp > ggml/src/ggml-backend-meta.cpp
fi
if [ -f /tmp/p3-speculative.cpp ]; then
  sed 's/\r$//' /tmp/p3-speculative.cpp > common/speculative.cpp
fi
if [ -f /tmp/p3-llama-context.cpp ]; then
  sed 's/\r$//' /tmp/p3-llama-context.cpp > src/llama-context.cpp
fi
if [ -f /tmp/p3-llama-context.h ]; then
  sed 's/\r$//' /tmp/p3-llama-context.h > src/llama-context.h
fi
if [ -f /tmp/p3-llama-ext.h ]; then
  sed 's/\r$//' /tmp/p3-llama-ext.h > src/llama-ext.h
fi
if [ -f /tmp/p3-ggml-cuda-cmakelists.txt ]; then
  sed 's/\r$//' /tmp/p3-ggml-cuda-cmakelists.txt > ggml/src/ggml-cuda/CMakeLists.txt
fi
echo "SRC_MD5:"
md5sum ggml/src/ggml-cuda/fattn-sm70-d256.cu ggml/src/ggml-cuda/fattn-sm70-decomp.cuh
echo "MARK_DECOMP_ENV=$(grep -c LLAMA_SM70_FA_DECOMP ggml/src/ggml-cuda/fattn-sm70-d256.cu || true)"
echo "MARK_WS=$(grep -c sm70_decomp_ws_reserve ggml/src/ggml-cuda/fattn-sm70-d256.cu || true)"
echo "MARK_T1C=$(grep -c 'T1C' ggml/src/ggml-cuda/fattn-sm70-d256.cu || true)"
echo "MARK_79T_FLAT=$(grep -c T1C_CHK ggml/src/ggml-cuda/fattn79t-prefill.cu || true)"
echo "MARK_79T_OBJ=$(grep -c ggml-cuda-fattn79t ggml/src/ggml-cuda/CMakeLists.txt || true)"
cmake --build build-instr -j176 --target llama-server llama-bench > "$MAKELOG" 2>&1
BUILD_RC=$?
echo "BUILD_RC=$BUILD_RC"
echo "ERROR_LINES=$(grep -c 'error:' "$MAKELOG" || true)"
if [ "$BUILD_RC" != "0" ]; then
  grep -m 30 -a 'error' "$MAKELOG"
  rm -f /tmp/LLAMA_BUILD_LOCK
  echo BUILD_DONE
  exit 1
fi
mkdir -p /root/libdir-gb
cp -a build-instr/bin/. /root/libdir-gb/
echo "MARK_LIB_DECOMP=$(strings /root/libdir-gb/libggml-cuda.so.0.24.0 | grep -c LLAMA_SM70_FA_DECOMP || true)"
rm -f /tmp/LLAMA_BUILD_LOCK
echo BUILD_DONE
