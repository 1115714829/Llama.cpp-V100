#!/bin/bash
# Path B build: fattn-sm70-d256 only. Does NOT pull /tmp/t15 shadow sources.
set -uo pipefail
LOG=/tmp/pathb-build.log
MAKELOG=/tmp/pathb-build-make.log
exec > "$LOG" 2>&1
echo "BUILD_START $(date)"
if [ -f /tmp/LLAMA_BUILD_LOCK ]; then echo "LOCK_HELD"; echo BUILD_DONE; exit 2; fi
if pgrep -f 'llama-serve[r] --model' >/dev/null; then echo "BUSY_SERVER"; echo BUILD_DONE; exit 2; fi
if pgrep -f 'llama-benc[h] -m' >/dev/null; then echo "BUSY_BENCH"; echo BUILD_DONE; exit 2; fi
if pgrep -f 'cmake --buil[d]' >/dev/null; then echo "BUSY_BUILD"; echo BUILD_DONE; exit 2; fi
touch /tmp/LLAMA_BUILD_LOCK
cd /root/llm/test/v100-opt/llama.cpp || { echo NO_TREE; rm -f /tmp/LLAMA_BUILD_LOCK; echo BUILD_DONE; exit 2; }
# Install Path B sources from /tmp (scp payloads). Strip CR only - never `tr -d r`.
# (PowerShell->ssh eats double quotes: a host-side `tr -d "\r"` becomes `tr -d r`.)
if [ ! -f /tmp/pathb-fattn-sm70-d256.cu ] || [ ! -f /tmp/pathb-fattn-sm70-d256-kernel.cuh ]; then
  echo NO_SRC_PAYLOAD
  rm -f /tmp/LLAMA_BUILD_LOCK
  echo BUILD_DONE
  exit 2
fi
sed 's/\r$//' /tmp/pathb-fattn-sm70-d256.cu > ggml/src/ggml-cuda/fattn-sm70-d256.cu
sed 's/\r$//' /tmp/pathb-fattn-sm70-d256-kernel.cuh > ggml/src/ggml-cuda/fattn-sm70-d256-kernel.cuh
echo "SRC_MD5:"
md5sum ggml/src/ggml-cuda/fattn-sm70-d256.cu ggml/src/ggml-cuda/fattn-sm70-d256-kernel.cuh
grep -c 'fattn-sm70-d256-kernel.cuh' ggml/src/ggml-cuda/fattn-sm70-d256.cu
grep -c 'LLAMA_SM70_Q8_DIRECT' ggml/src/ggml-cuda/fattn-sm70-d256.cu
cmake --build build-instr -j128 --target ggml-cuda llama-bench > "$MAKELOG" 2>&1
BUILD_RC=$?
echo "BUILD_RC=$BUILD_RC"
echo "ERROR_LINES=$(grep -c 'error:' "$MAKELOG" || true)"
echo "BUILT_TARGETS=$(grep -c 'Built target' "$MAKELOG" || true)"
if [ "$BUILD_RC" != "0" ]; then
  grep -m 30 -a 'error' "$MAKELOG"
  rm -f /tmp/LLAMA_BUILD_LOCK
  echo BUILD_DONE
  exit 1
fi
mkdir -p /root/libdir-pathb
cp -a build-instr/bin/. /root/libdir-pathb/
echo "MARK_Q8=$(strings /root/libdir-pathb/libggml-cuda.so | grep -c LLAMA_SM70_Q8_DIRECT || true)"
echo "MARK_FA_GEMM=$(strings /root/libdir-pathb/libggml-cuda.so | grep -c LLAMA_SM70_FA_GEMM || true)"
echo "MARK_D256=$(strings /root/libdir-pathb/libggml-cuda.so | grep -c LLAMA_SM70_D256 || true)"
echo "LIB_MD5:"
md5sum /root/libdir-pathb/libggml-cuda.so.0.24.0 /root/libdir-pathb/libggml-base.so.0.24.0 2>/dev/null || md5sum /root/libdir-pathb/libggml-cuda.so /root/libdir-pathb/libggml-base.so
rm -f /tmp/LLAMA_BUILD_LOCK
echo BUILD_DONE
