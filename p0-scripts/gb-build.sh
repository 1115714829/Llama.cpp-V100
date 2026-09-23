#!/bin/bash
# P-GRAPHTAX build: llama-graph.cpp + llama-kv-cache.{cpp,h} (n_kv shape bucketing).
# Follows pathb-build.sh discipline: build lock, busy checks, CR strip, marker+md5 proof.
set -uo pipefail
LOG=/tmp/gb-build.log
MAKELOG=/tmp/gb-build-make.log
exec > "$LOG" 2>&1
echo "BUILD_START $(date)"
if [ -f /tmp/LLAMA_BUILD_LOCK ]; then echo "LOCK_HELD"; echo BUILD_DONE; exit 2; fi
if pgrep -f 'llama-serve[r] --model' >/dev/null; then echo "BUSY_SERVER"; echo BUILD_DONE; exit 2; fi
if pgrep -f 'llama-benc[h] -m' >/dev/null; then echo "BUSY_BENCH"; echo BUILD_DONE; exit 2; fi
if pgrep -f 'cmake --buil[d]' >/dev/null; then echo "BUSY_BUILD"; echo BUILD_DONE; exit 2; fi
touch /tmp/LLAMA_BUILD_LOCK
cd /root/llm/test/v100-opt/llama.cpp || { echo NO_TREE; rm -f /tmp/LLAMA_BUILD_LOCK; echo BUILD_DONE; exit 2; }
if [ ! -f /tmp/gb-llama-graph.cpp ] || [ ! -f /tmp/gb-llama-kv-cache.cpp ] || [ ! -f /tmp/gb-llama-kv-cache.h ]; then
  echo NO_SRC_PAYLOAD; rm -f /tmp/LLAMA_BUILD_LOCK; echo BUILD_DONE; exit 2
fi
sed 's/\r$//' /tmp/gb-llama-graph.cpp > src/llama-graph.cpp
sed 's/\r$//' /tmp/gb-llama-kv-cache.cpp > src/llama-kv-cache.cpp
sed 's/\r$//' /tmp/gb-llama-kv-cache.h > src/llama-kv-cache.h
echo "SRC_MD5:"
md5sum src/llama-graph.cpp src/llama-kv-cache.cpp src/llama-kv-cache.h
echo "MARK_BUCKET_H=$(grep -c GGML_KV_BUCKET_RATIO src/llama-kv-cache.h || true)"
echo "MARK_BUCKET_GRAPH=$(grep -c llama_kv_bucket_n src/llama-graph.cpp || true)"
echo "MARK_BUCKET_KV=$(grep -c llama_kv_bucket_n src/llama-kv-cache.cpp || true)"
cmake --build build-instr -j128 --target llama-server llama-bench > "$MAKELOG" 2>&1
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
mkdir -p /root/libdir-gb
cp -a build-instr/bin/. /root/libdir-gb/
echo "MARK_LIB_BUCKET=$(strings /root/libdir-gb/libllama.so.0.4.1 | grep -c GGML_KV_BUCKET_RATIO || true)"
echo "LIB_MD5:"
md5sum /root/libdir-gb/libggml-cuda.so.0.24.0 /root/libdir-gb/libggml-base.so.0.24.0 /root/libdir-gb/libllama.so.0.4.1 2>/dev/null
rm -f /tmp/LLAMA_BUILD_LOCK
echo BUILD_DONE
