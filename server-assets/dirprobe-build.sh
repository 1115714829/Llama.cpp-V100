#!/bin/bash
set -uo pipefail
LOG=/tmp/dirprobe-build.log
cd /root/llm/test/v100-opt/llama.cpp || exit 99
date
cmake --build build-instr --config Release -j128 > "$LOG" 2>&1
RC=$?
echo "BUILD_RC=$RC"
echo "ERROR_LINES=$(grep -c -e error "$LOG")"
echo "BUILT_TARGET_LINES=$(grep -c -e 'Built target' "$LOG")"
echo "BUILDING_LINES=$(grep -c -e 'Building CXX' -e 'Building CUDA' "$LOG")"
if [ "$RC" != "0" ]; then
  echo "--- last 60 lines of build log ---"
  tail -60 "$LOG"
  echo "DIRPROBE_BUILD_FAILED"
  exit 1
fi
cp -a build-instr/bin/. /root/libdir-instr/
echo "CP_RC=$?"
echo "MARKER_COUNT=$(strings /root/libdir-instr/libggml-cuda.so.0.24.0 | grep -a -c DIRECT_DEBUG)"
md5sum /root/libdir-instr/libggml-cuda.so.0.24.0 /root/libdir-instr/libllama-common.so.0.4.1 /root/libdir-instr/libllama.so.0.4.1
ls -la /root/libdir-instr/libggml-cuda.so.0.24.0
date
echo "DIRPROBE_BUILD_DONE"
