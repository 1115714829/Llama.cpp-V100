#!/bin/bash
# Phase 0: normalize CRLF, launch the Q8_0 sha256 verify, and build the timing instrument.
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp

echo "=== [1] CRLF normalize (files come from a Windows checkout) ==="
sed -i 's/\r$//' "$SRC/src/llama-context.h" "$SRC/src/llama-context.cpp"
echo -n "llama-context.cpp LLAMA_ROUND_TIMING refs: "; grep -c LLAMA_ROUND_TIMING "$SRC/src/llama-context.cpp"
echo -n "llama-context.h   round counter fields     : "; grep -c "n_rt_rounds\|n_rt_rebuild\|t_rt_compute_us" "$SRC/src/llama-context.h"

echo ""
echo "=== [2] start Q8_0 sha256 verification (background, 29 GB read) ==="
bash /root/p33-verify-q8.sh

echo ""
echo "=== [3] launch incremental build ==="
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j82 --target llama-server llama-bench" > /tmp/build-rt.log 2>&1 &
echo "build pid=$!"
sleep 20
echo "--- build log head ---"
tail -8 /tmp/build-rt.log
echo P34_DONE
