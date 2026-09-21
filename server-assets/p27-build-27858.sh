#!/bin/bash
# Phase 1a: build the PR #27858 port into the server tree (DFlash2 CPU-side selector for --split-mode tensor)
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
LOG=/tmp/test27858.log
: > "$LOG"
exec >> "$LOG" 2>&1

echo "=== env (RAM/GPU before build; big -j eats HBM2-backed RAM) ==="
free -g | head -3
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "=== normalize CRLF (files come from a Windows checkout) ==="
cd "$SRC" || exit 1
for f in common/common.cpp common/speculative.cpp common/speculative.h \
         src/llama-ext.h src/llama-model.cpp src/llama-model.h \
         src/models/dflash.cpp ggml/src/ggml-backend-meta.cpp ; do
  sed -i 's/\r$//' "$f"
done

echo "=== port symbols present (all must be > 0) ==="
echo -n "dflash.cpp use_cpu_selector        : "; grep -c 'use_cpu_selector' src/models/dflash.cpp
echo -n "speculative.cpp is_dflash2_cpu     : "; grep -c 'is_dflash2_cpu' common/speculative.cpp
echo -n "speculative.cpp load_selector      : "; grep -c 'load_dflash2_selector' common/speculative.cpp
echo -n "speculative.cpp build_sel_cpu      : "; grep -c 'build_dflash2_selector_cpu' common/speculative.cpp
echo -n "speculative.h is_block_draft       : "; grep -c 'common_speculative_is_block_draft' common/speculative.h
echo -n "common.cpp is_block_draft          : "; grep -c 'common_speculative_is_block_draft' common/common.cpp
echo -n "llama-ext.h get_split_mode         : "; grep -c 'llama_model_get_split_mode' src/llama-ext.h
echo -n "llama-model.cpp get_split_mode     : "; grep -c 'llama_model_get_split_mode' src/llama-model.cpp
echo -n "dflash.cpp selector gate           : "; grep -c 'split_mode() != LLAMA_SPLIT_MODE_TENSOR' src/models/dflash.cpp

echo "=== temporary diagnostic must be ABSENT (expect 0) ==="
echo -n "meta.cpp diagnostic                : "; grep -c 'per-row op does not support' ggml/src/ggml-backend-meta.cpp

echo "=== launch incremental build (llama-server + llama-bench) ==="
nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j8 --target llama-server llama-bench" > /tmp/build-p27858.log 2>&1 &
echo "launched pid=$!"
sleep 15
echo "--- build log head ---"
tail -5 /tmp/build-p27858.log
echo TEST27858_LAUNCHED
