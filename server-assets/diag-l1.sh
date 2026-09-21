#!/bin/bash
# Diagnose why the L1 sweep produced no pp/tg values: show the RAW output of one run.
set -uo pipefail
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
export CUDA_VISIBLE_DEVICES=2
BIN=/root/libdir-pristine/llama-bench
export LD_LIBRARY_PATH=/root/libdir-pristine

echo "=== [1] does -ub/-b work? full output, ub=64 ==="
"$BIN" -m "$M" -p 512 -n 32 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 -b 64 -ub 64 -r 3 2>&1 | tail -25
echo ""
echo "=== [2] same, but WITHOUT -b/-ub (baseline invocation) ==="
"$BIN" -m "$M" -p 512 -n 32 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 -r 3 2>&1 | tail -12
echo ""
echo "=== [3] what does --help say about batch flags? ==="
"$BIN" --help 2>&1 | grep -iE "ubatch|batch-size" | head -8
echo DIAG_DONE
