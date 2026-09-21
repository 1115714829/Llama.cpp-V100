#!/bin/bash
# Discriminator: does the instrument fire under llama-bench (a much simpler decode path)?
set -uo pipefail
L=/root/libdir-rt
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
LOG=/tmp/p41.log
: > "$LOG"
export LD_LIBRARY_PATH="$L"
export CUDA_VISIBLE_DEVICES=0

echo "=== lib version sanity ===" | tee -a "$LOG"
"$L/llama-bench" --version 2>&1 | head -2 | tee -a "$LOG"
echo "marker in loaded lib: $(grep -c -a 'round timing (LLAMA_ROUND_TIMING)' "$L/libllama.so.0.4.1")" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== llama-bench with LLAMA_ROUND_TIMING=1 (single card, Q2_K_XL) ===" | tee -a "$LOG"
LLAMA_ROUND_TIMING=1 "$L/llama-bench" -m "$M" -p 64 -n 64 -r 1 -ngl 999 >> "$LOG" 2>&1
echo "bench exit=$?" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== our markers in the bench output ===" | tee -a "$LOG"
grep -a -e "LLAMA_ROUND_TIMING = " -e "round timing" -e "graphs reused" "$LOG" | head -12

echo "" | tee -a "$LOG"
echo "=== tail of bench output ===" | tee -a "$LOG"
tail -14 "$LOG"
echo P41_DONE
