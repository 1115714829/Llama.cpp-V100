#!/bin/bash
# Diagnose why I=96 fails at larger J: show the raw output.
set -uo pipefail
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
export CUDA_VISIBLE_DEVICES=0
for UB in 8 16 32; do
  echo "########## volta4 (I=96) UB=$UB ##########"
  LD_LIBRARY_PATH=/root/libdir-volta4 /root/libdir-volta4/llama-bench \
    -m "$M" -p 128 -n 8 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 -b "$UB" -ub "$UB" -r 1 2>&1 | tail -8
  echo "exit=$?"
  echo ""
done
echo "########## volta2 (I=128) UB=16 for contrast ##########"
LD_LIBRARY_PATH=/root/libdir-volta2 /root/libdir-volta2/llama-bench \
  -m "$M" -p 128 -n 8 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 -b 16 -ub 16 -r 1 2>&1 | tail -8
echo T3DIAG_DONE
