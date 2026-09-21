#!/bin/bash
# t9b part 2: mechanically confirm which kernel runs at ne11=8 for each variant.
# If ncu is unavailable, fall back to the construction argument (documented separately).
set -uo pipefail
M=/root/llm/models/Qwen3.8-27B-TurboFCFusion-gguf/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf
export CUDA_VISIBLE_DEVICES=0

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu NOT INSTALLED -> rely on the UB=4 control (identical) vs UB=8 (15% apart) argument"
  echo NCU_DONE
  exit 0
fi
echo "ncu: $(ncu --version 2>&1 | tail -1)"

for v in pristine volta2; do
  echo ""
  echo "########## $v  (-ub 8, ne11=8) ##########"
  LD_LIBRARY_PATH=/root/libdir-$v ncu --csv -k "regex:mul_mat" -c 10 \
    /root/libdir-$v/llama-bench -m "$M" -p 512 -n 8 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 \
    -b 8 -ub 8 -r 1 2>&1 \
    | grep -oE "mul_mat_vec_q[a-zA-Z_0-9]*|mul_mat_q[a-zA-Z_0-9]*|mmq[a-zA-Z_0-9]*" \
    | sort | uniq -c | sort -rn | head -10
done
echo NCU_DONE
