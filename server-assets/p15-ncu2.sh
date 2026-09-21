#!/bin/bash
# t9b part 2 (retry): confirm the kernel switch at ne11=8 using the FULL PATH to ncu.
set -uo pipefail
NCU=/usr/local/cuda-12.4/bin/ncu
M=/root/llm/models/Qwen3.8-27B-TurboFCFusion-gguf/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf
export CUDA_VISIBLE_DEVICES=0

"$NCU" --version 2>&1 | grep -iE "version" | head -2

for v in pristine volta2; do
  echo ""
  echo "########## $v  (UB=8, ne11=8) ##########"
  LD_LIBRARY_PATH=/root/libdir-$v "$NCU" --csv -k "regex:mul_mat" -c 8 \
    /root/libdir-$v/llama-bench -m "$M" -p 512 -n 8 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 \
    -b 8 -ub 8 -r 1 > "/tmp/ncu-$v.out" 2>&1
  echo "exit=$?  bytes=$(wc -c < /tmp/ncu-$v.out)"
  echo "--- kernel names seen ---"
  grep -oE "mul_mat_vec_q[a-zA-Z_0-9]*|mul_mat_q[a-zA-Z_0-9]*" "/tmp/ncu-$v.out" | sort | uniq -c | sort -rn | head -8
  echo "--- any ncu error? ---"
  grep -iE "error|not permitted|fail" "/tmp/ncu-$v.out" | head -3 || echo "(none)"
done
echo NCU2_DONE
