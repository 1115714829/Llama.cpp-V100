#!/bin/bash
# t9b part 2 (nsys attempt): name the kernels at ne11=8 for each variant.
set -uo pipefail
NSYS=/usr/local/cuda-12.4/bin/nsys
M=/root/llm/models/Qwen3.8-27B-TurboFCFusion-gguf/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf
export CUDA_VISIBLE_DEVICES=0

for v in pristine volta2; do
  echo ""
  echo "########## $v  (UB=8 -> ne11=8) ##########"
  LD_LIBRARY_PATH=/root/libdir-$v "$NSYS" profile --stats=true -o "/tmp/nsys-$v" --force-overwrite true \
    /root/libdir-$v/llama-bench -m "$M" -p 512 -n 8 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 \
    -b 8 -ub 8 -r 1 > "/tmp/nsys-$v.txt" 2>&1
  echo "exit=$?"
  echo "--- mmq/mmvq kernel names in the GPU kernel summary ---"
  grep -iE "mul_mat_vec_q|mul_mat_q|mmq" "/tmp/nsys-$v.txt" | head -12
  echo "--- errors? ---"
  grep -iE "error|fail|not permitted" "/tmp/nsys-$v.txt" | head -3 || echo "(none)"
done
echo NSYS_DONE
