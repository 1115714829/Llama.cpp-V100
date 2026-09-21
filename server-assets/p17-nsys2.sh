#!/bin/bash
# t9b part 2 (3rd attempt): profile the 9.14 GiB model (the profiler cannot load the 18 GiB one).
# Goal: name the kernel at ne11=8 for pristine (MMVQ) vs volta2 (MMQ).
set -uo pipefail
NSYS=/usr/local/cuda-12.4/bin/nsys
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
export CUDA_VISIBLE_DEVICES=0

for v in pristine volta2; do
  echo ""
  echo "########## $v  (Q2_K_XL, UB=8 -> ne11=8) ##########"
  LD_LIBRARY_PATH=/root/libdir-$v "$NSYS" profile --stats=true -o "/tmp/ns2-$v" --force-overwrite true \
    /root/libdir-$v/llama-bench -m "$M" -p 512 -n 8 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 \
    -b 8 -ub 8 -r 1 > "/tmp/ns2-$v.txt" 2>&1
  echo "exit=$?"
  echo "--- kernel names (mul_mat family) ---"
  grep -oE "mul_mat_vec_q[a-zA-Z_0-9]*|mul_mat_q[a-zA-Z_0-9]*|mul_mat_vec_q\+[a-zA-Z_0-9]*" "/tmp/ns2-$v.txt" \
    | sort | uniq -c | sort -rn | head -8
  echo "--- errors? ---"
  grep -iE "^llama_bench: error|not permitted|failed to load" "/tmp/ns2-$v.txt" | head -2 || echo "(none)"
done
echo ""
echo "=== also: which kernel does the UB=4 (control) case pick? ==="
grep -oE "mul_mat_vec_q[a-zA-Z_0-9]*|mul_mat_q[a-zA-Z_0-9]*" /tmp/ns2-pristine.txt | sort -u | head -6
echo NSYS2_DONE
