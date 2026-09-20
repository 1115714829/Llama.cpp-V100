#!/bin/bash
# t9b part 1: verify C5 (Volta K-quant crossover) on the PRODUCTION model (Q4_K_M), 2 cards.
# Same-source A/B: pristine (crossover 8) vs volta2 (crossover 4).
#   -ub 4 -> both use MMVQ  -> expect NO difference (control)
#   -ub 8 -> volta2 uses MMQ, pristine uses MMVQ -> expect a difference if C5 works on Q4_K
set -uo pipefail
M=/root/llm/models/Qwen3.8-27B-TurboFCFusion-gguf/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf
export CUDA_VISIBLE_DEVICES=0,1

echo "=== model ==="
ls -la "$M"
echo "=== gpu before ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo ""

# interleave the two variants to cancel drift
for UB in 4 8; do
  for v in pristine volta2; do
    ROW=$(LD_LIBRARY_PATH=/root/libdir-$v /root/libdir-$v/llama-bench \
      -m "$M" -p 512 -n 32 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 \
      --tensor-split 1/1 -b "$UB" -ub "$UB" -r 3 2>&1 | grep -F "pp512")
    printf "UB=%-2s %-8s %s\n" "$UB" "$v" "$ROW"
  done
done
echo ""
echo "=== lib md5 (must differ) ==="
md5sum /root/libdir-pristine/libggml-cuda.so.0.24.0 /root/libdir-volta2/libggml-cuda.so.0.24.0
echo ""
echo "=== gpu after ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo Q4KM_C5_DONE
