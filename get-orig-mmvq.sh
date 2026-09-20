#!/bin/bash
# Extract the original (b11053, pre-C4) mmvq.cu from git into the study dir.
cd /mnt/f/vllm+llama.cpp/llama.cpp
git show HEAD:ggml/src/ggml-cuda/mmvq.cu > /tmp/mmvq-orig.cu
cp /tmp/mmvq-orig.cu /mnt/f/vllm+llama.cpp/1cat-vllm-v100-study/mmvq-orig.cu
OUT=/mnt/f/vllm+llama.cpp/1cat-vllm-v100-study/mmvq-orig.cu
echo "lines: $(wc -l < "$OUT")"
echo "VOLTA count (expect 0): $(grep -c MMVQ_PARAMETERS_VOLTA "$OUT")"
echo "=== diff (orig vs working-tree C4), first 45 lines ==="
diff "$OUT" ggml/src/ggml-cuda/mmvq.cu | head -45
echo DONE
