#!/bin/bash
# T2 numerics jury + block-size invariance bisect: the decomposed merge math
# must be block-size independent. PPL(8192) == PPL(24576) == PPL(65536) =>
# block-invariant bug (geometry/scale); divergent => merge math bug.
set -u
M=/mnt/3.84t/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
run() {
  dec=$1
  kbn=$2
  env LLAMA_SM70_FA_DECOMP=$dec GGML_KV_BUCKET_RATIO=1.08 GGML_DECOMP_KBN=$kbn \
    LD_LIBRARY_PATH=/root/libdir-gb CUDA_VISIBLE_DEVICES=0,1,2 GGML_GALLOCR_SLOTS=3 \
    /root/libdir-gb/llama-perplexity -m "$M" -f /root/llm/test/bl-prompt90.txt \
    -b 512 -ub 512 -ctk q8_0 -ctv q8_0 --flash-attn on \
    --split-mode tensor --tensor-split 1,1,1 -ngl 999 > /tmp/ppl-$dec-$kbn.log 2>&1
  echo "rc=$? decomp=$dec kbn=$kbn"
  grep -aE 'Final estimate' /tmp/ppl-$dec-$kbn.log
}
echo === PPL decomp=0 auto ===
run 0 0
echo === PPL decomp=1 kbn=8192 ===
run 1 8192
echo === PPL decomp=1 kbn=24576 ===
run 1 24576
echo === PPL decomp=1 kbn=65536 ===
run 1 65536
echo PPL_DONE
