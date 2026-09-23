#!/bin/bash
# T2 numerics jury: perplexity closeness between Path A and decomp on the same
# corpus (<=0.1% relative diff = summation noise only; more = real bug).
set -u
M=/mnt/3.84t/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
run() {
  dec=$1
  env LLAMA_SM70_FA_DECOMP=$dec GGML_KV_BUCKET_RATIO=1.08 \
    LD_LIBRARY_PATH=/root/libdir-gb CUDA_VISIBLE_DEVICES=0,1,2 GGML_GALLOCR_SLOTS=3 \
    /root/libdir-gb/llama-perplexity -m "$M" -f /root/llm/test/bl-prompt90.txt \
    -b 512 -ub 512 -ctk q8_0 -ctv q8_0 --flash-attn on \
    --split-mode tensor --tensor-split 1,1,1 -ngl 999 > /tmp/ppl-$dec.log 2>&1
  echo "rc=$?"
  grep -aE 'Final estimate|PPL' /tmp/ppl-$dec.log
}
echo === PPL decomp=0 ===
run 0
echo === PPL decomp=1 ===
run 1
echo PPL_DONE
