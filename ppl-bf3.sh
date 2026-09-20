#!/bin/bash
# ppl-bf3.sh -- non-NCCL perplexity arm.  llama-perplexity is a ~210 KB launcher shell; the
# kernels/comm live in libggml-cuda.so.  So reuse the launcher and point LD_LIBRARY_PATH at the
# butterfly libs (ed1b2f7f), which is what defines the numerics.
set -u
L=/root/libdir-bf-ppl
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
C=/root/ppl-corpus.txt

cp -f /root/libdir-nccl/llama-perplexity "$L/llama-perplexity"
echo "=== launcher md5: reuse vs original source ==="
md5sum /root/libdir-nccl/llama-perplexity "$L/llama-perplexity"
echo "=== bf libs (must be ed1b2f7f = butterfly build) ==="
md5sum "$L/libggml-cuda.so.0.24.0"

env CUDA_VISIBLE_DEVICES=0,1,2 LD_LIBRARY_PATH="$L" GGML_CUDA_P2P=1 \
  "$L/llama-perplexity" -m "$M" -f "$C" -c 512 --chunks 64 -ngl 999 \
    --split-mode tensor --tensor-split 1,1,1 --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 > /tmp/ppl-bf3.log 2>&1
echo "exit=$?"
grep -aE 'Final estimate' /tmp/ppl-bf3.log || tail -6 /tmp/ppl-bf3.log
echo "=== confirm which allreduce the bf arm used ==="
grep -aiE 'NCCL|AllReduce' /tmp/ppl-bf3.log | head -3
echo PPL_BF3_DONE
