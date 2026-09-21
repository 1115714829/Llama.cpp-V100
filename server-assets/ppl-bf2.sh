#!/bin/bash
# ppl-bf2.sh -- rerun the non-NCCL perplexity arm with a complete libdir snapshot.
set -u
B=/root/llm/test/v100-opt/llama.cpp/build/bin
L=/root/libdir-bf-ppl
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
C=/root/ppl-corpus.txt

mkdir -p "$L"
cp -a "$B"/. "$L"/ 2>/dev/null
echo "=== snapshot md5s (bf arm must be the butterfly build) ==="
md5sum "$L/libggml-cuda.so.0.24.0" "$L/libllama.so.0.4.1" "$L/libllama-common.so.0.4.1"
ls -l "$L/llama-perplexity"

env CUDA_VISIBLE_DEVICES=0,1,2 LD_LIBRARY_PATH="$L" GGML_CUDA_P2P=1 \
  "$L/llama-perplexity" -m "$M" -f "$C" -c 512 --chunks 64 -ngl 999 \
    --split-mode tensor --tensor-split 1,1,1 --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 > /tmp/ppl-bf2.log 2>&1
echo "exit=$?"
grep -aE 'Final estimate' /tmp/ppl-bf2.log || tail -5 /tmp/ppl-bf2.log
echo PPL_BF2_DONE
