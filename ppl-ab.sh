#!/bin/bash
# ppl-ab.sh -- target-model perplexity under old all-reduce (butterfly) vs new (NCCL).
# Same corpus, same settings; only the all-reduce implementation differs.
set -u
PKGLIB=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/lib
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
C=/root/ppl-corpus.txt
echo "corpus: $(md5sum "$C" | awk '{print $1}')  bytes=$(wc -c < "$C")"

run() { # <tag> <libdir> <ldextra>
  echo "##### PPL ARM $1  libdir=$2  ldextra=$3"
  env CUDA_VISIBLE_DEVICES=0,1,2 LD_LIBRARY_PATH="$2${3:+:$3}" GGML_CUDA_P2P=1 \
    "$2/llama-perplexity" -m "$M" -f "$C" -c 512 --chunks 64 -ngl 999 \
      --split-mode tensor --tensor-split 1,1,1 --flash-attn on \
      --cache-type-k q8_0 --cache-type-v q8_0 > /tmp/ppl-$1.log 2>&1
  echo "exit=$?  $(md5sum "$2/libggml-cuda.so.0.24.0" | awk '{print $1}') libggml-cuda"
  grep -aE "Final estimate|PPL" /tmp/ppl-$1.log | tail -3
  grep -aiE "NCCL|AllReduce" /tmp/ppl-$1.log | head -3
}

run bf   /root/libdir-rt   ""
run nccl /root/libdir-nccl "$PKGLIB"
echo PPL_AB_DONE
