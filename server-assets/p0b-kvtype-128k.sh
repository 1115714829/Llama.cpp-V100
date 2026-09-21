#!/bin/bash
# P0b: KV dtype A/B at 128K (the depth where 1cat's own table shows the biggest KV effect)
set -u
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
L=/root/libdir-nccl
export GGML_CUDA_P2P=1
BASE=-m $M -ngl 999 -fa on -sm tensor -ts 1/1/1 -r 2 -o md
echo === P0b 128K start ===; date
for KV in q8_0 f16; do
  echo ########## ctx=131072 kv=$KV ##########
  LD_LIBRARY_PATH=$L CUDA_VISIBLE_DEVICES=0,1,2 $L/llama-bench $BASE -p 131072 -n 8 -d 0 -ctk $KV -ctv $KV 2>&1 | grep -a -e pp -e tg -e error | tail -3
  echo --- decode at depth 131072 ---
  LD_LIBRARY_PATH=$L CUDA_VISIBLE_DEVICES=0,1,2 $L/llama-bench $BASE -p 8 -n 128 -d 131072 -ctk $KV -ctv $KV 2>&1 | grep -a -e pp -e tg -e error | tail -3
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -3
done
echo P0B_ALL_DONE; date