#!/bin/bash
# P0: KV dtype A/B at long context -- q8_0 (our current) vs f16 (1cat's long-ctx choice)
# Evidence: 1cat docs/design/sm70_qwen38_130_acceptance.md:45  (FP16 KV much faster at 128K/256K)
# NOTE: llama-bench -ts uses SLASH syntax (1/1/1). Comma (1,1,1) aborts at load.
set -u
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
L=/root/libdir-nccl
export GGML_CUDA_P2P=1
BASE="-m $M -ngl 999 -fa on -sm tensor -ts 1/1/1 -r 2 -o md"
echo "=== P0 KV dtype A/B start $(date) ==="
for CTX in 32768 65536; do
  for KV in q8_0 f16; do
    echo ""
    echo "########## ctx=$CTX  kv=$KV ##########"
    echo "--- prefill pp$CTX ---"
    LD_LIBRARY_PATH=$L CUDA_VISIBLE_DEVICES=0,1,2 $L/llama-bench $BASE -p $CTX -n 8 -d 0 -ctk $KV -ctv $KV 2>&1 | grep -a -e "pp" -e "tg" -e error | tail -4
    echo "--- decode tg128 at depth $CTX ---"
    LD_LIBRARY_PATH=$L CUDA_VISIBLE_DEVICES=0,1,2 $L/llama-bench $BASE -p 8 -n 128 -d $CTX -ctk $KV -ctv $KV 2>&1 | grep -a -e "pp" -e "tg" -e error | tail -4
    echo "--- gpu mem ---"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -3
  done
done
echo "P0_ALL_DONE $(date)"
