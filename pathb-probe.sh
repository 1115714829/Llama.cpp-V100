#!/bin/bash
# One-shot routing probe: confirm q8-direct accept string + run pp 32K once.
set -uo pipefail
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
export CUDA_VISIBLE_DEVICES=0,1,2
export GGML_CUDA_P2P=1
export L=/root/libdir-pathb
export LD_LIBRARY_PATH=/root/libdir-pathb
export LLAMA_SM70_Q8_DIRECT=1
export LLAMA_SM70_D256_DEBUG=1
unset GGML_META_FULLGRAPH
unset LLAMA_SM70_D256_Q4_DIRECT
M=/mnt/3.84t/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
LOG=/tmp/pathb-probe.log
{
  echo "PROBE_START $(date)"
  "${L}/llama-bench" -m "${M}" -p 32768 -n 0 -r 1 -ub 2048 -ts 1/1/1 \
    --flash-attn 1 -ctk q8_0 -ctv q8_0
  echo "PROBE_END $(date)"
} > "${LOG}" 2>&1
echo "LOG=${LOG}"
grep -E 'sm70-d256|ACCEPT|REJECT|pp32768|error' "${LOG}" | head -40
