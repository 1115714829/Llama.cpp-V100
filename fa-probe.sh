#!/bin/bash
# Probe: Path A FA/prefill numbers + trait sizes. Evidence only, no A/B claim.
set -uo pipefail
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
export CUDA_VISIBLE_DEVICES=0,1,2
export GGML_CUDA_P2P=1
export L=/root/libdir-pathb
export LD_LIBRARY_PATH=/root/libdir-pathb
export GGML_GALLOCR_SLOTS=3
unset GGML_META_FULLGRAPH LLAMA_SM70_Q8_DIRECT LLAMA_SM70_FA_GEMM LLAMA_SM70_REGP
M=/mnt/3.84t/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
LOG=/tmp/fa-probe.log
{
  echo "PROBE_START $(date)"
  echo "MARK_FA_GEMM=$(strings ${L}/libggml-cuda.so | grep -c LLAMA_SM70_FA_GEMM || true)"
  # Path A stock (FA_GEMM off) - real numbers
  "${L}/llama-bench" -m "${M}" -p 8192 -n 0 -r 2 -ub 2048 -ts 1/1/1 \
    --flash-attn 1 -ctk q8_0 -ctv q8_0
  "${L}/llama-bench" -m "${M}" -p 32768 -n 0 -r 2 -ub 2048 -ts 1/1/1 \
    --flash-attn 1 -ctk q8_0 -ctv q8_0
  echo "PROBE_END $(date)"
} > "${LOG}" 2>&1
echo "LOG=${LOG}"
tail -20 "${LOG}"
