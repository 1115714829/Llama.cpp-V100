#!/bin/bash
# Path B A/B: staged vs q8-direct. Single variable = LLAMA_SM70_Q8_DIRECT.
# Usage: bash /root/pathb-ab.sh <TAG> <PP> <REPS> <Q8DIRECT 0|1> [NODROP=1]
set -uo pipefail
TAG="${1:?tag}"
PP="${2:-32768}"
REPS="${3:-2}"
Q8="${4:-1}"
NODROP="${5:-0}"
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/root/libdir-pathb:/usr/local/cuda-12.4/bin
export CUDA_VISIBLE_DEVICES=0,1,2
export GGML_CUDA_P2P=1
# Long-context: default 8-slot gallocr OOMs at >=32K (R282). Limit slots.
export GGML_GALLOCR_SLOTS=3
export L=/root/libdir-pathb
export LD_LIBRARY_PATH=/root/libdir-pathb
export LLAMA_SM70_Q8_DIRECT="${Q8}"
# Optional: LLAMA_SM70_FA_GEMM / LLAMA_SM70_REGP from parent env (A/B arms).
# Default both unset = Path A stock.
unset GGML_META_FULLGRAPH
unset LLAMA_SM70_D256_Q4_DIRECT
# D256 stays default ON (Path A). Q8_DIRECT is the only A/B variable.
M=/mnt/3.84t/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
LOG=/tmp/pathb-ab-"${TAG}".log
if [ "${NODROP}" != "1" ]; then
  sync
  echo 3 > /proc/sys/vm/drop_caches
fi
{
  echo "AB_START $(date) TAG=${TAG} PP=${PP} REPS=${REPS} Q8_DIRECT=${Q8} NODROP=${NODROP}"
  echo "LIB_MD5 $(md5sum "${L}"/libggml-cuda.so.0.24.0 | awk '{print $1}')"
  echo "MARK_Q8=$(strings "${L}"/libggml-cuda.so | grep -c LLAMA_SM70_Q8_DIRECT || true)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  "${L}/llama-bench" -m "${M}" \
      -p "${PP}" -n 0 -r "${REPS}" -ub 2048 -ts 1/1/1 \
      --flash-attn 1 -ctk q8_0 -ctv q8_0
  echo "AB_END $(date)"
} > "${LOG}" 2>&1
echo "LOG=${LOG}"
tail -30 "${LOG}"
