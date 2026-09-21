#!/bin/bash
# Launch the stress-test.sh for a variant in the background on the server.
# Usage: bash launch-stress.sh <VARIANT> <BENCH_BIN> [MODEL] [CONTEXT] [GEN] [GPUS] [CACHE_K] [CACHE_V]
set -uo pipefail
VARIANT=${1:-orig}
BENCH_BIN=${2:-/root/llm/test/v100-opt/llama.cpp/build/bin/llama-bench}
MODEL=${3:-/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf}
CONTEXT=${4:-262144}
GEN=${5:-128}
GPUS=${6:-2,5}
CACHE_K=${7:-f16}
CACHE_V=${8:-f16}
nohup env VARIANT="$VARIANT" BENCH_BIN="$BENCH_BIN" MODEL="$MODEL" CONTEXT="$CONTEXT" GEN="$GEN" GPUS="$GPUS" \
  CACHE_K="$CACHE_K" CACHE_V="$CACHE_V" \
  bash /root/stress-test.sh > "/tmp/stress-${VARIANT}.out" 2>&1 &
echo "launched $VARIANT PID=$! log=/tmp/stress-${VARIANT}.out (cache k=$CACHE_K v=$CACHE_V)"
echo LAUNCH_DONE
