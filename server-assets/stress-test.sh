#!/bin/bash
# llama-stress-test: standard parameterized llama.cpp stress test (single/multi-GPU).
# Measures TTFT (prefill), decode (tg), and prefill wall time at a chosen context, for one build (variant).
# Run on the GPU server. All inputs are env vars (see SKILL.md).
set -uo pipefail

MODEL=${MODEL:?MODEL (gguf path) is required}
CONTEXT=${CONTEXT:-262144}
GEN=${GEN:-128}
GPUS=${GPUS:-2,5}
BENCH_BIN=${BENCH_BIN:?BENCH_BIN (llama-bench path) is required}
# llama-bench tensor split uses slash syntax (1/1 = even across 2 GPUs).
TENSOR_SPLIT=${TENSOR_SPLIT:-1/1}
MAIN_GPU=${MAIN_GPU:-0}
UBATCH=${UBATCH:-512}
VARIANT=${VARIANT:-run}
USE_MTP=${USE_MTP:-0}
MTP_DRAFT=${MTP_DRAFT:-}
DFLASH2=${DFLASH2:-0}
# KV cache type: f16 (default) or q8_0/q4_0 (smaller, lets larger contexts fit).
CACHE_K=${CACHE_K:-f16}
CACHE_V=${CACHE_V:-f16}

LOG=/tmp/stress-${VARIANT}.log
echo "=== llama-stress-test: variant=$VARIANT ==="
echo "model:        $MODEL"
echo "context:      $CONTEXT   gen: $GEN"
echo "gpus:         $GPUS   tensor-split: $TENSOR_SPLIT   main-gpu: $MAIN_GPU   ubatch: $UBATCH"
echo "cache:        k=$CACHE_K v=$CACHE_V"
echo "bench bin:    $BENCH_BIN"
echo "----------------------------------------"

if [ ! -x "$BENCH_BIN" ]; then
  echo "ERROR: BENCH_BIN not found / not executable: $BENCH_BIN"; exit 1
fi
if [ ! -f "$MODEL" ]; then
  echo "ERROR: MODEL not found: $MODEL"; exit 1
fi

# Speculative-decode args.
SPEC_ARGS=""
if [ "$USE_MTP" = "1" ]; then
  if [ -z "$MTP_DRAFT" ]; then
    echo "ERROR: USE_MTP=1 but MTP_DRAFT is empty"; exit 1
  fi
  if [ ! -f "$MTP_DRAFT" ]; then
    echo "ERROR: MTP_DRAFT not found: $MTP_DRAFT"; exit 1
  fi
  SPEC_ARGS="--model-draft $MTP_DRAFT"
  echo "MTP draft:    $MTP_DRAFT"
elif [ "$DFLASH2" = "1" ]; then
  echo "NOTE: DFlash2 is a 1cat-vLLM feature (not in llama.cpp); interface reserved, skipping."
fi

export CUDA_VISIBLE_DEVICES=$GPUS
echo "=== running llama-bench  -p $CONTEXT -n $GEN ==="
echo "(the 256k prefill is long; this can take ~7-11 min on V100)"
T0=$(date +%s)
$BENCH_BIN -m "$MODEL" -p "$CONTEXT" -n "$GEN" -ngl -1 -ub "$UBATCH" \
  -ctk "$CACHE_K" -ctv "$CACHE_V" \
  --tensor-split "$TENSOR_SPLIT" --main-gpu "$MAIN_GPU" $SPEC_ARGS 2>&1 | tee "$LOG"
T1=$(date +%s)
WALL=$((T1 - T0))

echo "=========================================="
echo "=== summary: variant=$VARIANT  wall=${WALL}s ==="
echo "raw pp/tg lines:"
grep -E "pp${CONTEXT}|tg${GEN}" "$LOG" | tail -4
# The t/s is the last numeric field in the table row.
PP_LINE=$(grep -E "pp${CONTEXT}[^0-9]" "$LOG" | tail -1)
if [ -n "$PP_LINE" ]; then
  PP=$(echo "$PP_LINE" | grep -oE '[0-9]+(\.[0-9]+)?' | tail -1)
  if [ -n "${PP:-}" ]; then
    TTFT=$(awk -v c="$CONTEXT" -v p="$PP" 'BEGIN{ if (p+0>0) printf "%.1f", c/p; else print "n/a" }')
    echo "TTFT: prefill $CONTEXT tok @ ${PP} tok/s = ${TTFT}s"
  else
    echo "pp line: $PP_LINE"
  fi
fi
TG_LINE=$(grep -E "tg${GEN}[^0-9]" "$LOG" | tail -1)
if [ -n "$TG_LINE" ]; then
  TG=$(echo "$TG_LINE" | grep -oE '[0-9]+(\.[0-9]+)?' | tail -1)
  echo "decode: tg${GEN} = ${TG:-n/a} tok/s"
fi
echo STRESS_DONE
