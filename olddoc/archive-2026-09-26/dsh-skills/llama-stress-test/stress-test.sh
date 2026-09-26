#!/bin/bash
# llama-stress-test: standard parameterized llama.cpp stress test (1..6 GPU).
# Measures TTFT (prefill), decode (tg) and prefill wall time at a chosen context, for one build (variant).
# Run on the GPU server. All inputs are env vars (see SKILL.md).
set -uo pipefail

MODEL=${MODEL:?MODEL (gguf path) is required}
CONTEXT=${CONTEXT:-262144}
GEN=${GEN:-128}
GPUS=${GPUS:-2,5}
BENCH_BIN=${BENCH_BIN:?BENCH_BIN (llama-bench path) is required}
# llama-bench tensor split uses SLASH syntax (1/1 = even across 2 GPUs).
TENSOR_SPLIT=${TENSOR_SPLIT:-1/1}
MAIN_GPU=${MAIN_GPU:-0}
UBATCH=${UBATCH:-512}
VARIANT=${VARIANT:-run}
REPS=${REPS:-1}
FLASH_ATTN=${FLASH_ATTN:-1}
CACHE_K=${CACHE_K:-q8_0}
CACHE_V=${CACHE_V:-q8_0}

FA_ARGS=""
if [ "$FLASH_ATTN" = "1" ]; then FA_ARGS="-fa on"; fi

LOG=/tmp/stress-${VARIANT}.log
: > "$LOG"

{
echo "=== llama-stress-test: variant=$VARIANT ==="
echo "date:         $(date -Is)"
echo "model:        $MODEL"
echo "context:      $CONTEXT   gen: $GEN   reps: $REPS"
echo "gpus:         $GPUS   tensor-split: $TENSOR_SPLIT   main-gpu: $MAIN_GPU   ubatch: $UBATCH"
echo "flash-attn:   $FA_ARGS   cache: k=$CACHE_K v=$CACHE_V"
echo "bench bin:    $BENCH_BIN"
echo "--- binary provenance (record this! A/B must be same-commit) ---"
ls -la "$BENCH_BIN"
"$BENCH_BIN" --version 2>&1 | grep -iE "version|commit" | head -3
echo "--- gpu state before ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "----------------------------------------"
} | tee -a "$LOG"

if [ ! -x "$BENCH_BIN" ]; then
  echo "ERROR: BENCH_BIN not found / not executable: $BENCH_BIN"; exit 1
fi
if [ ! -f "$MODEL" ]; then
  echo "ERROR: MODEL not found: $MODEL"; exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPUS

rep=1
while [ "$rep" -le "$REPS" ]; do
  echo "=== rep $rep/$REPS : llama-bench -p $CONTEXT -n $GEN ===" | tee -a "$LOG"
  echo "(256k prefill is long; ~10-11 min per rep on 2x V100)" | tee -a "$LOG"
  T0=$(date +%s)
  "$BENCH_BIN" -m "$MODEL" -p "$CONTEXT" -n "$GEN" -ngl -1 -ub "$UBATCH" \
    -ctk "$CACHE_K" -ctv "$CACHE_V" $FA_ARGS \
    --tensor-split "$TENSOR_SPLIT" --main-gpu "$MAIN_GPU" 2>&1 | tee -a "$LOG"
  T1=$(date +%s)
  WALL=$((T1 - T0))

  PP=$(grep -E "pp${CONTEXT}[^0-9]" "$LOG" | tail -1 | grep -oE '[0-9]+(\.[0-9]+)?' | tail -1)
  TG=$(grep -E "tg${GEN}[^0-9]" "$LOG" | tail -1 | grep -oE '[0-9]+(\.[0-9]+)?' | tail -1)
  TTFT=$(awk -v c="$CONTEXT" -v p="${PP:-0}" 'BEGIN{ if (p+0>0) printf "%.1f", c/p; else print "n/a" }')
  echo "SUMMARY variant=$VARIANT rep=$rep pp=$PP tg=$TG ttft_s=$TTFT wall_s=$WALL" | tee -a "$LOG"
  rep=$((rep + 1))
done

echo "--- gpu state after ---" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "$LOG"
echo "=== all SUMMARY lines (copy these into the report) ==="
grep "SUMMARY " "$LOG"
echo "log: $LOG"
echo STRESS_DONE
