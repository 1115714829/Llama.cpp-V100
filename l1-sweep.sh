#!/bin/bash
# L1 sweep: find where MMVQ (GEMV) should hand over to MMQ on V100, per quant type.
# V100 dispatch (verified): ne11<=8 -> mmvq ; 9..63 -> MMQ ; >=64 -> FP16 MMA (1cat-sm70-gemm-tactics.md 5.3.1)
# We vary the ubatch (-ub), which sets the per-kernel ne11 during prefill.
# Usage: LIBS="pristine c4" bash /root/l1-sweep.sh
set -uo pipefail

M=${MODEL:-/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf}
GPUS=${GPUS:-0}
REPS=${REPS:-3}
LOG=/tmp/l1-sweep.log
: > "$LOG"
export CUDA_VISIBLE_DEVICES=$GPUS

echo "=== L1 mmvq->mmq crossover sweep  $(date -Is) ===" | tee -a "$LOG"
echo "model: $M" | tee -a "$LOG"
echo "gpus:  $GPUS   reps: $REPS" | tee -a "$LOG"
echo "sweep -ub (sets ne11 in prefill mm): 2 4 6 8 10 12 16 24 32 48 64" | tee -a "$LOG"
echo "" | tee -a "$LOG"

for v in ${LIBS:-pristine c4}; do
  BIN=/root/libdir-$v/llama-bench
  if [ ! -x "$BIN" ]; then echo "MISSING $BIN"; continue; fi
  echo "########## variant=$v  $(date -Is) ##########" | tee -a "$LOG"
  echo "--- lib md5 (must DIFFER between variants) ---" | tee -a "$LOG"
  md5sum "/root/libdir-$v/libggml-cuda.so.0.24.0" | tee -a "$LOG"
  "$BIN" --version 2>&1 | grep -i version | tee -a "$LOG"

  for UB in 2 4 6 8 10 12 16 24 32 48 64; do
    OUT=$(LD_LIBRARY_PATH=/root/libdir-$v "$BIN" \
      -m "$M" -p 512 -n 32 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 \
      -b "$UB" -ub "$UB" -r "$REPS" 2>&1)
    PP=$(echo "$OUT" | grep -F "pp512" | awk -F'|' '{print $NF}' | tr -d ' ')
    TG=$(echo "$OUT" | grep -F "tg32"  | awk -F'|' '{print $NF}' | tr -d ' ')
    echo "SWEEP variant=$v ub=$UB pp512=\"$PP\" tg32=\"$TG\"" | tee -a "$LOG"
  done
  echo "" | tee -a "$LOG"
done

echo "=== done $(date -Is) ===" | tee -a "$LOG"
grep -F "SWEEP " "$LOG"
echo L1_SWEEP_DONE
