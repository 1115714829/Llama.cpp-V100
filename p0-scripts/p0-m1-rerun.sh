#!/bin/bash
# M1 rerun at -ub 512 (R282 lesson: long-context needs ub<=512 + limited slots).
# First attempt (ub 2048) OOM'd at prompt warmup (res=-2) on ALL four arms; no data
# was observed, so this caliber change is pre-registration-clean (see P0-PREREG addendum).
set -u
MODEL=/mnt/3.84t/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
BIN=/root/llm/test/v100-opt/llama.cpp/build-instr/bin/llama-bench
LIBS=/root/libdir-t8b
LOGDIR=/root/llm/test/p0-logs
export LD_LIBRARY_PATH=$LIBS
export GGML_CUDA_P2P=1
export GGML_GALLOCR_SLOTS=3
export LLAMA_SM70_D256=1
S=$LOGDIR/m1rerun-summary.log
{
  echo "=== M1 rerun start $(date -Is) ==="
  echo "caliber: -p 131072 -n 0 -r 3 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0; env same as sweep"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
} | tee "$S"

arm() {
  name=$1; cvd=$2; ts=$3
  echo "=== ARM $name $(date -Is) CVD=$cvd TS=$ts ===" | tee -a "$S"
  T0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$cvd "$BIN" -m "$MODEL" -p 131072 -n 0 -r 3 -b 512 -ub 512 \
    -fa on -ctk q8_0 -ctv q8_0 --tensor-split "$ts" --main-gpu 0 \
    > "$LOGDIR/$name.log" 2>&1
  rc=$?; T1=$(date +%s)
  echo "--- $name rc=$rc wall=$((T1-T0))s" | tee -a "$S"
  grep -E '(pp|tg)[0-9]+' "$LOGDIR/$name.log" | tail -2 | tee -a "$S"
}

arm m1r-3a 0,1,2   1/1/1
arm m1r-4a 0,1,2,3 1/1/1/1
arm m1r-4b 0,1,2,3 1/1/1/1
arm m1r-3b 0,1,2   1/1/1

{
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  echo "=== M1 rerun done $(date -Is) ==="
  echo M1R_DONE
} | tee -a "$S"
