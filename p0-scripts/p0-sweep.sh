#!/bin/bash
# P-P0 campaign: prefill TP3/4/6 scaling sweep (R223 gap) + M1 KV-head imbalance anchor.
# Only variable across arms = -ts (and the matching CUDA_VISIBLE_DEVICES).
# Stack: /root/libdir-t8b (= B5 adopted chain, md5s recorded below), env constant.
# NODROP diagnostic caliber (page-cache hot), per ledger precedent (R273/R275/R276).
set -u

MODEL=/mnt/3.84t/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
BIN=/root/llm/test/v100-opt/llama.cpp/build-instr/bin/llama-bench
LIBS=/root/libdir-t8b
LOGDIR=/root/llm/test/p0-logs
mkdir -p "$LOGDIR"

export LD_LIBRARY_PATH=$LIBS
export GGML_CUDA_P2P=1
export GGML_GALLOCR_SLOTS=3
export LLAMA_SM70_D256=1

S=$LOGDIR/summary.log
{
  echo "=== P0 sweep start $(date -Is) ==="
  echo "model=$MODEL"
  echo "bin=$BIN libs=$LIBS"
  echo "env: GGML_CUDA_P2P=1 GGML_GALLOCR_SLOTS=3 LLAMA_SM70_D256=1 (constant, not variables)"
  echo "--- lib provenance (md5) ---"
  md5sum $LIBS/libggml-cuda.so.0.24.0 $LIBS/libggml-base.so.0.24.0 $LIBS/libllama.so.0.4.1 $LIBS/libllama-common.so.0.4.1
  echo "--- binary provenance ---"
  ls -la "$BIN"
  "$BIN" --version 2>&1 | head -3
  echo "--- gpu state before ---"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
} | tee "$S"

arm() {
  name=$1; cvd=$2; ts=$3; pres=$4; reps=$5
  echo "=== ARM $name $(date -Is) CVD=$cvd TS=$ts P=$pres R=$reps ===" | tee -a "$S"
  T0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$cvd "$BIN" -m "$MODEL" -p "$pres" -n 0 -r "$reps" -b 2048 -ub 2048 \
    -fa on -ctk q8_0 -ctv q8_0 --tensor-split "$ts" --main-gpu 0 \
    > "$LOGDIR/$name.log" 2>&1
  rc=$?
  T1=$(date +%s)
  echo "--- $name rc=$rc wall=$((T1-T0))s" | tee -a "$S"
  grep -E '(pp|tg)[0-9]+' "$LOGDIR/$name.log" | tail -4 | tee -a "$S"
}

# (a) R223 scaling sweep: pp8192/pp32768, -r 8 per ledger rule. Mirror order.
arm sweep3a 0,1,2       1/1/1        8192,32768 8
arm sweep4a 0,1,2,3     1/1/1/1      8192,32768 8
arm sweep6a 0,1,2,3,4,5 1/1/1/1/1/1  8192,32768 8
arm sweep6b 0,1,2,3,4,5 1/1/1/1/1/1  8192,32768 8
arm sweep4b 0,1,2,3     1/1/1/1      8192,32768 8
arm sweep3b 0,1,2       1/1/1        8192,32768 8

# (b) M1 imbalance anchor: pp131072 (attention-dominant point), TP3 vs TP4,
# -r 3 + mirror (long-context caliber, R272 precedent; note deviation from -r>=8).
arm m1-3a 0,1,2   1/1/1    131072 3
arm m1-4a 0,1,2,3 1/1/1/1  131072 3
arm m1-4b 0,1,2,3 1/1/1/1  131072 3
arm m1-3b 0,1,2   1/1/1    131072 3

{
  echo "--- gpu state after ---"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  echo "=== all result rows ==="
  grep -E '(pp|tg)[0-9]+' $LOGDIR/sweep*.log $LOGDIR/m1-*.log
  echo "=== P0 sweep done $(date -Is) ==="
  echo P0_DONE
} | tee -a "$S"
