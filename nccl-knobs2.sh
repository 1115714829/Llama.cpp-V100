#!/bin/bash
# nccl-knobs2.sh -- round 2: the small-message latency knobs from the NCCL env reference that
# round 1 did not cover. Our allreduce tensors are ~160 KB (5120x8) and ~557 KB (17408x8) FP32,
# so the interesting question is which protocol/threshold/channel-count NCCL picks for them.
set -u
PKGLIB=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/lib
H=/root/p60-ab-harness2.sh

echo "=== waiting for the formal run to be done ==="
for i in $(seq 1 200); do
  grep -q FINAL_NCCL_DONE /tmp/final-nccl.log 2>/dev/null && { echo "formal run done"; break; }
  sleep 15
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

sweep() { # <tag> [ENV=VAL ...]
  tag=$1; shift
  echo "##### KNOB ARM $tag env='$*'"
  ( export "$@"; env NODROP=1 TAG="$tag" L=/root/libdir-nccl CARDS=0,1,2 SPLIT=tensor PORT=8150 P2P=1 bash "$H" ) 2>&1 | tail -18
}

sweep k-llthr      NCCL_LL_THRESHOLD=1048576
sweep k-singlering NCCL_SINGLE_RING_THRESHOLD=1048576
sweep k-ctas1      NCCL_MIN_CTAS=1 NCCL_MAX_CTAS=1
sweep k-buf512k    NCCL_BUFFSIZE=524288
sweep k-ctazero    NCCL_CTA_POLICY=ZERO
sweep k-nthr128    NCCL_NTHREADS=128
sweep k-tree       NCCL_ALGO=Tree
sweep k-combo      NCCL_ALGO=Ring NCCL_PROTO=LL,Simple NCCL_SINGLE_RING_THRESHOLD=1048576 NCCL_BUFFSIZE=524288 NCCL_NTHREADS=128
sweep k-combo2     NCCL_LL_THRESHOLD=1048576 NCCL_SINGLE_RING_THRESHOLD=1048576 NCCL_MIN_CTAS=1 NCCL_MAX_CTAS=1 NCCL_NTHREADS=128

echo "=== knob medians ==="
for t in k-llthr k-singlering k-ctas1 k-buf512k k-ctazero k-nthr128 k-tree k-combo k-combo2; do
  printf '%-13s ' "$t"
  grep -a -E "^prompt[123]: |^MEDIAN_TG=" /tmp/p60-$t.log 2>/dev/null | tr '\n' ' '
  echo
done
echo NCCL_KNOBS2_DONE
