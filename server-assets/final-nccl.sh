#!/bin/bash
# final-nccl.sh -- formal deliverable runs, using the ORIGINAL harness /root/p60-ab-harness.sh
# (same script as the baseline run), with drop_caches, for both all-reduce implementations.
set -u
PKGLIB=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/lib
H=/root/p60-ab-harness.sh

echo "=== waiting for NCCL env sweep to finish ==="
for i in $(seq 1 160); do
  grep -q NCCL_ENV_SWEEP_DONE /tmp/nccl-env-sweep.log 2>/dev/null && { echo "env sweep done"; break; }
  sleep 15
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# make the ORIGINAL harness (no LDEXTRA knob) able to load libnccl.so.2 from the snapshot dir
ln -sf "$PKGLIB/libnccl.so.2" /root/libdir-nccl/libnccl.so.2
ls -l /root/libdir-nccl/libnccl.so.2

run() { # <tag> <libdir> <split> <port>
  echo "##### FINAL ARM $1 libdir=$2 split=$3"
  env TAG="$1" L="$2" CARDS=0,1,2 SPLIT="$3" PORT="$4" P2P=1 \
    bash "$H" 2>&1 | tee -a /tmp/final-nccl-summary.log | tail -30
}

run final-bf-tp3   /root/libdir-rt   tensor 8160
run final-nccl-tp3 /root/libdir-nccl tensor 8161

echo "=== greedy hash comparison ==="
for t in final-bf-tp3 final-nccl-tp3; do
  printf '%-16s ' "$t"; grep -a 'greedy:' /tmp/p60-$t.log | tail -1
done
echo "=== medians ==="
for t in final-bf-tp3 final-nccl-tp3; do
  printf '%-16s ' "$t"; grep -a -E "^prompt[123]: |^MEDIAN_TG=" /tmp/p60-$t.log | tr '\n' ' '; echo
done
echo FINAL_NCCL_DONE
