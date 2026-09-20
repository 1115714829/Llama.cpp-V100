#!/bin/bash
# nccl-ab.sh -- wait for the NCCL build to land, then A/B the all-reduce implementations.
# NODROP=1 for iteration speed (page-cache hot load ~15 s). Confirmatory runs use drop_caches.
set -u
PKGLIB=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/lib
H=/root/p60-ab-harness2.sh
echo "=== waiting for NCCL build ==="
for i in $(seq 1 240); do
  if grep -q BUILD_NCCL_DONE /tmp/build-nccl.log 2>/dev/null; then echo "build done after ${i} polls"; break; fi
  sleep 15
done
grep -q BUILD_NCCL_DONE /tmp/build-nccl.log || { echo "BUILD_NEVER_FINISHED"; grep -a "error\|Error" /tmp/build-nccl.log | tail -20; exit 1; }
echo "=== nccl linkage of built lib ==="
ldd /root/libdir-nccl/libggml-cuda.so.0.24.0
ls -l /root/libdir-nccl/llama-perplexity /root/libdir-nccl/llama-server 2>&1
echo "=== free gpu ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

run() {
  # run <tag> <libdir> <ldextra> <cards> <split> <port>
  echo "##### ARM $1 libdir=$2 ldextra=$3 cards=$4 split=$5"
  env NODROP=1 TAG="$1" L="$2" LDEXTRA="$3" CARDS="$4" SPLIT="$5" PORT="$6" P2P=1 \
    bash "$H" 2>&1 | tee -a /tmp/nccl-ab-summary.log | tail -25
}

run ab-bf-tp3   /root/libdir-rt   ""       0,1,2      tensor 8131
run ab-nccl-tp3 /root/libdir-nccl "$PKGLIB" 0,1,2      tensor 8132
run ab-nccl-tp4 /root/libdir-nccl "$PKGLIB" 0,1,2,3    tensor 8133
run ab-nccl-tp6 /root/libdir-nccl "$PKGLIB" 0,1,2,3,4,5 tensor 8134
run ab-bf-layer /root/libdir-rt   ""       0,1,2      layer  8135
run ab-nccl-layer /root/libdir-nccl "$PKGLIB" 0,1,2    layer  8136

echo "=== summary ==="
grep -a -E "^##### ARM|MEDIAN_TG=|prompt[123]: |allreduce init|NCCL|target decode\+sync|spec timing|health ok" /tmp/nccl-ab-summary.log | tail -120
echo NCCL_AB_DONE
