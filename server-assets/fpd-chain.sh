#!/bin/bash
# Chain: install the FPD probe, rebuild, run combined probe arms. ASCII only.
set -u
TREE=/root/llm/test/v100-opt/llama.cpp
L=/root/libdir-instr
OUT=/tmp/fpd-chain.log
: > $OUT
log() { echo "$*" >> $OUT; }
log "CHAIN_START $(date +%H:%M:%S)"

busy() {
  pgrep -f 'llama-benc[h] -m' > /dev/null && return 0
  pgrep -f 'llama-serve[r] --model' > /dev/null && return 0
  pgrep -f 'cmake --buil[d]' > /dev/null && return 0
  pgrep -f 'nvcc' > /dev/null && return 0
  return 1
}

idle=0
for i in $(seq 1 180); do
  if busy; then idle=0; else idle=$((idle+1)); fi
  log "IDLE_WAIT i=$i idle=$idle $(date +%H:%M:%S)"
  if [ $idle -ge 2 ]; then break; fi
  sleep 20
done
if [ $idle -lt 2 ]; then log "ABORT_MACHINE_BUSY"; log "FPD_CHAIN_DONE"; exit 1; fi
log "MACHINE_IDLE $(date +%H:%M:%S)"

SRC=$TREE/ggml/src/ggml-backend.cpp
M1=$(md5sum /root/fpd/ggml-backend.cpp | cut -d' ' -f1)
M2=$(md5sum $SRC | cut -d' ' -f1)
log "SRC_MD5_BEFORE=$M2 UPLOAD_MD5=$M1"
if [ "$M1" != "$M2" ]; then cp /root/fpd/ggml-backend.cpp $SRC; log "SRC_SWAPPED"; fi
log "SRC_MD5_AFTER=$(md5sum $SRC | cut -d' ' -f1)"

cd $TREE || { log "ABORT_NO_TREE"; log "FPD_CHAIN_DONE"; exit 1; }
( while true; do sleep 60; echo "BUILD_HB $(date +%H:%M:%S) $(tail -1 /tmp/fpd-build.log 2>/dev/null)" >> $OUT; done ) &
HB=$!
cmake --build build-instr --config Release -j128 > /tmp/fpd-build.log 2>&1
RC=$?
kill $HB 2>/dev/null
log "BUILD_RC=$RC"
log "BUILD_ERRORS=$(grep -c -e error /tmp/fpd-build.log)"
log "BUILD_TARGETS=$(grep -c -e 'Built target' /tmp/fpd-build.log)"
log "BUILD_TAIL=$(tail -1 /tmp/fpd-build.log)"
if [ $RC -ne 0 ]; then log "ABORT_BUILD_FAILED"; grep -a -e error /tmp/fpd-build.log | head -20 >> $OUT; log "FPD_CHAIN_DONE"; exit 1; fi
log "SRC_MD5_POSTBUILD=$(md5sum $SRC | cut -d' ' -f1)"
cp -a build-instr/bin/. $L/
log "LIB_MD5_CUDA=$(md5sum $L/libggml-cuda.so.0.24.0 | cut -d' ' -f1)"
log "LIB_MD5_BASE=$(md5sum $L/libggml-base.so.0.24.0 | cut -d' ' -f1)"
log "MARK_FPD=$(strings $L/libggml-base.so.0.24.0 | grep -a -c GGML_SCHED_FP_DIFF)"
log "MARK_MKEY=$(strings $L/libggml-base.so.0.24.0 | grep -a -c GGML_META_KEY_DEBUG)"
log "MARK_FAK=$(strings $L/libggml-cuda.so.0.24.0 | grep -a -c GGML_CUDA_FA_KERNEL_DEBUG)"

if busy; then for i in $(seq 1 90); do busy || break; log "WAIT_ARM i=$i"; sleep 20; done; fi
cd /root || exit 1
env GGML_SCHED_FP_DIFF=1 GGML_SCHED_SPLIT_TIMING=1 GGML_META_KEY_DEBUG=1 GGML_CUDA_FA_KERNEL_DEBUG=1 CARDS=0,1,2 SPLIT=tensor L=$L P2P=1 TAG=fpdA PORT=8141 NPRED=256 NODROP=1 bash /root/p60-ab-harness.sh > /tmp/fpdA-driver.log 2>&1 < /dev/null
log "ARM_A_RC=$?"
grep -a -e MEDIAN_TG -e greedy -e STATUS= /tmp/fpdA-driver.log >> $OUT
env GGML_SCHED_FP_DIFF=1 GGML_SCHED_SPLIT_TIMING=1 GGML_SCHED_SPLIT_CACHE=1 CARDS=0,1,2 SPLIT=tensor L=$L P2P=1 TAG=fpdB PORT=8142 NPRED=256 NODROP=1 bash /root/p60-ab-harness.sh > /tmp/fpdB-driver.log 2>&1 < /dev/null
log "ARM_B_RC=$?"
grep -a -e MEDIAN_TG -e greedy -e STATUS= /tmp/fpdB-driver.log >> $OUT

log "===FPD_HEAD_A==="
grep -a FPD /tmp/p60-fpdA-server.log | head -50 >> $OUT
log "===FPD_TAIL_A==="
grep -a FPD /tmp/p60-fpdA-server.log | tail -24 >> $OUT
log "===SCHED_A==="
grep -a SCHED /tmp/p60-fpdA-server.log | tail -4 >> $OUT
log "===SCHED_B==="
grep -a SCHED /tmp/p60-fpdB-server.log | tail -4 >> $OUT
log "===FAK_A==="
grep -a FAK /tmp/p60-fpdA-server.log | head -20 >> $OUT
log "===MKEY_A==="
grep -a MKEY /tmp/p60-fpdA-server.log | head -24 >> $OUT
log "CHAIN_END $(date +%H:%M:%S)"
log "FPD_CHAIN_DONE"
