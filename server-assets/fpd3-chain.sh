#!/bin/bash
# Chain 3: build N6a (GGML_META_REBUILD_CACHE) and A/B it, plus a clean MTP control. ASCII only.
set -u
TREE=/root/llm/test/v100-opt/llama.cpp
L=/root/libdir-instr
OUT=/tmp/fpd3-chain.log
: > $OUT
log() { echo "$*" >> $OUT; }
log "CHAIN3_START $(date +%H:%M:%S)"

busy() {
  pgrep -f 'llama-benc[h] -m' > /dev/null && return 0
  pgrep -f 'llama-serve[r] --model' > /dev/null && return 0
  pgrep -f 'cmake --buil[d]' > /dev/null && return 0
  pgrep -f 'nvcc' > /dev/null && return 0
  return 1
}

for i in $(seq 1 240); do
  busy || break
  log "WAIT_IDLE i=$i $(date +%H:%M:%S)"
  sleep 30
done
busy && { log "ABORT_MACHINE_BUSY"; log "FPD3_DONE"; exit 1; }
log "MACHINE_IDLE $(date +%H:%M:%S)"

touch /tmp/LLAMA_BUILD_LOCK
SRC=$TREE/ggml/src/ggml-backend-meta.cpp
M1=$(md5sum /root/fpd/ggml-backend-meta.cpp | cut -d' ' -f1)
M2=$(md5sum $SRC | cut -d' ' -f1)
log "META_MD5_BEFORE=$M2 UPLOAD=$M1"
if [ "$M1" != "$M2" ]; then cp /root/fpd/ggml-backend-meta.cpp $SRC; log "META_SWAPPED"; fi
log "META_MD5_AFTER=$(md5sum $SRC | cut -d' ' -f1)"

cd $TREE || { rm -f /tmp/LLAMA_BUILD_LOCK; log "ABORT_NO_TREE"; log "FPD3_DONE"; exit 1; }
cmake --build build-instr --config Release -j128 > /tmp/fpd3-build.log 2>&1
RC=$?
log "BUILD_RC=$RC ERRORS=$(grep -c -e error /tmp/fpd3-build.log) TARGETS=$(grep -c -e 'Built target' /tmp/fpd3-build.log)"
log "BUILD_TAIL=$(tail -1 /tmp/fpd3-build.log)"
if [ $RC -ne 0 ]; then grep -a -e error /tmp/fpd3-build.log | head -20 >> $OUT; rm -f /tmp/LLAMA_BUILD_LOCK; log "FPD3_DONE"; exit 1; fi
cp -a build-instr/bin/. $L/
rm -f /tmp/LLAMA_BUILD_LOCK
log "LIB_MD5_BASE=$(md5sum $L/libggml-base.so.0.24.0 | cut -d' ' -f1)"
log "LIB_MD5_CUDA=$(md5sum $L/libggml-cuda.so.0.24.0 | cut -d' ' -f1)"
log "MARK_RCACHE=$(strings $L/libggml-base.so.0.24.0 | grep -a -c GGML_META_REBUILD_CACHE)"

cd /root || exit 1
env CARDS=0,1,2 SPLIT=tensor L=$L P2P=1 TAG=n6c0 PORT=8161 NPRED=256 NODROP=1 bash /root/p60-ab-harness.sh > /tmp/n6c0-driver.log 2>&1 < /dev/null
log "ARM_C0_RC=$?"
grep -a -e MEDIAN_TG -e greedy -e STATUS= /tmp/n6c0-driver.log >> $OUT

env GGML_META_REBUILD_CACHE=1 GGML_META_KEY_DEBUG=1 GGML_CUDA_GRAPH_DEBUG=1 GGML_META_HOST_TIMING=1 CARDS=0,1,2 SPLIT=tensor L=$L P2P=1 TAG=n6c1 PORT=8162 NPRED=256 NODROP=1 bash /root/p60-ab-harness.sh > /tmp/n6c1-driver.log 2>&1 < /dev/null
log "ARM_C1_RC=$?"
grep -a -e MEDIAN_TG -e greedy -e STATUS= /tmp/n6c1-driver.log >> $OUT
log "===C1_META==="
grep -a META /tmp/p60-n6c1-server.log | tail -2 >> $OUT
log "===C1_GRAPH==="
grep -a 'GRAPH. calls=' /tmp/p60-n6c1-server.log | tail -3 >> $OUT
log "===C1_MKEY==="
grep -a MKEY /tmp/p60-n6c1-server.log | grep -a 'i=0' | head -20 >> $OUT

env SPEC="--spec-type draft-mtp --spec-draft-n-max 4" CARDS=0,1,2 SPLIT=tensor L=$L P2P=1 TAG=n6mtp PORT=8163 NPRED=256 NODROP=1 bash /root/p60-ab-harness.sh > /tmp/n6mtp-driver.log 2>&1 < /dev/null
log "ARM_MTP_RC=$?"
grep -a -e MEDIAN_TG -e greedy -e STATUS= /tmp/n6mtp-driver.log >> $OUT

log "===TIMINGS_C0==="
python3 /root/timings.py n6c0 >> $OUT 2>&1
log "===TIMINGS_C1==="
python3 /root/timings.py n6c1 >> $OUT 2>&1
log "===TIMINGS_MTP==="
python3 /root/timings.py n6mtp >> $OUT 2>&1
log "CHAIN3_END $(date +%H:%M:%S)"
log "FPD3_DONE"
