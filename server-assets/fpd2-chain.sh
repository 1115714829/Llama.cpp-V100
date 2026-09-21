#!/bin/bash
# Chain 2: full attribution with existing probes + MTP vs DFlash2. No build. ASCII only.
set -u
L=/root/libdir-instr
OUT=/tmp/fpd2-chain.log
: > $OUT
log() { echo "$*" >> $OUT; }
log "CHAIN2_START $(date +%H:%M:%S)"

busy() {
  pgrep -f 'llama-benc[h] -m' > /dev/null && return 0
  pgrep -f 'llama-serve[r] --model' > /dev/null && return 0
  pgrep -f 'cmake --buil[d]' > /dev/null && return 0
  pgrep -f 'nvcc' > /dev/null && return 0
  return 1
}

for i in $(seq 1 240); do
  if grep -q -a FPD_CHAIN_DONE /tmp/fpd-chain.log 2>/dev/null; then break; fi
  log "WAIT_CHAIN1 i=$i $(date +%H:%M:%S)"
  sleep 30
done
log "CHAIN1_SEEN=$(grep -a -c FPD_CHAIN_DONE /tmp/fpd-chain.log 2>/dev/null)"

idle=0
for i in $(seq 1 180); do
  if busy; then idle=0; else idle=$((idle+1)); fi
  log "IDLE_WAIT i=$i idle=$idle $(date +%H:%M:%S)"
  if [ $idle -ge 2 ]; then break; fi
  sleep 20
done
if [ $idle -lt 2 ]; then log "ABORT_MACHINE_BUSY"; log "FPD2_DONE"; exit 1; fi
cd /root || exit 1

env GGML_META_HOST_TIMING=1 GGML_CUDA_GRAPH_DEBUG=1 GGML_CUDA_DIRECT_DEBUG=1 GGML_CUDA_OP_TIMING=1 CARDS=0,1,2 SPLIT=tensor L=$L P2P=1 TAG=attr PORT=8151 NPRED=256 NODROP=1 bash /root/p60-ab-harness.sh > /tmp/attr-driver.log 2>&1 < /dev/null
log "ARM_ATTR_RC=$?"
grep -a -e MEDIAN_TG -e greedy -e STATUS= /tmp/attr-driver.log >> $OUT
log "===META==="
grep -a META /tmp/p60-attr-server.log | tail -2 >> $OUT
log "===GRAPHCOUNTERS==="
grep -a 'GRAPH. calls=' /tmp/p60-attr-server.log | tail -5 >> $OUT
log "===PROPDIFF==="
grep -a 'prop diff' /tmp/p60-attr-server.log | head -8 >> $OUT
log "===DIRECTPROBE==="
grep -a DIRECT_PROBE /tmp/p60-attr-server.log | tail -6 >> $OUT
log "===OPTIMING==="
grep -a 'OP. ' /tmp/p60-attr-server.log | tail -30 >> $OUT

env SPEC="--spec-type draft-mtp --spec-draft-n-max 4" CARDS=0,1,2 SPLIT=tensor L=$L P2P=1 TAG=mtp4 PORT=8152 NPRED=256 NODROP=1 bash /root/p60-ab-harness.sh > /tmp/mtp4-driver.log 2>&1 < /dev/null
log "ARM_MTP4_RC=$?"
grep -a -e MEDIAN_TG -e greedy -e STATUS= -e LAUNCH_FAILED /tmp/mtp4-driver.log >> $OUT
log "===MTP4SPEC==="
grep -a spec= /tmp/mtp4-driver.log | head -2 >> $OUT
if grep -q -a LAUNCH_FAILED /tmp/mtp4-driver.log; then log "===MTP4SERVER==="; tail -20 /tmp/p60-mtp4-server.log >> $OUT; fi

log "===TIMINGS_ATTR==="
python3 /root/timings.py attr >> $OUT 2>&1
log "===TIMINGS_MTP4==="
python3 /root/timings.py mtp4 >> $OUT 2>&1
log "CHAIN2_END $(date +%H:%M:%S)"
log "FPD2_DONE"
