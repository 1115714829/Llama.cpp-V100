#!/bin/bash
# Chain 4: TP sweep with the current NCCL+P2P library, authoritative NPRED=512, identical probe set. ASCII only.
set -u
L=/root/libdir-instr
OUT=/tmp/fpd4-chain.log
: > $OUT
log() { echo "$*" >> $OUT; }
log "CHAIN4_START $(date +%H:%M:%S)"

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
  sleep 20
done
busy && { log "ABORT_MACHINE_BUSY"; log "FPD4_DONE"; exit 1; }
log "MACHINE_IDLE $(date +%H:%M:%S)"
log "LIB_MD5_BASE=$(md5sum $L/libggml-base.so.0.24.0 | cut -d' ' -f1)"
log "LIB_MD5_CUDA=$(md5sum $L/libggml-cuda.so.0.24.0 | cut -d' ' -f1)"

cd /root || exit 1
P="GGML_META_HOST_TIMING=1 GGML_CUDA_AR_TIMING=1"
i=0
for c in 0,1,2 0,1,2,3 0,1,2,3,4,5; do
  i=$((i+1))
  T=tp$i
  log "===ARM $T CARDS=$c $(date +%H:%M:%S)==="
  env $P CARDS=$c SPLIT=tensor L=$L P2P=1 TAG=$T PORT=817$i NPRED=512 NODROP=1 bash /root/p60-ab-harness.sh > /tmp/$T-driver.log 2>&1 < /dev/null
  log "ARM_${T}_RC=$?"
  grep -a -e MEDIAN_TG -e greedy -e STATUS= -e LAUNCH_FAILED /tmp/$T-driver.log >> $OUT
  log "META_$T=$(grep -a META /tmp/p60-$T-server.log | tail -1)"
  log "AR_$T=$(grep -a ar_us_avg /tmp/p60-$T-server.log | tail -1)"
  python3 /root/timings.py $T >> $OUT 2>&1
done
log "CHAIN4_END $(date +%H:%M:%S)"
log "FPD4_DONE"
