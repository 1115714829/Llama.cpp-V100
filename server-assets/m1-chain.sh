#!/bin/bash
# Chain 5 = M1: differential arm (no-spec vs DFlash2) to count the draft's dispatches per round. ASCII only.
set -u
L=/root/libdir-instr
OUT=/tmp/m1-chain.log
: > $OUT
log() { echo "$*" >> $OUT; }
log "M1_START $(date +%H:%M:%S)"
busy() {
  pgrep -f 'llama-benc[h] -m' > /dev/null && return 0
  pgrep -f 'llama-serve[r] --model' > /dev/null && return 0
  pgrep -f 'cmake --buil[d]' > /dev/null && return 0
  return 1
}
for i in $(seq 1 180); do busy || break; log "WAIT i=$i"; sleep 20; done
busy && { log "ABORT_BUSY"; log "M1_DONE"; exit 1; }
log "IDLE $(date +%H:%M:%S)"
log "LIB_BASE=$(md5sum $L/libggml-base.so.0.24.0 | cut -d' ' -f1) LIB_CUDA=$(md5sum $L/libggml-cuda.so.0.24.0 | cut -d' ' -f1)"
cd /root || exit 1
P="GGML_META_HOST_TIMING=1 GGML_CUDA_GRAPH_DEBUG=1"
i=0
for s in none dflash; do
  i=$((i+1))
  if [ $s = none ]; then S="--spec-type none"; else S="--spec-type draft-dflash --spec-draft-n-max 7"; fi
  T=m1$s
  log "===ARM $T spec=$S $(date +%H:%M:%S)==="
  env $P SPEC="$S" CARDS=0,1,2 SPLIT=tensor L=$L P2P=1 TAG=$T PORT=818$i NPRED=512 NODROP=1 bash /root/p60-ab-harness.sh > /tmp/$T-driver.log 2>&1 < /dev/null
  log "RC_$T=$?"
  grep -a -e MEDIAN_TG -e greedy -e STATUS= -e LAUNCH_FAILED /tmp/$T-driver.log >> $OUT
  log "META_$T=$(grep -a META /tmp/p60-$T-server.log | tail -1)"
  log "GRAPH_$T=$(grep -a 'GRAPH. calls=' /tmp/p60-$T-server.log | tail -1)"
  log "RT_$T=$(grep -a 'RT. perf' /tmp/p60-$T-server.log | tail -1)"
  python3 /root/timings.py $T >> $OUT 2>&1
done
log "M1_END $(date +%H:%M:%S)"
log "M1_DONE"
