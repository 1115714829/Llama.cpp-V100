#!/bin/bash
# Chain 6 = M1 (corrected): DFlash2 n=7 vs n=3, FULL spec (model-draft included). Decides B1 vs B2.
set -u
L=/root/libdir-instr
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
OUT=/tmp/m1b-chain.log
: > $OUT
log() { echo "$*" >> $OUT; }
log "M1B_START $(date +%H:%M:%S)"
busy() {
  pgrep -f 'llama-benc[h] -m' > /dev/null && return 0
  pgrep -f 'llama-serve[r] --model' > /dev/null && return 0
  pgrep -f 'cmake --buil[d]' > /dev/null && return 0
  return 1
}
for i in $(seq 1 180); do busy || break; log "WAIT i=$i"; sleep 20; done
busy && { log "ABORT_BUSY"; log "M1B_DONE"; exit 1; }
log "IDLE $(date +%H:%M:%S)"
log "LIB_BASE=$(md5sum $L/libggml-base.so.0.24.0 | cut -d' ' -f1) LIB_CUDA=$(md5sum $L/libggml-cuda.so.0.24.0 | cut -d' ' -f1)"
log "DRAFT_MD5=$(md5sum $D | cut -d' ' -f1) DRAFT_SIZE=$(stat -c %s $D)"
cd /root || exit 1
P="GGML_META_HOST_TIMING=1 GGML_CUDA_GRAPH_DEBUG=1 LLAMA_SPEC_TIMING=1"
i=0
for n in 7 3; do
  i=$((i+1))
  S="--model-draft $D --spec-type draft-dflash --spec-draft-n-max $n"
  T=m1n$n
  log "===ARM $T n_max=$n $(date +%H:%M:%S)==="
  env $P SPEC="$S" CARDS=0,1,2 SPLIT=tensor L=$L P2P=1 TAG=$T PORT=819$i NPRED=512 NODROP=1 bash /root/p60-ab-harness.sh > /tmp/$T-driver.log 2>&1 < /dev/null
  log "RC_$T=$?"
  grep -a -e MEDIAN_TG -e greedy -e STATUS= -e LAUNCH_FAILED /tmp/$T-driver.log >> $OUT
  log "META_$T=$(grep -a META /tmp/p60-$T-server.log | tail -1)"
  log "GRAPH_$T=$(grep -a calls= /tmp/p60-$T-server.log | grep -a -e 'calls=' | tail -1)"
  log "SPECT_$T=$(grep -a -e 'spec timing' /tmp/p60-$T-server.log | tail -2)"
  python3 /root/timings.py $T >> $OUT 2>&1
done
log "M1B_END $(date +%H:%M:%S)"
log "M1B_DONE"
