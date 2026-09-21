#!/bin/bash
# Zero-code sweep: speculative draft depth n_max (affects both the verify batch size and the acceptance rate).
cd /root || exit 1
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
for n in 3 5 7; do
  echo "===== ARM n_max=$n ====="
  S="--model-draft $D --spec-type draft-dflash --spec-draft-n-max $n --spec-draft-ngl 999"
  env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=nmax$n NPRED=512 SPEC="$S" bash /root/p60-ab-harness.sh 2>&1 | grep -a -e prompt -e MEDIAN -e greedy
  grep -a 'spec timing' /tmp/p60-nmax$n-server.log 2>/dev/null | tail -1
done
echo NMAX_DONE