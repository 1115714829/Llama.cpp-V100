#!/bin/bash
# Draft block-size scaling: does draft_decode scale with n_max (compute-bound) or not (weight-read-bound)?
cd /root
SUM=/tmp/nmax-arms-summary.txt
: > $SUM
DFT=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
run_arm() {
  nm=$1
  tag=nmax$nm
  echo "=== ARM $tag (spec-draft-n-max=$nm) ===" >> $SUM
  SPEC="--model-draft $DFT --spec-type draft-dflash --spec-draft-n-max $nm" \
  GGML_CUDA_AR_TIMING=1 CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=$tag NPRED=192 NODROP=1 \
    bash /root/p60-ab-harness.sh > /tmp/$tag-driver.log 2>&1
  grep -a tg= /tmp/$tag-driver.log >> $SUM
  grep -a MEDIAN_TG /tmp/$tag-driver.log >> $SUM
  grep -a spec.timing /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a ar_us_avg /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a decode+sync /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a BLOCK /tmp/p60-$tag-server.log | tail -1 >> $SUM
  echo "--- done $tag" >> $SUM
}
run_arm 1
run_arm 7
run_arm 15
echo NMAX_ARMS_DONE >> $SUM
