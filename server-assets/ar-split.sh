#!/bin/bash
# Split target vs draft allreduce: no-draft (M=1) vs draft active (M=2 block).
cd /root
SUM=/tmp/ar-split.txt
: > $SUM
DFT=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
run_arm() {
  tag=$1; spec=$2
  echo "=== ARM $tag spec=$spec ===" >> $SUM
  SPEC="$spec" GGML_CUDA_AR_TIMING=1 CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=$tag NPRED=384 NODROP=1 \
    bash /root/p60-ab-harness.sh > /tmp/$tag-driver.log 2>&1
  grep -a tg= /tmp/$tag-driver.log >> $SUM
  grep -a calls= /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a ar_us_avg /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a decode+sync /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a n_ctx=8192 /tmp/p60-$tag-server.log | tail -1 >> $SUM
}
run_arm nospec "--spec-type none"
run_arm nmax1 "--model-draft $DFT --spec-type draft-dflash --spec-draft-n-max 1"
echo AR_SPLIT_DONE >> $SUM
