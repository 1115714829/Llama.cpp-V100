#!/bin/bash
cd /root
SUM=/tmp/graph-split.txt
: > $SUM
DFT=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
run_arm() {
  tag=$1; spec=$2
  echo "=== ARM $tag ===" >> $SUM
  SPEC="$spec" GGML_CUDA_GRAPH_DEBUG=1 CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=$tag NPRED=192 NODROP=1 \
    bash /root/p60-ab-harness.sh > /tmp/$tag-driver.log 2>&1
  grep -a calls= /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a props.changed /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a decode+sync /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a n_ctx=8192 /tmp/p60-$tag-server.log | tail -1 >> $SUM
}
run_arm gnospec "--spec-type none"
run_arm gdraft "--model-draft $DFT --spec-type draft-dflash --spec-draft-n-max 7"
echo GRAPH_SPLIT_DONE >> $SUM
