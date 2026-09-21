#!/bin/bash
# A/B: pre-change FA table (libdir-prefa) vs M5 Q_in_reg=false (libdir-instr)
cd /root
SUM=/tmp/fa-ab-summary.txt
: > $SUM
LB=/root/libdir-instr/llama-bench
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf

run_bench() {
  tag=$1; lib=$2
  echo "=== BENCH $tag lib=$lib ===" >> $SUM
  LD_LIBRARY_PATH=$lib CUDA_VISIBLE_DEVICES=0,1,2 $LB -m $M -ngl 999 -sm tensor -ts 1/1/1 -p 8192,32768 -n 0 -r 2 -ctk f16 -ctv f16 -fa 1 -ub 2048 -o md > /tmp/fa-$tag.log 2>&1
  grep -a pp /tmp/fa-$tag.log >> $SUM
}
run_bench pre /root/libdir-prefa
run_bench post /root/libdir-instr

run_arm() {
  tag=$1; lib=$2
  echo "=== YARD $tag lib=$lib ===" >> $SUM
  env CARDS=0,1,2 SPLIT=tensor L=$lib P2P=1 TAG=$tag NPRED=384 NODROP=1 \
    bash /root/p60-ab-harness.sh > /tmp/$tag-driver.log 2>&1
  grep -a tg= /tmp/$tag-driver.log >> $SUM
  grep -a MEDIAN_TG /tmp/$tag-driver.log >> $SUM
  grep -a greedy /tmp/$tag-driver.log >> $SUM
  grep -a spec.timing /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a decode+sync /tmp/p60-$tag-server.log | tail -1 >> $SUM
}
run_arm ypre /root/libdir-prefa
run_arm ypost /root/libdir-instr
echo FA_AB_DONE >> $SUM
