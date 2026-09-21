#!/bin/bash
SUM=/tmp/tp-ab.txt
: > $SUM
rm -f /tmp/p60-*-server.log
run_arm() {
  echo ===ARM $1 CARDS=$2=== >> $SUM
  env CARDS=$2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=$1 NPRED=192 bash /root/p60-ab-harness.sh > /tmp/$1-driver.log 2>&1
  grep -a -e prompt1 -e prompt2 -e prompt3 -e MEDIAN_TG -e greedy -e FAILED -e LAUNCH_FAILED /tmp/$1-driver.log >> $SUM
  grep -a n_ctx=8192 /tmp/$1-driver.log | tail -1 >> $SUM
}
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> $SUM
run_arm tp2a 0,1
run_arm tp3a 0,1,2
run_arm tp3b 0,1,2
run_arm tp2b 0,1
echo TP_AB_DONE >> $SUM
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> $SUM
