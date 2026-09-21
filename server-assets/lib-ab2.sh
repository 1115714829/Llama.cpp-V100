#!/bin/bash
SUM=/tmp/lib-ab.txt
: > $SUM
run_arm() {
  echo ===ARM $1 L=$2=== >> $SUM
  timeout 1500 env CARDS=0,1,2 SPLIT=tensor L=$2 P2P=1 TAG=$1 NPRED=192 bash /root/p60-ab-harness.sh > /tmp/$1-driver.log 2>&1
  echo ARM_RC=$? >> $SUM
  grep -a -e prompt1 -e prompt2 -e prompt3 -e MEDIAN_TG -e greedy -e FAILED /tmp/$1-driver.log >> $SUM
  grep -a n_ctx=8192 /tmp/$1-driver.log | tail -1 >> $SUM
}
run_arm la1 /root/libdir-nccl
run_arm lb1 /root/libdir-instr
run_arm lb2 /root/libdir-instr
run_arm la2 /root/libdir-nccl
echo LIB_AB_DONE >> $SUM
