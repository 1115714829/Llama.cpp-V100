#!/bin/bash
# 2026-09-21: official protocol (NPRED=512) A/Bs. Arm 1-4: draft placement (DEVD). Arm 5-8: TP2 vs TP3.
SUM=/tmp/round97.txt
: > $SUM
run_arm() {
  echo ===ARM $1 CARDS=$2 DEVD=$3=== >> $SUM
  timeout 1800 env CARDS=$2 SPLIT=tensor L=/root/libdir-instr P2P=1 DEVD=$3 TAG=$1 NPRED=512 bash /root/p60-ab-harness.sh > /tmp/$1-driver.log 2>&1
  echo ARM_RC=$? >> $SUM
  grep -a -e prompt1 -e prompt2 -e prompt3 -e MEDIAN_TG -e greedy -e FAILED /tmp/$1-driver.log >> $SUM
  grep -a n_ctx=8192 /tmp/$1-driver.log | tail -1 >> $SUM
  grep -a draft.acceptance /tmp/p60-$1-server.log 2>/dev/null | tail -1 >> $SUM
}
pkill -9 -f lib.ab.sh
sleep 2
run_arm dvA 0,1,2 CUDA0
run_arm dvB 0,1,2
run_arm dvC 0,1,2 CUDA3
run_arm dvD 0,1,2
run_arm tp2a 0,1
run_arm tp3a 0,1,2
run_arm tp2b 0,1
run_arm tp3b 0,1,2
echo ROUND97_DONE >> $SUM
