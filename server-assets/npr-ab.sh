#!/bin/bash
SUM=/tmp/npr-ab.txt
: > $SUM
n=0
while [ $n -lt 120 ]; do
  if grep -q LIB_AB_DONE /tmp/lib-ab.txt 2>/dev/null; then break; fi
  sleep 20; n=$((n+1))
done
run_arm() {
  echo ===ARM $1 NPRED=$2=== >> $SUM
  timeout 1800 env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=$1 NPRED=$2 bash /root/p60-ab-harness.sh > /tmp/$1-driver.log 2>&1
  echo ARM_RC=$? >> $SUM
  grep -a -e prompt1 -e prompt2 -e prompt3 -e MEDIAN_TG -e greedy -e FAILED /tmp/$1-driver.log >> $SUM
  grep -a n_ctx=8192 /tmp/$1-driver.log | tail -1 >> $SUM
  grep -a 'spec timing' /tmp/p60-$1-server.log 2>/dev/null | tail -1 >> $SUM
}
run_arm na1 512
run_arm nb1 192
run_arm nb2 192
run_arm na2 512
echo NPR_AB_DONE >> $SUM
