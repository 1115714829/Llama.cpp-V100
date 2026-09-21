#!/bin/bash
# Round 97 rerun: official protocol NPRED=512. Port-guarded (a stale llama-server held 8120 and voided all 8 arms).
SUM=/tmp/round97.txt
: > $SUM
port_busy() { (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null; }
guard() {
  pkill -9 -x llama-server 2>/dev/null
  local n=0
  while [ $n -lt 45 ]; do
    port_busy $1 || return 0
    sleep 2; n=$((n+1))
  done
  echo PORT_STILL_BUSY=$1 >> $SUM
  return 1
}
run_arm() {
  local tag=$1 cards=$2 devd=$3 port=$4
  echo ===ARM $tag CARDS=$cards DEVD=$devd PORT=$port=== >> $SUM
  guard $port || return 1
  timeout 1800 env PORT=$port CARDS=$cards SPLIT=tensor L=/root/libdir-instr P2P=1 DEVD=$devd TAG=$tag NPRED=512 bash /root/p60-ab-harness.sh > /tmp/$tag-driver.log 2>&1
  echo ARM_RC=$? >> $SUM
  grep -a -e prompt1 -e prompt2 -e prompt3 -e MEDIAN_TG -e greedy -e FAILED -e LAUNCH_FAILED /tmp/$tag-driver.log >> $SUM
  grep -a n_ctx=8192 /tmp/$tag-driver.log | tail -1 >> $SUM
  python3 /root/timings.py $tag >> $SUM 2>/dev/null
}
run_arm dvA 0,1,2 CUDA0 8131
run_arm dvB 0,1,2 none 8132
run_arm dvC 0,1,2 CUDA3 8133
run_arm dvD 0,1,2 none 8134
run_arm tp2a 0,1 none 8135
run_arm tp3a 0,1,2 none 8136
run_arm tp2b 0,1 none 8137
run_arm tp3b 0,1,2 none 8138
echo ROUND97_DONE >> $SUM
