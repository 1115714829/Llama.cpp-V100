#!/bin/bash
# Round 97c: TP2 vs TP3 at the OFFICIAL protocol (NPRED=512). DEVD arms dropped (see docs: Meta()/output.weight block).
SUM=/tmp/round97c.txt
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
  local tag=$1 cards=$2 port=$3
  echo ===ARM $tag CARDS=$cards PORT=$port=== >> $SUM
  guard $port || return 1
  timeout 2400 env PORT=$port CARDS=$cards SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=$tag NPRED=512 bash /root/p60-ab-harness.sh > /tmp/$tag-driver.log 2>&1
  echo ARM_RC=$? >> $SUM
  grep -a -e prompt1 -e prompt2 -e prompt3 -e MEDIAN_TG -e greedy -e LAUNCH_FAILED /tmp/$tag-driver.log >> $SUM
  grep -a n_ctx=8192 /tmp/$tag-driver.log | tail -1 >> $SUM
  python3 /root/timings.py $tag >> $SUM 2>/dev/null
}
run_arm tp3a 0,1,2 8141
run_arm tp2a 0,1 8142
run_arm tp2b 0,1 8143
run_arm tp3b 0,1,2 8144
echo ROUND97C_DONE >> $SUM
