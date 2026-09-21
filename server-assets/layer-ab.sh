#!/bin/bash
# 检查 split 模式：layer vs tensor（同库同参数，唯一变量是 split）
SUM=/tmp/layer-ab.txt
: > $SUM
port_busy() { (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null; }
guard() {
  pkill -9 -x llama-server 2>/dev/null
  local n=0
  while [ $n -lt 45 ]; do port_busy $1 || return 0; sleep 2; n=$((n+1)); done
  echo PORT_BUSY=$1 >> $SUM; return 1
}
run_arm() {
  echo ===ARM $1 SPLIT=$2 CARDS=$3=== >> $SUM
  guard $4 || return 1
  timeout 2400 env PORT=$4 CARDS=$3 SPLIT=$2 L=/root/libdir-instr P2P=1 TAG=$1 NPRED=512 bash /root/p60-ab-harness.sh > /tmp/$1-driver.log 2>&1
  echo ARM_RC=$? >> $SUM
  grep -a -e prompt1 -e prompt2 -e prompt3 -e MEDIAN_TG -e greedy -e LAUNCH_FAILED /tmp/$1-driver.log >> $SUM
  grep -a n_ctx=8192 /tmp/$1-driver.log | tail -1 >> $SUM
  python3 /root/timings.py $1 >> $SUM 2>/dev/null
}
run_arm lyA layer 0,1,2 8201
run_arm tsA tensor 0,1,2 8202
run_arm tsB tensor 0,1,2 8203
run_arm lyB layer 0,1,2 8204
echo LAYER_AB_DONE >> $SUM
