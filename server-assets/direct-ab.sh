#!/bin/bash
SUM=/tmp/direct-ab.txt
: > $SUM
port_busy() { (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null; }
guard() {
  pkill -9 -x llama-server 2>/dev/null
  local n=0
  while [ $n -lt 45 ]; do port_busy $1 || return 0; sleep 2; n=$((n+1)); done
  echo PORT_BUSY=$1 >> $SUM; return 1
}
run_arm() {
  echo ===ARM $1 PORT=$2=== >> $SUM
  guard $2 || return 1
  timeout 2400 env PORT=$2 CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=$1 NPRED=512 GGML_META_HOST_TIMING=1 GGML_CUDA_GRAPH_DEBUG=1 bash /root/p60-ab-harness.sh > /tmp/$1-driver.log 2>&1
  echo ARM_RC=$? >> $SUM
  grep -a -e prompt1 -e MEDIAN_TG -e LAUNCH_FAILED /tmp/$1-driver.log >> $SUM
  grep -a -e calls= /tmp/p60-$1-server.log 2>/dev/null | tail -1 >> $SUM
  grep -a -h META /tmp/p60-$1-server.log 2>/dev/null | tail -1 >> $SUM
  grep -a -h perf: /tmp/p60-$1-server.log 2>/dev/null | tail -2 >> $SUM
}
run_arm dirA 8181
run_arm dirB 8182
echo DIRECT_AB_DONE >> $SUM
