#!/bin/bash
# 判定主机与 GPU 是否重叠：加 / 不加 LLAMA_ROUND_TIMING_SYNC（后者强制每步设备同步 => 串行化）
SUM=/tmp/sync-ab.txt
: > $SUM
port_busy() { (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null; }
guard() {
  pkill -9 -x llama-server 2>/dev/null
  local n=0
  while [ $n -lt 45 ]; do port_busy $1 || return 0; sleep 2; n=$((n+1)); done
  echo PORT_BUSY=$1 >> $SUM; return 1
}
run_arm() {
  echo ===ARM $1 EXTRASYNC=$2=== >> $SUM
  guard $3 || return 1
  timeout 2400 env PORT=$3 CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=$1 NPRED=512 $2 bash /root/p60-ab-harness.sh > /tmp/$1-driver.log 2>&1
  echo ARM_RC=$? >> $SUM
  grep -a -e prompt1 -e MEDIAN_TG -e LAUNCH_FAILED -e greedy /tmp/$1-driver.log >> $SUM
  grep -a n_ctx=8192 /tmp/$1-driver.log | tail -1 >> $SUM
  grep -a -e round.timing -e decode.sync /tmp/p60-$1-server.log 2>/dev/null | tail -4 >> $SUM
}
run_arm sy0 '' 8221
run_arm sy1 'LLAMA_ROUND_TIMING_SYNC=1' 8222
run_arm sy1b 'LLAMA_ROUND_TIMING_SYNC=1' 8223
run_arm sy0b '' 8224
echo SYNC_AB_DONE >> $SUM
