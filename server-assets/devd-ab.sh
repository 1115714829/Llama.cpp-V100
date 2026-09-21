#!/bin/bash
SUM=/tmp/devd-ab.txt
: > $SUM
run_arm() {
  echo ===ARM $1 CARDS=$2 DEVD=$3=== >> $SUM
  timeout 1500 env CARDS=$2 SPLIT=tensor L=/root/libdir-instr P2P=1 DEVD=$3 TAG=$1 NPRED=192 bash /root/p60-ab-harness.sh > /tmp/$1-driver.log 2>&1
  echo ARM_RC=$? >> $SUM
  grep -a -e prompt1 -e prompt2 -e prompt3 -e MEDIAN_TG -e greedy -e FAILED -e LAUNCH_FAILED -e abort -e Abort -e GGML_ASSERT /tmp/$1-driver.log >> $SUM
  grep -a n_ctx=8192 /tmp/$1-driver.log | tail -1 >> $SUM
  grep -a -h -e Meta -e meta_device -e crash /tmp/p60-$1-server.log 2>/dev/null | head -3 >> $SUM
}
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> $SUM
run_arm dv0 0,1,2 CUDA0
run_arm dvb 0,1,2
run_arm dv3 0,1,2 CUDA3
run_arm dvc 0,1,2
echo DEVD_AB_DONE >> $SUM
