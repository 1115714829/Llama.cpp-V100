#!/bin/bash
# Sequential NCCL tuning arms for the AR measurement (main line owns the box).
cd /root
SUM=/tmp/nccl-arms-summary.txt
: > $SUM
run_arm() {
  tag=$1; shift
  echo "=== ARM $tag : $* ===" >> $SUM
  date >> $SUM
  env "$@" GGML_CUDA_AR_TIMING=1 CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=$tag NPRED=96 NODROP=1 \
    bash /root/p60-ab-harness.sh > /tmp/$tag-driver.log 2>&1
  grep -a MEDIAN_TG /tmp/$tag-driver.log >> $SUM
  grep -a prompt1 /tmp/$tag-driver.log >> $SUM
  grep -a spec.timing /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a ar_us_avg /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a calls= /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a decode+sync /tmp/p60-$tag-server.log | tail -1 >> $SUM
  echo "--- done $tag" >> $SUM
}
run_arm arctl
run_arm arll128 NCCL_PROTO=LL128
run_arm ar1ch NCCL_MAX_NCHANNELS=1
run_arm artree NCCL_ALGO=Tree
echo ALL_ARMS_DONE >> $SUM
