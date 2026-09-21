#!/bin/bash
# A/B: NCCL allreduce (off) vs device-side push allreduce (on). Official protocol, drop_caches.
echo AR_AB_START
cd /root
echo '=========== ARM 1: baseline (NCCL) ==========='
env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=aroff NPRED=512 bash /root/p60-ab-harness.sh 2>&1 | tail -25
echo '=========== ARM 2: device-side AR ==========='
env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=aron ARENV=DEVICE=1 NPRED=512 bash /root/p60-ab-harness.sh 2>&1 | tail -25
echo '--- arm2 allreduce + round timing ---'
grep -a -e 'device-side push AllReduce enabled' -e 'ar_us_avg' -e 'RT. perf' /tmp/p60-aron.log 2>/dev/null | tail -4
echo AR_AB_DONE