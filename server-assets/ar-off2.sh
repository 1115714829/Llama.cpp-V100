#!/bin/bash
# After the device-AR arm, re-measure the baseline with the SAME library (env unset) for a clean A/B.
while ! grep -q ARON_DONE /tmp/aron.log 2>/dev/null; do sleep 20; done
cd /root
env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=aroff2 NPRED=512 bash /root/p60-ab-harness.sh 2>&1 | tail -22
grep -a ar_us_avg /tmp/p60-aroff2-server.log 2>/dev/null | tail -1
grep -a 'RT. perf' /tmp/p60-aroff2-server.log 2>/dev/null | grep -a -e Qwen3.8-27B | tail -1
echo AROFF2_DONE