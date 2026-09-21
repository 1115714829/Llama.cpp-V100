#!/bin/bash
for t in $(cat /root/m5-cand/listprobe.txt); do
  bash /root/m5-probe.sh $t "-Xptxas -v"
done
echo M5_PROBEALL_DONE
