#!/bin/bash
# Final consistency check: reproduce the official baseline with the current library (canonical + default-off FA probe).
cd /root
env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=final NPRED=512 bash /root/p60-ab-harness.sh 2>&1 | grep -a -e prompt -e MEDIAN -e greedy -e libs -e 'libggml-cuda'
echo FINAL_DONE