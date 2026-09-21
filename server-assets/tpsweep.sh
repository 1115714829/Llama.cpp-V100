#!/bin/bash
# Zero-code: re-sweep the tensor-parallel degree at 8K with the CURRENT library (last sweep predates NCCL+P2P).
while ! grep -q NMAX_DONE /tmp/nmax.log 2>/dev/null; do sleep 30; done
echo TPSWEEP_START
cd /root || exit 1
echo '===== TP2 (cards 0,1) ====='
env CARDS=0,1 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=tp2c NPRED=512 bash /root/p60-ab-harness.sh 2>&1 | grep -a -e prompt -e MEDIAN -e greedy -e 'tensor-split'
echo '===== TP3 (cards 0,1,2) ====='
env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=tp3c NPRED=512 bash /root/p60-ab-harness.sh 2>&1 | grep -a -e prompt -e MEDIAN -e greedy -e 'tensor-split'
echo '===== TP4 (cards 0,1,2,3) ====='
env CARDS=0,1,2,3 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=tp4c NPRED=512 bash /root/p60-ab-harness.sh 2>&1 | grep -a -e prompt -e MEDIAN -e greedy -e 'tensor-split'
echo TPSWEEP_DONE