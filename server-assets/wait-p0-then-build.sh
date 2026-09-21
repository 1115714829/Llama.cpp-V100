#!/bin/bash
# wait for the P0 experiment to finish (no NFS contention), then build the instrumented tree
while ! grep -q P0_ALL_DONE /tmp/p0.log; do sleep 20; done
sleep 5
bash /root/build-instr.sh
echo CHAIN_DONE