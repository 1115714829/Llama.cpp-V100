#!/bin/bash
while ! grep -q P0_ALL_DONE /tmp/p0.log; do sleep 20; done
sleep 5
bash /root/p0b-kvtype-128k.sh
bash /root/build-instr.sh
echo CHAIN2_DONE