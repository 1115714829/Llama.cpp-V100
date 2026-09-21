#!/bin/bash
# wait for P0 to finish, then build the instrumented tree (P0b 128K A/B moved to after the build)
while ! grep -q P0_ALL_DONE /tmp/p0.log; do sleep 20; done
sleep 5
bash /root/build-instr.sh
echo CHAIN3_DONE