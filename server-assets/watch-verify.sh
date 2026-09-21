#!/bin/bash
while ! grep -q BUILD_ALL_DONE /tmp/build-instr.log; do sleep 30; done
sleep 5
bash /root/b-verify.sh
echo WATCHER_DONE