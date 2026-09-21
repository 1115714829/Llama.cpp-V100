#!/bin/bash
LOG=/root/fak-build-run.log
for i in $(seq 1 120); do
  if grep -q -e BUILD_RC "$LOG"; then
    echo WAIT_DONE
    exit 0
  fi
  sleep 30
done
echo WAIT_TIMEOUT
exit 1
