#!/bin/bash
for i in $(seq 1 45); do
  if grep -q D256_DONE /tmp/d256.log 2>/dev/null; then break; fi
  sleep 10
done
date +%H:%M:%S
tail -4 /tmp/d256.log