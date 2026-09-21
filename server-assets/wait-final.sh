#!/bin/bash
for i in $(seq 1 45); do
  if grep -q FINAL_DONE /tmp/final.log 2>/dev/null; then break; fi
  sleep 10
done
date +%H:%M:%S
cat /tmp/final.log