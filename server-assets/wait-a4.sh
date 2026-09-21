#!/bin/bash
for i in $(seq 1 50); do
  if grep -q A4_DONE /tmp/a4.log 2>/dev/null; then break; fi
  sleep 10
done
date +%H:%M:%S
cat /tmp/a4.log