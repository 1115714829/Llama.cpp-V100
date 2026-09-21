#!/bin/bash
for i in $(seq 1 28); do
  if grep -q 'ARM 2' /tmp/ar-ab.log 2>/dev/null; then break; fi
  sleep 10
done
date +%H:%M:%S
cat /tmp/ar-ab.log
echo ===AROFF-AR===; grep -a -e ar_us_avg -e 'RT. perf' /tmp/p60-aroff-server.log 2>/dev/null | tail -2