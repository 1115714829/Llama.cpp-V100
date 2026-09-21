#!/bin/bash
for i in $(seq 1 30); do
  if grep -q 'device-side push AllReduce enabled' /tmp/p60-aron-server.log 2>/dev/null; then break; fi
  sleep 10
done
echo -n 'mode line: '; grep -a -c 'device-side push AllReduce enabled' /tmp/p60-aron-server.log 2>/dev/null
for i in $(seq 1 24); do
  if grep -q 'prompt1:' /tmp/p60-aron.log 2>/dev/null; then break; fi
  sleep 10
done
date +%H:%M:%S
tail -12 /tmp/aron.log
grep -a ar_us_avg /tmp/p60-aron-server.log 2>/dev/null | tail -1