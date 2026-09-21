#!/bin/bash
for i in $(seq 1 40); do
  if grep -q A4_DONE /tmp/a4.log 2>/dev/null; then break; fi
  sleep 10
done
date +%H:%M:%S
grep -a -e 'floor=' -e tg64 /tmp/a4.log