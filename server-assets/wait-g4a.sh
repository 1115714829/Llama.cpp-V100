#!/bin/bash
for i in $(seq 1 40); do
  if grep -q 'ARM-B' /tmp/g4-ab.log 2>/dev/null; then break; fi
  sleep 10
done
date +%H:%M:%S
sed -n '/ARM-A/,$p' /tmp/g4-ab.log