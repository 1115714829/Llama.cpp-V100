#!/bin/bash
for i in $(seq 1 44); do
  if grep -q MEDIAN_TG /tmp/lc-spec-lc32-f16.txt 2>/dev/null; then break; fi
  sleep 20
done
echo '=== lc32-f16 ==='
cat /tmp/lc-spec-lc32-f16.txt 2>/dev/null
echo '=== par tail ==='
tail -4 /tmp/lc-par.log