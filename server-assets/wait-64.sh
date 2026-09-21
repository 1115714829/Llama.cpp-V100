#!/bin/bash
for i in $(seq 1 60); do
  if grep -q MEDIAN_TG /tmp/lc-spec-lc64-f16.txt 2>/dev/null; then break; fi
  sleep 20
done
echo '=== lc64-q8 ==='
grep -a -e prompt -e MEDIAN -e timings -e 'RT. perf' -e 'AR.' /tmp/lc-spec-lc64-q8.txt 2>/dev/null
echo '=== lc64-f16 ==='
grep -a -e prompt -e MEDIAN -e timings -e 'RT. perf' -e 'AR.' /tmp/lc-spec-lc64-f16.txt 2>/dev/null
echo '=== chain ==='
tail -3 /tmp/lc-chain.log 2>/dev/null