#!/bin/bash
for i in $(seq 1 24); do
  if grep -q MEDIAN_TG /tmp/lc-spec-lc64-tp4-q8.txt 2>/dev/null; then break; fi
  sleep 20
done
grep -a -e prompt -e MEDIAN -e timings -e perf -e AR. /tmp/lc-spec-lc64-tp4-q8.txt 2>/dev/null | head -12