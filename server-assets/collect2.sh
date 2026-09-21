#!/bin/bash
# collect2.sh -- wait for the whole queue, then print ONLY compact evidence lines.
set -u
for i in $(seq 1 200); do
  grep -q FINISH_PPL_DONE /tmp/finish-ppl.log 2>/dev/null && { echo "queue done"; break; }
  sleep 15
done
echo "=== ITEM1 baseline file ==="
grep -aE 'MEDIAN|TAG|config|tok/s' /root/goal-baseline.txt | head -12
echo "=== ITEM2/3 final arms: medians + timing ==="
for t in final-bf-tp3 final-nccl-tp3; do
  echo "## $t"
  grep -aE '^config:|^MEDIAN_TG=|^prompt[123]:|^greedy:|perf:|spec timing|target decode\+sync' /tmp/p60-$t.log 2>/dev/null
done
echo "=== ITEM4 ppl ==="
grep -aE 'corpus:|PPL ARM|estimate|exit=|libggml-cuda' /tmp/finish-ppl.log 2>/dev/null
echo COLLECT2_DONE
