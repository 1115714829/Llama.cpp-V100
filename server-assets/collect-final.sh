#!/bin/bash
# collect-final.sh -- wait for every queued job, then print ONLY the evidence the Goal asks for.
set -u
for i in $(seq 1 260); do
  grep -q FINISH_PPL_DONE /tmp/finish-ppl.log 2>/dev/null && { echo "ppl done"; break; }
  sleep 15
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "=== ITEM 1: /root/goal-baseline.txt ==="
cat /root/goal-baseline.txt

echo "=== ITEM 2/3: final arms (same script as the baseline run) ==="
for t in final-bf-tp3 final-nccl-tp3; do
  echo "## $t"
  grep -aE '^TAG=|^config:|^libs:|^prompt[123]: |^MEDIAN_TG=|^greedy:|perf:|spec timing|target decode\+sync|allreduce init' /tmp/p60-$t.log
done

echo "=== ITEM 4: perplexity (AL + ppl fallback, numerics changed with NCCL) ==="
grep -aE 'corpus:|PPL ARM|Final estimate|exit=|libggml-cuda|NCCL|AllReduce' /tmp/finish-ppl.log

echo "=== round-2 knobs ==="
grep -aE '^##### KNOB ARM|MEDIAN_TG=' /tmp/nccl-knobs2.log

echo "=== round-1 knobs (env sweep) ==="
grep -aE '^##### ENV ARM|MEDIAN_TG=' /tmp/nccl-env-sweep.log
echo COLLECT_DONE
