#!/bin/bash
# finish-ppl.sh -- after the knob sweep, run the perplexity comparison (Goal item 4 fallback).
set -u
for i in $(seq 1 200); do
  grep -q NCCL_KNOBS2_DONE /tmp/nccl-knobs2.log 2>/dev/null && { echo "knob sweep done"; break; }
  sleep 15
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
bash /root/ppl-ab.sh
echo FINISH_PPL_DONE
