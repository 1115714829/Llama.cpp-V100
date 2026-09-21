#!/bin/bash
# ppl-after-final.sh -- run the perplexity comparison right after the formal runs (skip knob round 2).
set -u
for i in $(seq 1 120); do
  grep -q FINAL_NCCL_DONE /tmp/final-nccl.log 2>/dev/null && { echo "formal run done"; break; }
  sleep 15
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
bash /root/ppl-ab.sh
echo FINISH_PPL_DONE
