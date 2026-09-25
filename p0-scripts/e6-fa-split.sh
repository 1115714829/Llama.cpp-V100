#!/bin/bash
# e6-fa-split.sh - FA split-KV sweep (GGML_CUDA_FA_SPLIT_FLOOR).
#
# The decode FA runs at ~11% FLOPS utilization: q=8 is too small and the long
# KV is walked serially per block. GGML_CUDA_FA_SPLIT_FLOOR raises the KV split
# so blocks run in parallel. Sweep the floor and keep the best.
set -u
RUNLOG=/root/llm/test/e6-run.log
: > "$RUNLOG"

run() {
  local tag=$1; shift
  local envs="$*"
  pkill -f 'llama-serve[r] --model' 2>/dev/null
  sleep 4
  tmux kill-session -t srv 2>/dev/null
  tmux new-session -d -s srv "TAG=$tag SPEC=1 ROUND=1 UB=2048 $envs bash /root/llm/test/t1c-run.sh"
  local code=000
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8082/health)
    [ "$code" = "200" ] && break
    sleep 5
  done
  echo "HEALTH $tag code=$code $envs" >> "$RUNLOG"
  if [ "$code" != "200" ]; then
    tail -4 /root/llm/test/$tag.log >> "$RUNLOG"
    return 1
  fi
  python3 /root/llm/test/stress_client.py --base-url http://127.0.0.1:8082 \
    --model Qwen3.8-27B-Q8_0-BL2 --prompt-tokens 235930 --prompt-file bl-prompt90.txt \
    --gen 320 --reps 1 --tag $tag --timeout 1500 >> "$RUNLOG" 2>&1
  grep -a "draft acceptance" /root/llm/test/$tag.log | tail -1 >> "$RUNLOG"
  pkill -f 'llama-serve[r] --model' 2>/dev/null
  sleep 4
}

run e6f0 ""
for F in 1 2 4 8 16; do
  run "e6f$F" "GGML_CUDA_FA_SPLIT_FLOOR=$F"
done
echo "E6_DONE rc=0" >> "$RUNLOG"
