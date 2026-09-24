#!/bin/bash
# T1-C correctness gate: identical greedy request, engine ON vs OFF; the two
# completion shas must match. Marks: GATE_HEALTH / GATE / GATE_DONE.
set -u
LOG=/tmp/t1cg.log
: > "$LOG"

run_arm() {
  local tag=$1; shift
  local extra="$*"
  pkill -f 'llama-serve[r] --model' 2>/dev/null
  sleep 4
  tmux kill-session -t srv 2>/dev/null
  tmux new-session -d -s srv "TAG=$tag UB=2048 $extra bash /root/llm/test/t1c-run.sh"
  local code=000
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8082/health)
    if [ "$code" = "200" ]; then break; fi
    sleep 5
  done
  echo "GATE_HEALTH $tag code=$code tries=$i" >> "$LOG"
  if [ "$code" != "200" ]; then
    tail -4 /root/llm/test/$tag.log >> "$LOG"
    return 1
  fi
  python3 /root/llm/test/t1c-gate-client.py "$tag" >> "$LOG" 2>&1
  pkill -f 'llama-serve[r] --model' 2>/dev/null
  sleep 4
}

run_arm g1 ""
run_arm g0 "NO79T=1"
echo "GATE_DONE $(date +%H:%M:%S)" >> "$LOG"
