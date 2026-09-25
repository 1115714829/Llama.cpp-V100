#!/bin/bash
# e15-fa-ab.sh - staged-baseline A/B for stage B-2 (FA kernel, Split-D mma vs TILE).
#
# A: FISHLIKEXIE_BLACK_MAGIC=0  (TILE path, the current baseline)
# B: FISHLIKEXIE_BLACK_MAGIC=2  (Split-D mma path, admits decode rows q>=1)
# ABBA order, 256K spec-on. Decision metric: ms/round (mean), tg reported alongside.
# Gate: greedy sha must stay bcda0092 (REJ off by default; FA is arithmetic, not sampling).
set -u
RUNLOG=/root/llm/test/e15-run.log
: > "$RUNLOG"

run() {
  local tag=$1; shift
  local envs="$*"
  tmux kill-session -t srv 2>/dev/null
  sleep 3
  tmux new-session -d -s srv "TAG=$tag SPEC=1 ROUND=1 UB=2048 $envs bash /root/llm/test/t1c-run.sh"
  local code=000
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8082/health)
    [ "$code" = "200" ] && break
    sleep 5
  done
  echo "HEALTH $tag code=$code $envs" >> "$RUNLOG"
  if [ "$code" != "200" ]; then
    tail -6 /root/llm/test/$tag.log >> "$RUNLOG"
    return 1
  fi
  python3 /root/llm/test/stress_client.py --base-url http://127.0.0.1:8082 \
    --model Qwen3.8-27B-Q8_0-BL2 --prompt-tokens 235930 --prompt-file bl-prompt90.txt \
    --gen 320 --reps 1 --tag $tag --timeout 1500 >> "$RUNLOG" 2>&1
  grep -a -E "draft acceptance|sm70 d256" /root/llm/test/$tag.log | tail -4 >> "$RUNLOG"
  tmux kill-session -t srv 2>/dev/null
  sleep 4
}

run e15a1 "FISHLIKEXIE_BLACK_MAGIC=0"
run e15b1 "FISHLIKEXIE_BLACK_MAGIC=2"
run e15b2 "FISHLIKEXIE_BLACK_MAGIC=2"
run e15a2 "FISHLIKEXIE_BLACK_MAGIC=0"
echo "E15_DONE rc=0" >> "$RUNLOG"
