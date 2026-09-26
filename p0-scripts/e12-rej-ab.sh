#!/bin/bash
# e12-rej-ab.sh - A/B for exact probabilistic rejection sampling (stage A).
#
# Arm A: LLAMA_SPEC_REJ=0 (greedy match accept, the old rule)
# Arm B: LLAMA_SPEC_REJ=1 (min(1, p_target/p_draft) accept, 1cat's probabilistic)
# 256K spec-on, 2 reps each, ABBA order. Expect AL 2.9 -> 3.4+ on B.
set -u
RUNLOG=/root/llm/test/e12-run.log
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
    --gen 320 --reps 2 --tag $tag --timeout 1500 >> "$RUNLOG" 2>&1
  grep -a "draft acceptance" /root/llm/test/$tag.log | tail -2 >> "$RUNLOG"
  tmux kill-session -t srv 2>/dev/null
  sleep 4
}

run e12a "LLAMA_SPEC_REJ=0"
run e12b "LLAMA_SPEC_REJ=1"
echo "E12_DONE rc=0" >> "$RUNLOG"
