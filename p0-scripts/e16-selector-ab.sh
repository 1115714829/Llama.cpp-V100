#!/bin/bash
# e16-selector-ab.sh - stage C-1: in-graph GPU selector vs CPU selector.
#
# A: GGML_SPEC_SELECTOR_INGRAPH=0 (CPU selector, baseline)
# B: GGML_SPEC_SELECTOR_INGRAPH=1 (in-graph lattice walk, GPU)
# ABBA, 256K spec-on. Metric: ms/round (mean), tg alongside.
set -u
RUNLOG=/root/llm/test/e16-run.log
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
  grep -a "draft acceptance" /root/llm/test/$tag.log | tail -1 >> "$RUNLOG"
  tmux kill-session -t srv 2>/dev/null
  sleep 4
}

run e16a1 "GGML_SPEC_SELECTOR_INGRAPH=0"
run e16b1 "GGML_SPEC_SELECTOR_INGRAPH=1"
run e16b2 "GGML_SPEC_SELECTOR_INGRAPH=1"
run e16a2 "GGML_SPEC_SELECTOR_INGRAPH=0"
echo "E16_DONE rc=0" >> "$RUNLOG"
