#!/bin/bash
# e22-fuse-ab.sh - same-source A/B for the R383 CPY+ADD fusion.
#
# Judge = the [OP] GPU-time table, not the wall clock: R378 measured +/-11 ms/round
# of wall noise, which swamps a ~3 ms effect. Diagnostic envs keep graphs/streams
# single so per-node events stay valid; kernel GPU times are unaffected.
#
# A: GGML_CUDA_NO_FUSE_CPY_ADD=1  (copy + add as two kernels)
# B: fusion on (default)
set -u
RUNLOG=/root/llm/test/e22-run.log
: > "$RUNLOG"
DIAG="GGML_CUDA_OP_TIMING=1 GGML_CUDA_DISABLE_GRAPHS=1 GGML_META_SUBGRAPH_CAPTURE=0 GGML_CUDA_NO_CONCURRENT=1"

run() {
  local tag=$1; shift
  local envs="$*"
  tmux kill-session -t srv 2>/dev/null
  sleep 3
  tmux new-session -d -s srv "TAG=$tag SPEC=1 ROUND=1 UB=2048 $DIAG $envs bash /root/llm/test/t1c-run.sh"
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
    --gen 320 --reps 1 --tag $tag --timeout 1800 >> "$RUNLOG" 2>&1
  echo "=== $tag [OP] ===" >> "$RUNLOG"
  grep -a "\[OP\]" /root/llm/test/$tag.log | tail -26 >> "$RUNLOG"
  tmux kill-session -t srv 2>/dev/null
  sleep 4
}

run e22a "GGML_CUDA_NO_FUSE_CPY_ADD=1"
run e22b ""
echo "E22_DONE rc=0" >> "$RUNLOG"
