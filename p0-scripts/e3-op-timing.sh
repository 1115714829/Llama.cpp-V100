#!/bin/bash
# e3-op-timing.sh - per-op GPU timing table for the 256K spec-on decode step.
#
# GGML_CUDA_OP_TIMING=1 records a start/stop CUDA event around every graph node
# (events land on the stream, so elapsed = real GPU time, not host submit wall).
# GGML_CUDA_DISABLE_GRAPHS=1 is required: per-node events are only valid when the
# graph is not captured. GPU kernel times are the same with or without capture,
# so the table is valid for attribution even though the wall clock differs.
set -u
RUNLOG=/root/llm/test/e3-run.log
: > "$RUNLOG"
pkill -f 'llama-serve[r] --model' 2>/dev/null
sleep 4
tmux kill-session -t srv 2>/dev/null
tmux new-session -d -s srv "TAG=e3 SPEC=1 ROUND=1 UB=2048 GGML_CUDA_OP_TIMING=1 GGML_CUDA_DISABLE_GRAPHS=1 GGML_META_SUBGRAPH_CAPTURE=0 bash /root/llm/test/t1c-run.sh"
code=000
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8082/health)
  [ "$code" = "200" ] && break
  sleep 5
done
echo "HEALTH code=$code" >> "$RUNLOG"
if [ "$code" != "200" ]; then
  tail -4 /root/llm/test/e3.log >> "$RUNLOG"
  echo "E3_DONE rc=1" >> "$RUNLOG"
  exit 1
fi
python3 /root/llm/test/stress_client.py --base-url http://127.0.0.1:8082 \
  --model Qwen3.8-27B-Q8_0-BL2 --prompt-tokens 235930 --prompt-file bl-prompt90.txt \
  --gen 320 --reps 1 --tag e3 --timeout 1800 >> "$RUNLOG" 2>&1
sleep 3
pkill -f 'llama-serve[r] --model' 2>/dev/null
sleep 4
echo "=== [OP] tables (last 3 reports) ===" >> "$RUNLOG"
grep -a "\[OP\]" /root/llm/test/e3.log | tail -90 >> "$RUNLOG"
echo "E3_DONE rc=0" >> "$RUNLOG"
