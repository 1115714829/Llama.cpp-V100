#!/bin/bash
# bl1run.sh - BL1 pure-decode baseline (vLLM without DFlash2), model from /mnt/3.84t.
RUNLOG=/root/llm/test/bl1run.log
: > "$RUNLOG"
tmux kill-session -t bl1 2>/dev/null
sleep 3
tmux new-session -d -s bl1 "bash /tmp/bl1-start.sh > /root/llm/test/bl1nospec.log 2>&1"
code=000
for i in $(seq 1 90); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8001/health)
  [ "$code" = "200" ] && break
  sleep 5
done
echo "HEALTH code=$code" >> "$RUNLOG"
if [ "$code" != "200" ]; then
  tail -6 /root/llm/test/bl1nospec.log >> "$RUNLOG"
  tmux kill-session -t bl1 2>/dev/null
  echo "BL1RUN_DONE rc=1 health" >> "$RUNLOG"
  exit 1
fi
python3 /root/llm/test/stress_client.py --base-url http://127.0.0.1:8001 \
  --model Qwen3.8-27B-FP8 --prompt-tokens 235930 --prompt-file bl-prompt90.txt \
  --gen 320 --reps 2 --tag bl1ns --timeout 1500 >> "$RUNLOG" 2>&1
tmux kill-session -t bl1 2>/dev/null
sleep 6
echo "BL1RUN_DONE rc=0" >> "$RUNLOG"
