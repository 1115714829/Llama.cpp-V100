#!/bin/bash
# e10-clean-stress.sh - clean 256K DFlash2 stress, 3 reps, no diagnostics.
#
# Answers "where are we now" on the current binary (TILE q8_0 direct + lm_head
# granularity + split floor reverted). No OP_TIMING / no graph disable / no env.
# 3 reps because wall-clock A/B noise is +/-9 ms per round (R378).
set -u
RUNLOG=/root/llm/test/e10-run.log
: > "$RUNLOG"
pkill -f 'llama-serve[r] --model' 2>/dev/null
sleep 4
tmux kill-session -t srv 2>/dev/null
tmux new-session -d -s srv "TAG=e10 SPEC=1 ROUND=1 UB=2048 bash /root/llm/test/t1c-run.sh"
code=000
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8082/health)
  [ "$code" = "200" ] && break
  sleep 5
done
echo "HEALTH code=$code" >> "$RUNLOG"
if [ "$code" != "200" ]; then
  tail -4 /root/llm/test/e10.log >> "$RUNLOG"
  echo "E10_DONE rc=1" >> "$RUNLOG"
  exit 1
fi
python3 /root/llm/test/stress_client.py --base-url http://127.0.0.1:8082 \
  --model Qwen3.8-27B-Q8_0-BL2 --prompt-tokens 235930 --prompt-file bl-prompt90.txt \
  --gen 320 --reps 3 --tag e10 --timeout 1500 >> "$RUNLOG" 2>&1
grep -a -E "draft acceptance" /root/llm/test/e10.log | tail -3 >> "$RUNLOG"
pkill -f 'llama-serve[r] --model' 2>/dev/null
sleep 4
echo "E10_DONE rc=0" >> "$RUNLOG"
