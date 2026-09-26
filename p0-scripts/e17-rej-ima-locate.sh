#!/bin/bash
# e17-rej-ima-locate.sh - locate the IMA under LLAMA_SPEC_REJ=1 with CUDA_LAUNCH_BLOCKING=1.
# One 256K spec-on rep; the blocking launcher names the kernel that faults.
set -u
RUNLOG=/root/llm/test/e17-run.log
: > "$RUNLOG"
tmux kill-session -t srv 2>/dev/null
sleep 3
tmux new-session -d -s srv "TAG=e17 SPEC=1 ROUND=1 UB=2048 LLAMA_SPEC_REJ=1 CUDA_LAUNCH_BLOCKING=1 bash /root/llm/test/t1c-run.sh"
code=000
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8082/health)
  [ "$code" = "200" ] && break
  sleep 5
done
echo "HEALTH code=$code" >> "$RUNLOG"
if [ "$code" != "200" ]; then
  tail -6 /root/llm/test/e17.log >> "$RUNLOG"
  echo "E17_DONE rc=1 health" >> "$RUNLOG"
  exit 1
fi
python3 /root/llm/test/stress_client.py --base-url http://127.0.0.1:8082 \
  --model Qwen3.8-27B-Q8_0-BL2 --prompt-tokens 235930 --prompt-file bl-prompt90.txt \
  --gen 320 --reps 1 --tag e17 --timeout 1500 >> "$RUNLOG" 2>&1
sleep 2
echo "=== IMA / kernel info ===" >> "$RUNLOG"
grep -a -B4 -A10 -E "CUDA error|illegal|assert" /root/llm/test/e17.log | tail -30 >> "$RUNLOG"
tmux kill-session -t srv 2>/dev/null
sleep 5
echo "E17_DONE rc=0" >> "$RUNLOG"
