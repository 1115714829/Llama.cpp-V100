#!/bin/bash
# T1-C smoke client: waits for health, then one short request through stress_client.py.
#   TAG=dq2 PT=2560 GEN=8 bash t1c-stress.sh
set -u
TAG=${TAG:-dq2}
PT=${PT:-2560}
GEN=${GEN:-8}
REPS=${REPS:-1}
SLOG=/root/llm/test/$TAG-stress.log
: > "$SLOG"
echo "SMOKE_START $(date)" >> "$SLOG"
ok=0
for i in $(seq 1 120); do
  if curl -sf -m 3 http://127.0.0.1:8082/health >/dev/null 2>&1; then ok=1; break; fi
  sleep 5
done
echo "HEALTH_OK=$ok after $i tries" >> "$SLOG"
if [ "$ok" != "1" ]; then echo "SMOKE_DONE" >> "$SLOG"; exit 1; fi
python3 /root/llm/test/stress_client.py \
  --base-url http://127.0.0.1:8082 \
  --model ${MODEL:-Qwen3.8-27B-Q8_0-BL2} \
  --prompt-tokens "$PT" --gen "$GEN" --reps "$REPS" --tag "$TAG" --timeout 900 \
  >> "$SLOG" 2>&1
echo "STRESS_RC=$?" >> "$SLOG"
echo "SMOKE_DONE" >> "$SLOG"
