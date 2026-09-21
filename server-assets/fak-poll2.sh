#!/bin/bash
LOG=/tmp/fak-poll2.log
: > $LOG
IDLE=0
for i in $(seq 1 45); do
  A=$(pgrep -f z1-arm | wc -l)
  B=$(pgrep -f z1-harness | wc -l)
  C=$(pgrep -x llama-server | wc -l)
  D=$(pgrep -x llama-bench | wc -l)
  echo "poll $i $(date +%H:%M:%S) z1arm=$A z1harness=$B server=$C bench=$D" >> $LOG
  if [ "$A" = "0" ] && [ "$B" = "0" ] && [ "$C" = "0" ] && [ "$D" = "0" ]; then
    IDLE=$((IDLE+1))
  else
    IDLE=0
  fi
  if [ "$IDLE" -ge 2 ]; then
    echo "IDLE_CONFIRMED after $i polls (2 consecutive empty)" >> $LOG
    pgrep -a -x llama-bench >> $LOG 2>&1
    bash /root/fak-build-go.sh >> $LOG 2>&1
    echo "LAUNCH_DONE rc=$?" >> $LOG
    exit 0
  fi
  sleep 60
done
echo "STILL_BUSY after 45 polls" >> $LOG
exit 1
