#!/bin/bash
LOG=/tmp/fak-poll.log
for i in $(seq 1 100); do
  LAST=$(tail -1 "$LOG" 2>/dev/null)
  case "$LAST" in
    IDLE*) echo "$LAST"; exit 0;;
    STILL_BUSY*) echo "$LAST"; exit 0;;
  esac
  sleep 30
done
echo NO_VERDICT
exit 1
