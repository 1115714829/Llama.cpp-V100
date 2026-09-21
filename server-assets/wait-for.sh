#!/bin/bash
# wait-for.sh -- block until a terminator appears, with TWO-LEVEL safety. Always returns.
#
#   wait-for.sh <file> <pattern> [max_minutes] [stall_ticks]
#
# Level 1 (inner, every 60 s): look for the pattern.
#   Anomaly: file missing, or the file size has not changed for stall_ticks consecutive
#   ticks (default 5 = 5 minutes) -> the producer is dead or stuck -> abort at once.
#   A stall check is used instead of pgrep on purpose: pgrep -f matches this script's own
#   argv and any bash -c wrapper, which made the earlier producer check useless.
# Level 2 (outer, every 10 minutes): a HEARTBEAT line with the size, so a silent stall is
#   visible even if level 1 misbehaves. Hard cap = max_minutes.
#
# ALWAYS prints exactly one WAIT_RESULT= line:
#   OK | OK_AFTER_STALL | ABORT_NOFILE | ABORT_STALLED | TIMEOUT
set -uo pipefail

F=${1:?file}; PAT=${2:?pattern}; MAXM=${3:-30}; STALL=${4:-5}
INNER=60
OUTER_TICKS=10
CAP=$(( MAXM * 60 / INNER ))

size_of() { stat -c %s "$F" 2>/dev/null || echo 0; }

last=$(size_of); same=0
n=0; outer=0
while [ $n -lt $CAP ]; do
  if grep -q -a -e "$PAT" "$F" 2>/dev/null; then
    echo "WAIT_RESULT=OK tick=$n minutes=$(( n * INNER / 60 ))"
    exit 0
  fi
  if [ ! -e "$F" ]; then
    echo "WAIT_RESULT=ABORT_NOFILE tick=$n file=$F"
    exit 2
  fi
  cur=$(size_of)
  if [ "$cur" = "$last" ]; then
    same=$(( same + 1 ))
    if [ $same -ge $STALL ]; then
      echo "WAIT_RESULT=ABORT_STALLED tick=$n same_ticks=$same size=$cur file=$F"
      exit 5
    fi
  else
    same=0
    last=$cur
  fi
  sleep $INNER
  n=$(( n + 1 ))
  outer=$(( outer + 1 ))
  if [ $outer -ge $OUTER_TICKS ]; then
    outer=0
    echo "HEARTBEAT tick=$n size=$cur file=$F"
  fi
done
echo "WAIT_RESULT=TIMEOUT tick=$n cap_minutes=$MAXM file=$F"
exit 4
