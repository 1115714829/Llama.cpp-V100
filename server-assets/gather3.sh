#!/bin/bash
# Read-only: full run history of the summary timing lines from both test units.
set -uo pipefail
for u in llama-server-c4 llama-server-test; do
  echo "########## $u : ALL summary timing lines ##########"
  journalctl -u "$u" --no-pager -o short-iso 2>/dev/null \
    | grep -E "prompt eval time|   eval time =|total time =|n_gen =|draft acceptance|backend offload failed|model loaded|slot released|main: server is listening" \
    | sed 's/  */ /g'
  echo ""
  echo "---------- $u restart count (Started) ----------"
  journalctl -u "$u" --no-pager -o short-iso 2>/dev/null | grep -cE "Started|Starting"
  echo ""
done
echo GATHER3_DONE
