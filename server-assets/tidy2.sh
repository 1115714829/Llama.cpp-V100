#!/bin/bash
# Kill any remaining sentinel-waiting loops (they match 'n=0; while' in their argv).
pids=$(ps -eo pid,args | awk 'index($0, "n=0; while") > 0 && index($0, "awk") == 0 {print $1}')
echo "killing: $pids"
for p in $pids; do kill -9 $p 2>/dev/null; done
sleep 2
echo -n 'remaining: '; ps -eo args | grep -a -c 'n=0; while'
echo TIDY2_DONE