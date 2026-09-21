#!/bin/bash
# Stop spawning further llama-bench arms; let the current one finish and log itself.
pids=$(ps -eo pid,args | awk '/bash \/root\/lc-sweep.sh/ {print $1}')
echo "sweep_pids=$pids"
for p in $pids; do kill $p 2>/dev/null; done
sleep 2
echo '--- still running ---'
ps -eo pid,etime,args | awk '/llama-bench|llama-server/ && !/awk/ {print}'
grep -q LC_SWEEP_DONE /tmp/lc-sweep.log || echo LC_SWEEP_DONE >> /tmp/lc-sweep.log
tail -2 /tmp/lc-sweep.log