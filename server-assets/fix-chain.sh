#!/bin/bash
# Kill any running lc-chain.sh waiter, then start a single fresh one.
pids=$(ps -eo pid,args | awk '/bash \/root\/lc-chain.sh/ {print $1}')
echo "old_chain_pids=$pids"
for p in $pids; do kill $p 2>/dev/null; done
sleep 2
echo '--- remaining chain/par processes ---'
ps -eo pid,etime,args | awk '/lc-chain.sh|lc-par.sh/ && !/awk/ {print}'
nohup bash /root/lc-chain.sh > /tmp/lc-chain.log 2>&1 < /dev/null &
echo "new_chain_pid=$!"
sleep 2
echo '--- after ---'
ps -eo pid,etime,args | awk '/lc-chain.sh|lc-par.sh/ && !/awk/ {print}'