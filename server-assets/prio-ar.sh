#!/bin/bash
# Prioritize the AR small-scale test over the remaining low-value TP arms.
p=$(ps -eo pid,args | awk '/bash \/root\/lc-chain.sh/ {print $1}')
echo "chain_pids=$p"
for x in $p; do kill $x 2>/dev/null; done
s=$(pgrep -x llama-server)
echo "server_pid=$s"
for x in $s; do kill $x 2>/dev/null; done
sleep 3
for x in $s; do kill -9 $x 2>/dev/null; done
sleep 2
grep -q CHAIN_ALL_DONE /tmp/lc-chain.log || echo CHAIN_ALL_DONE >> /tmp/lc-chain.log
tail -2 /tmp/lc-chain.log
sleep 20
echo '--- ar test log ---'
cat /tmp/ar-test.log 2>/dev/null | tail -12
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader