#!/bin/bash
# 1) kill the duplicate TP chain (lc-chain2.sh)  2) restart the AR test waiter on CHAIN_ALL_DONE
p=$(ps -eo pid,args | awk '/bash \/root\/lc-chain2.sh/ {print $1}')
echo "chain2_pids=$p"
for x in $p; do kill $x 2>/dev/null; done
a=$(ps -eo pid,args | awk '/bash \/root\/ar-test.sh/ {print $1}')
echo "artest_pids=$a"
for x in $a; do kill $x 2>/dev/null; done
sleep 2
sed -i 's#/tmp/lc-chain2.log#/tmp/lc-chain.log#' /root/ar-test.sh
grep -n CHAIN /root/ar-test.sh | head -3
nohup bash /root/ar-test.sh > /tmp/ar-test.log 2>&1 < /dev/null &
echo "new_artest=$!"
sleep 2
ps -eo pid,etime,args | awk '/lc-chain|ar-test|fa-split/ && !/awk/ {print}'