#!/bin/bash
a=$(ps -eo pid,args | awk '/bash \/root\/ar-test.sh/ {print $1}')
echo "artest_pids=$a"
for x in $a; do kill $x 2>/dev/null; done
sleep 1
sed -i 's#CHAIN2_DONE#CHAIN_ALL_DONE#' /root/ar-test.sh
sed -n 1,8p /root/ar-test.sh
nohup bash /root/ar-test.sh > /tmp/ar-test.log 2>&1 < /dev/null &
echo "new_artest=$!"