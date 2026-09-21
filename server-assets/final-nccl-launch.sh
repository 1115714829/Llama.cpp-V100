#!/bin/bash
nohup bash /root/final-nccl.sh > /tmp/final-nccl.log 2>&1 < /dev/null &
echo "LAUNCHED pid=$!"
