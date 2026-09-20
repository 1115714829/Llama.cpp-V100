#!/bin/bash
nohup bash /root/nccl-ab.sh > /tmp/nccl-ab.log 2>&1 < /dev/null &
echo "LAUNCHED pid=$!"
