#!/bin/bash
nohup bash /root/nccl-env-sweep.sh > /tmp/nccl-env-sweep.log 2>&1 < /dev/null &
echo "LAUNCHED pid=$!"
