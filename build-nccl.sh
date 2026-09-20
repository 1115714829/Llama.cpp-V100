#!/bin/bash
nohup bash /root/build-nccl-inner.sh > /tmp/build-nccl.log 2>&1 < /dev/null &
echo "LAUNCHED pid=$!"
