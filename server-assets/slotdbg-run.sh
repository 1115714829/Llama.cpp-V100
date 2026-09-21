#!/bin/bash
set -u
pkill -9 -x llama-server
sleep 2
for i in $(seq 1 90); do
  if ! (exec 3<>/dev/tcp/127.0.0.1/8151) 2>/dev/null; then break; fi
  sleep 1
done
echo "leftover_llama_server=$(pgrep -xc llama-server)"
echo "gpu_before:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
nohup env PORT=8151 CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=slotdbg NPRED=64 LLAMA_GRAPH_SLOT_DEBUG=1 bash /root/p60-ab-harness.sh > /tmp/p60-slotdbg-run.log 2>&1 < /dev/null &
echo "LAUNCHED pid=$!"
