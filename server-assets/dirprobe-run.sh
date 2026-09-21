#!/bin/bash
set -uo pipefail
pkill -9 -x llama-server
sleep 5
n=0
while [ "$n" -lt 90 ]; do
  if ! (exec 3<>/dev/tcp/127.0.0.1/8191) 2>/dev/null; then
    break
  fi
  sleep 3
  n=$((n+3))
done
echo "PORT_WAIT_SEC=$n"
pgrep -x llama-server
echo "PGREP_RC=$?"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
date
timeout 2400 env PORT=8191 CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=dirprobe NPRED=512 GGML_CUDA_DIRECT_DEBUG=1 GGML_CUDA_GRAPH_DEBUG=1 bash /root/p60-ab-harness.sh > /tmp/dirprobe-driver.log 2>&1
echo "DRIVER_RC=$?"
date
echo "DIRPROBE_RUN_DONE"
