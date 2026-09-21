#!/bin/bash
# Stop the FA split probe chain (lowest priority) so the GPU time goes to the AR small-scale test.
pids=$(ps -eo pid,args | awk '/bash \/root\/fa-split-probe.sh/ {print $1}')
echo "probe_pids=$pids"
for p in $pids; do kill $p 2>/dev/null; done
sleep 2
ps -eo pid,etime,args | awk '/fa-split|lc-chain|lc-par|llama-server/ && !/awk/ {print}'