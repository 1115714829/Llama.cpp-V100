#!/bin/bash
# Yield the machine to the P-C implementation subagent: stop the zero-code Z2/Z3/Z4 chain.
for pat in 'bash /root/tpsweep.sh' 'bash /root/z34.sh'; do
  for p in $(ps -eo pid,args | awk -v pat="$pat" 'index($0, pat) > 0 {print $1}'); do echo "killing $p ($pat)"; kill $p 2>/dev/null; done
done
sleep 2
for p in $(pgrep -x llama-server) $(pgrep -x llama-bench); do kill $p 2>/dev/null; done
sleep 3
for p in $(pgrep -x llama-server) $(pgrep -x llama-bench); do kill -9 $p 2>/dev/null; done
sleep 2
echo '--- remaining measurement procs ---'
ps -eo pid,args | awk '/llama-server|llama-bench|tpsweep|z34|nmax/ && !/awk/ {print substr($0,1,90)}'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo ZCHAIN_STOPPED