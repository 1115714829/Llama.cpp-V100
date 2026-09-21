#!/bin/bash
# Tidy the box: report and stop any leftover measurement processes or orphaned chains.
echo '--- before ---'
ps -eo pid,etime,args | awk '/llama-server|llama-bench|lc-par|lc-chain|a4[.]sh|d256|final[.]sh|restore|a2-ab/ && !/awk/ {print substr($0,1,110)}'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
for p in $(pgrep -x llama-server) $(pgrep -x llama-bench); do kill $p 2>/dev/null; done
for pat in 'bash /root/lc-par[.]sh' 'bash /root/lc-chain[.]sh' 'bash /root/a4[.]sh' 'bash /root/d256[.]sh' 'bash /root/final[.]sh'; do
  for p in $(ps -eo pid,args | awk -v pat="$pat" 'index($0, pat) > 0 {print $1}'); do kill $p 2>/dev/null; done
done
sleep 3
echo '--- after ---'
ps -eo pid,args | awk '/llama-server|llama-bench|lc-par|lc-chain|a4[.]sh/ && !/awk/ {print substr($0,1,110)}'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo TIDY_DONE