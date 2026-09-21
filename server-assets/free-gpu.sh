#!/bin/bash
# Free the GPUs: the scalar end-to-end arm already answered the question; the next step is a kernel param sweep.
for pat in '/bash /root/ar-off2.sh' '/bash /root/ar-vec.sh' '/bash /root/p60-ab-harness.sh'; do
  for p in $(ps -eo pid,args | awk -v pat="$pat" '$0 ~ pat {print $1}'); do kill $p 2>/dev/null; done
done
sleep 2
for p in $(pgrep -x llama-server); do kill $p 2>/dev/null; done
sleep 3
for p in $(pgrep -x llama-server); do kill -9 $p 2>/dev/null; done
sleep 2
echo '--- remaining ---'
ps -eo pid,args | awk '/ar-off2|ar-vec|p60-ab|llama-server/ && !/awk/ {print}'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -3