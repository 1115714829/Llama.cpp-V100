#!/bin/bash
echo "--- listeners on 8281 ---"
ss -lptn 2>/dev/null | grep -e 8281
PIDS=$(ss -lptn 2>/dev/null | grep -e 8281 | grep -o -e 'pid=[0-9]*' | cut -d= -f2 | sort -u)
echo "PORT8281_PIDS=[$PIDS]"
for p in $PIDS; do
  echo "--- pid $p ---"
  tr '\000' '\n' < /proc/$p/environ | grep -e libdir-fak -e GGML_CUDA_FA_KERNEL_DEBUG
  tr '\000' ' ' < /proc/$p/cmdline | cut -c1-300
  echo
  kill -9 $p && echo "KILLED $p"
done
sleep 2
echo "--- remaining llama-server ---"
pgrep -a -x llama-server || echo NONE
