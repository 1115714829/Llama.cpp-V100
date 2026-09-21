#!/bin/bash
export GGML_CUDA_FA_KERNEL_DEBUG=1
export GGML_META_KEY_DEBUG=1
export CARDS=0,1,2 SPLIT=tensor L=/root/libdir-fak P2P=1 NPRED=64 PORT=8281 TAG=fak1
rm -f /tmp/fak-cmdline.txt /tmp/fak-env.txt
(
  for k in $(seq 1 900); do
    P=$(pgrep -x llama-server | head -1)
    if [ -n "$P" ]; then
      tr '\000' ' ' < /proc/$P/cmdline > /tmp/fak-cmdline.txt
      tr '\000' '\n' < /proc/$P/environ | grep -e GGML_ -e LD_LIBRARY > /tmp/fak-env.txt
      echo "PID=$P" >> /tmp/fak-cmdline.txt
      break
    fi
    sleep 2
  done
) &
WATCH=$!
echo "----- harness start $(date +%H:%M:%S) -----"
bash /root/p60-ab-harness.sh
echo "HARNESS_RC=$?"
wait $WATCH 2>/dev/null
echo "----- server cmdline -----"
cat /tmp/fak-cmdline.txt
echo
echo "----- probe env of server process -----"
cat /tmp/fak-env.txt
