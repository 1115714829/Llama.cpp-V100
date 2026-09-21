#!/bin/bash
export GGML_CUDA_FA_KERNEL_DEBUG=1
export CARDS=0,1,2 SPLIT=tensor L=/root/libdir-fak P2P=1 NPRED=64 PORT=8281 TAG=fak1
echo "----- harness start $(date +%H:%M:%S) -----"
bash /root/p60-ab-harness.sh
echo "HARNESS_RC=$?"
echo "----- server log FAK lines -----"
grep -a -e FAK /tmp/p60-fak1-server.log
echo "FAKCOUNT=$(grep -a -c -e FAK /tmp/p60-fak1-server.log)"
echo "----- server cmdline -----"
grep -a -m1 -e llama-server /tmp/p60-fak1-server.log
