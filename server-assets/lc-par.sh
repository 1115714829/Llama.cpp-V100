#!/bin/bash
# Round 43: long-context SPEC yardstick on cards 3,4,5 (parallel to the llama-bench sweep on 0,1,2).
echo PAR_START
env TAG=lc32-q8 CARDS=3,4,5 KV=q8_0 CTX=32768 NPRED=192 bash /root/lc-spec.sh
env TAG=lc32-f16 CARDS=3,4,5 KV=f16 CTX=32768 NPRED=192 bash /root/lc-spec.sh
env TAG=lc64-q8 CARDS=3,4,5 KV=q8_0 CTX=65536 NPRED=192 bash /root/lc-spec.sh
env TAG=lc64-f16 CARDS=3,4,5 KV=f16 CTX=65536 NPRED=192 bash /root/lc-spec.sh
echo PAR_DONE