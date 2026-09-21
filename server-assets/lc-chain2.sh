#!/bin/bash
# Chain 2: tensor-parallel degree at long context (zero-code lever for the 256K agent scenario).
while ! grep -q CHAIN_ALL_DONE /tmp/lc-chain.log 2>/dev/null; do sleep 30; done
echo CHAIN2_START
env TAG=lc64-tp4-q8 CARDS=0,1,2,3 KV=q8_0 CTX=65536 NPRED=192 bash /root/lc-spec.sh
env TAG=lc64-tp4-f16 CARDS=0,1,2,3 KV=f16 CTX=65536 NPRED=192 bash /root/lc-spec.sh
env TAG=lc64-tp6-q8 CARDS=0,1,2,3,4,5 KV=q8_0 CTX=65536 NPRED=192 bash /root/lc-spec.sh
echo CHAIN2_DONE