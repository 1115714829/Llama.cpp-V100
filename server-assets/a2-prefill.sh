#!/bin/bash
# A2 scope test: 32K prefill and per-chunk graph allocation, canonical lib vs A2 lib.
while ! grep -q A2_DONE /tmp/a2-run.log 2>/dev/null; do sleep 20; done
echo A2_SEEN
echo '--- lib fingerprints ---'
md5sum /root/libdir-canon/libllama.so.0.4.1 /root/libdir-instr/libllama.so.0.4.1
echo '=== ARM pre-baseline (canonical lib), CTX=32768 ==='
env TAG=a2pre CARDS=0,1,2 L=/root/libdir-canon KV=q8_0 CTX=32768 NPRED=8 bash /root/lc-spec.sh 2>&1 | grep -a -e prompt -e MEDIAN -e perf -e timings
echo '=== ARM with A2, CTX=32768 ==='
env TAG=a2post CARDS=0,1,2 L=/root/libdir-instr KV=q8_0 CTX=32768 NPRED=8 bash /root/lc-spec.sh 2>&1 | grep -a -e prompt -e MEDIAN -e perf -e timings
echo A2PRE_DONE