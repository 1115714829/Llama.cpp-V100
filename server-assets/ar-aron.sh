#!/bin/bash
# Wait for the forced rebuild, verify the binary marker, then run the device-AR arm only.
while ! grep -q ARDEV3_DONE /tmp/rebuild-ar.log 2>/dev/null; do sleep 20; done
echo '--- rebuild log ---'; cat /tmp/rebuild-ar.log
M=$(strings /root/libdir-instr/libggml-cuda.so.0.24.0 | grep -a -c GGML_CUDA_AR_DEVICE)
echo "marker=$M"
if [ "$M" = "0" ]; then echo ARON_DONE STATUS=NO_MARKER; exit 1; fi
cd /root
env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=aron ARENV=DEVICE=1 NPRED=512 bash /root/p60-ab-harness.sh 2>&1 | tail -30
echo '--- device AR evidence ---'
grep -a -e 'device-side push AllReduce enabled' -e 'device-side AllReduce init failed' /tmp/p60-aron-server.log 2>/dev/null | head -3
grep -a ar_us_avg /tmp/p60-aron-server.log 2>/dev/null | tail -1
grep -a 'RT. perf' /tmp/p60-aron-server.log 2>/dev/null | grep -a -e Qwen3.8-27B | tail -2
grep -a 'spec timing' /tmp/p60-aron-server.log 2>/dev/null | tail -1
echo ARON_DONE