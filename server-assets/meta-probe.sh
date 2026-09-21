#!/bin/bash
SUM=/tmp/meta-probe.txt
: > $SUM
n=0
while [ $n -lt 160 ]; do
  if ! pgrep -x llama-server > /dev/null 2>&1; then break; fi
  sleep 15; n=$((n+1))
done
echo WAITED_15s=$n >> $SUM
cd /root/llm/test/v100-opt/llama.cpp || exit 1
cmake --build build-instr --config Release -j128 > /tmp/meta-build.log 2>&1
echo BUILD_RC=$? >> $SUM
echo ERR_LINES=$(grep -c -e error /tmp/meta-build.log) >> $SUM
echo BUILT_TARGETS=$(grep -c 'Built target' /tmp/meta-build.log) >> $SUM
cp -a build-instr/bin/. /root/libdir-instr/
echo MARKER=$(strings /root/libdir-instr/libllama.so.0.4.1 | grep -a -c 'META. calls=') >> $SUM
md5sum /root/libdir-instr/libllama.so.0.4.1 /root/libdir-instr/libggml-cuda.so.0.24.0 >> $SUM
pkill -9 -x llama-server 2>/dev/null
sleep 5
timeout 2400 env PORT=8161 CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=metadiag NPRED=64 GGML_META_HOST_TIMING=1 bash /root/p60-ab-harness.sh > /tmp/metadiag-driver.log 2>&1
echo ARM_RC=$? >> $SUM
grep -a -h META /tmp/p60-metadiag-server.log 2>/dev/null | tail -8 >> $SUM
grep -a -h 'RT. perf' /tmp/p60-metadiag-server.log 2>/dev/null >> $SUM
echo META_PROBE_DONE >> $SUM
