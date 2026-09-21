#!/bin/bash
SUM=/tmp/meta2.txt
: > $SUM
pkill -9 -x llama-server 2>/dev/null
sleep 5
timeout 2400 env PORT=8171 CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=metadiag2 NPRED=64 GGML_META_HOST_TIMING=1 GGML_CUDA_GRAPH_DEBUG=1 bash /root/p60-ab-harness.sh > /tmp/metadiag2-driver.log 2>&1
echo ARM_RC=$? >> $SUM
grep -a -h META /tmp/p60-metadiag2-server.log 2>/dev/null | tail -4 >> $SUM
grep -a -h GRAPH /tmp/p60-metadiag2-server.log 2>/dev/null | tail -2 >> $SUM
grep -a -h 'RT. perf' /tmp/p60-metadiag2-server.log 2>/dev/null | tail -2 >> $SUM
echo META2_DONE >> $SUM
