#!/bin/bash
# P-B diagnostic: fragment/node/replay counts per round using the ALREADY COMMITTED env-gated probes (zero code change).
cd /root || exit 1
env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=pbdiag NODROP=1 NPRED=256 GGML_CUDA_GRAPH_DEBUG=1 GGML_SCHED_SPLIT_TIMING=1 GGML_CUDA_OP_TIMING=1 bash /root/p60-ab-harness.sh > /tmp/pbdiag-driver.log 2>&1
echo '=== driver tail ==='
grep -a -e prompt -e MEDIAN -e greedy -e 'RT. perf' -e 'spec timing' /tmp/pbdiag-driver.log | tail -12
echo '=== probe lines (graph/sched/op/AR) ==='
grep -a -e GRAPH -e graph -e SCHED -e sched -e capture -e replay -e direct -e 'AR.' -e op_timing /tmp/p60-pbdiag-server.log 2>/dev/null | tail -30
echo PBDIAG_DONE