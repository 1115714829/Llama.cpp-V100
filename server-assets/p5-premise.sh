#!/bin/bash
# P5 premise test: on 2 cards the internal pipeline IS available (n_devices==2).
# If internal (its own push-ish chunked kernel) beats NCCL at the same size,
# generalizing it to N devices is worth the work; if not, the 123 us/collective
# may be inherent and P5's expected +30% is in doubt.
set -u
export GGML_CUDA_P2P=1
L=/root/libdir-instr
echo ARM-NCCL; date
GGML_CUDA_ALLREDUCE=nccl CARDS=0,1 SPLIT=tensor TAG=p5pre-nccl L=$L P2P=1 NODROP=1 PORT=8170 bash /root/p60-ab-harness.sh
echo ARM-INTERNAL; date
GGML_CUDA_ALLREDUCE=internal CARDS=0,1 SPLIT=tensor TAG=p5pre-int L=$L P2P=1 NODROP=1 PORT=8171 bash /root/p60-ab-harness.sh
echo ARM-BUTTERFLY; date
GGML_CUDA_ALLREDUCE=none CARDS=0,1 SPLIT=tensor TAG=p5pre-bf L=$L P2P=1 NODROP=1 PORT=8172 bash /root/p60-ab-harness.sh
echo P5PRE_DONE; date