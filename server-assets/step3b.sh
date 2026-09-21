#!/bin/bash
set -u
export GGML_CUDA_P2P=1
L=/root/libdir-instr
GGML_CUDA_OP_TIMING=1 NOGRAPH=1 CARDS=0,1,2 SPLIT=tensor TAG=instr-op2 L=$L P2P=1 NODROP=1 PORT=8145 bash /root/p60-ab-harness.sh
echo STEP3B_DONE; date