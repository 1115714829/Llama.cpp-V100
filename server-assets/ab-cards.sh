#!/bin/bash
set -u
export GGML_CUDA_P2P=1
L=/root/libdir-instr
echo ARM1 TP3 tensor; date
CARDS=0,1,2 SPLIT=tensor TAG=ab-tp3b L=$L P2P=1 NODROP=1 PORT=8150 bash /root/p60-ab-harness.sh
echo ARM2 TP4 tensor; date
CARDS=0,1,2,3 SPLIT=tensor TAG=ab-tp4b L=$L P2P=1 NODROP=1 PORT=8151 bash /root/p60-ab-harness.sh
echo ARM3 TP2 tensor; date
CARDS=0,1 SPLIT=tensor TAG=ab-tp2b L=$L P2P=1 NODROP=1 PORT=8152 bash /root/p60-ab-harness.sh
echo AB_CARDS_DONE; date