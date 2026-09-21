#!/bin/bash
set -e
TREE=/root/llm/test/v100-opt/llama.cpp
BD=$TREE/build-instr
mkdir -p /root/m5-lib
cp -a $BD/bin/libllama-common.so.0.4.1 $BD/bin/libllama.so.0.4.1 $BD/bin/libggml.so.0.24.0 $BD/bin/libggml-cpu.so.0.24.0 /root/m5-lib/ 2>/dev/null || true
ls /root/m5-lib/ | tr '\n' ' '
echo
echo SETUP3_OK
