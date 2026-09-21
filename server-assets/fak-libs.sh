#!/bin/bash
mkdir -p /root/libdir-fak
cp -a /root/llm/test/v100-opt/llama.cpp/build-instr/bin/. /root/libdir-fak/
echo "----- md5 fak -----"
md5sum /root/libdir-fak/libggml-cuda.so /root/libdir-fak/libggml-base.so /root/libdir-fak/libllama.so /root/libdir-fak/libllama-common.so
echo "----- md5 instr (baseline, untouched) -----"
md5sum /root/libdir-instr/libggml-cuda.so /root/libdir-instr/libggml-base.so /root/libdir-instr/libllama.so /root/libdir-instr/libllama-common.so
