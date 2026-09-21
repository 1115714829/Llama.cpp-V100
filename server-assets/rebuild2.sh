#!/bin/bash
cd /root/llm/test/v100-opt/llama.cpp
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
cmake --build build-instr --config Release -j82
cp -a build-instr/bin/. /root/libdir-instr/
md5sum /root/libdir-instr/libggml-cuda.so.0.24.0
echo REBUILD2_DONE