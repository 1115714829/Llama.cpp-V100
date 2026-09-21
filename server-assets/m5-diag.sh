#!/bin/bash
# m5-diag.sh <tag> : build with instrumented fattn.cu and print kernel/config diagnostics
TAG=$1
TREE=/root/llm/test/v100-opt/llama.cpp
BD=$TREE/build-instr/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
GCUDA=$TREE/ggml/src/ggml-cuda
LIB=/root/m5-lib
OBJ=/root/m5-obj
export CUDA_VISIBLE_DEVICES=3
DEFS=$(sed -n 's/^CUDA_DEFINES = //p' $BD/flags.make | sed -e 's/\\"/"/g')
FLAGS="-O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_70,code=[sm_70] -Xcompiler=-fPIC -use_fast_math -extended-lambda"
cp /root/m5-cand/$TAG.cuh /root/m5-src/fattn-mma-f16.cuh
/usr/local/cuda-12.4/bin/nvcc -forward-unknown-to-host-compiler $DEFS --options-file $BD/includes_CUDA.rsp -I$GCUDA $FLAGS -x cu -c /root/m5-src/fattn.cu -o $OBJ/fattn.cu.o || { echo NVCC_FAIL; exit 1; }
/usr/bin/g++ -fPIC -shared -Wl,-soname,libggml-cuda.so.0 -o $LIB/libggml-cuda.so.0.24.0 @$OBJ/objects.rsp @$OBJ/linkLibs.rsp -L/usr/local/cuda-12.4/targets/ppc64le-linux/lib/stubs -L/usr/local/cuda-12.4/targets/ppc64le-linux/lib || { echo LINK_FAIL; exit 1; }
ln -sf libggml-cuda.so.0.24.0 $LIB/libggml-cuda.so.0
LD_LIBRARY_PATH=$LIB $LIB/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 -p 'nr23=\[6,1\],kv=4096,nb=512' 2>&1 | grep -a -e M5 -e hsk=256 | head -20
echo DIAG_DONE
