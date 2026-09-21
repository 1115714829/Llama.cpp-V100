#!/bin/bash
# usage: bash /root/m5-final.sh <tag>   (builds tag, then measures the full M5 shape matrix)
TAG=$1
TREE=/root/llm/test/v100-opt/llama.cpp
BD=$TREE/build-instr/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
GCUDA=$TREE/ggml/src/ggml-cuda
TI=fattn-mma-f16-instance-ncols1_32-ncols2_2
LIB=/root/m5-lib
OBJ=/root/m5-obj
LOG=/root/m5-log
export CUDA_VISIBLE_DEVICES=3
DEFS=$(sed -n 's/^CUDA_DEFINES = //p' $BD/flags.make | sed -e 's/\\"/"/g')
FLAGS="-O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_70,code=[sm_70] -Xcompiler=-fPIC -use_fast_math -extended-lambda"
INC="--options-file $BD/includes_CUDA.rsp -I$GCUDA"
cp /root/m5-cand/$TAG.cuh /root/m5-src/fattn-mma-f16.cuh
rm -f $OBJ/fattn.cu.o $OBJ/$TI.cu.o
/usr/local/cuda-12.4/bin/nvcc -forward-unknown-to-host-compiler $DEFS $INC $FLAGS -x cu -c /root/m5-src/fattn.cu -o $OBJ/fattn.cu.o > $LOG/final-$TAG.cc1.log 2>&1 || { echo CC1_FAIL; exit 1; }
/usr/local/cuda-12.4/bin/nvcc -forward-unknown-to-host-compiler $DEFS $INC $FLAGS -x cu -c /root/m5-src/ti/$TI.cu -o $OBJ/$TI.cu.o > $LOG/final-$TAG.cc2.log 2>&1 || { echo CC2_FAIL; exit 1; }
/usr/bin/g++ -fPIC -shared -Wl,-soname,libggml-cuda.so.0 -o $LIB/libggml-cuda.so.0.24.0 @$OBJ/objects.rsp @$OBJ/linkLibs.rsp -L/usr/local/cuda-12.4/targets/ppc64le-linux/lib/stubs -L/usr/local/cuda-12.4/targets/ppc64le-linux/lib > $LOG/final-$TAG.ld.log 2>&1 || { echo LINK_FAIL; exit 1; }
ln -sf libggml-cuda.so.0.24.0 $LIB/libggml-cuda.so.0
echo "lib_md5=$(md5sum $LIB/libggml-cuda.so.0.24.0 | cut -d' ' -f1)"
for rep in 1 2 3; do
  for kv in 4096 16384 65536 131072; do
    LD_LIBRARY_PATH=$LIB $LIB/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 -p "nr23=\[6,1\],kv=$kv,nb=512" 2>&1 | grep -a -e TFLOPS | sed -e "s/^/  nb512 /"
  done
  LD_LIBRARY_PATH=$LIB $LIB/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 -p "nr23=\[6,1\],kv=131072,nb=1" 2>&1 | grep -a -e TFLOPS | sed -e "s/^/  nb1   /"
  echo "  endrep $TAG rep=$rep"
done
LD_LIBRARY_PATH=$LIB $LIB/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 -p "hsk=512,hsv=512,nr23=\[8,1\],kv=49152" 2>&1 | grep -a -e TFLOPS | sed -e "s/^/  ref512 /"
echo M5_FINAL_DONE
