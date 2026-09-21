#!/bin/bash
# usage: bash /root/m5-sweep.sh <listfile>
LIST=$1
TREE=/root/llm/test/v100-opt/llama.cpp
BD=$TREE/build-instr/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
GCUDA=$TREE/ggml/src/ggml-cuda
TI=fattn-mma-f16-instance-ncols1_32-ncols2_2
LOG=/root/m5-log
LIB=/root/m5-lib
OBJ=/root/m5-obj
mkdir -p $LOG
export CUDA_VISIBLE_DEVICES=3

DEFS=$(sed -n 's/^CUDA_DEFINES = //p' $BD/flags.make | sed -e 's/\\"/"/g')
FLAGS="-O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_70,code=[sm_70] -Xcompiler=-fPIC -use_fast_math -extended-lambda"
INC="--options-file $BD/includes_CUDA.rsp -I$GCUDA"

build_one () {
  tag=$1
  cp /root/m5-cand/$tag.cuh /root/m5-src/fattn-mma-f16.cuh
  rm -f $OBJ/fattn.cu.o $OBJ/$TI.cu.o
  /usr/local/cuda-12.4/bin/nvcc -forward-unknown-to-host-compiler $DEFS $INC $FLAGS -x cu -c /root/m5-src/fattn.cu -o $OBJ/fattn.cu.o > $LOG/$tag.cc1.log 2>&1 || return 1
  /usr/local/cuda-12.4/bin/nvcc -forward-unknown-to-host-compiler $DEFS $INC $FLAGS -x cu -c /root/m5-src/ti/$TI.cu -o $OBJ/$TI.cu.o > $LOG/$tag.cc2.log 2>&1 || return 2
  /usr/bin/g++ -fPIC -shared -Wl,-soname,libggml-cuda.so.0 -o $LIB/libggml-cuda.so.0.24.0 @$OBJ/objects.rsp @$OBJ/linkLibs.rsp -L/usr/local/cuda-12.4/targets/ppc64le-linux/lib/stubs -L/usr/local/cuda-12.4/targets/ppc64le-linux/lib > $LOG/$tag.ld.log 2>&1 || return 3
  ln -sf libggml-cuda.so.0.24.0 $LIB/libggml-cuda.so.0
  ln -sf libggml-cuda.so.0 $LIB/libggml-cuda.so
  return 0
}

for tag in $(cat $LIST); do
  echo "########## $tag"
  t0=$(date +%s)
  build_one $tag
  rc=$?
  t1=$(date +%s)
  if [ $rc -ne 0 ]; then
    echo "  BUILD_FAIL step=$rc"
    tail -15 $LOG/$tag.cc1.log
    tail -15 $LOG/$tag.cc2.log
    tail -8 $LOG/$tag.ld.log
    continue
  fi
  echo "  build_s=$((t1-t0)) obj1=$(md5sum $OBJ/fattn.cu.o | cut -d' ' -f1) obj2=$(md5sum $OBJ/$TI.cu.o | cut -d' ' -f1) lib=$(md5sum $LIB/libggml-cuda.so.0.24.0 | cut -d' ' -f1)"
  for rep in 1 2 3; do
    LD_LIBRARY_PATH=$LIB $LIB/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 -p 'nr23=\[6,1\],kv=65536,nb=512' 2>&1 | grep -a -e hsk=256 -e error -e failed | sed -e 's/^/  A /'
    LD_LIBRARY_PATH=$LIB $LIB/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 -p 'nr23=\[6,1\],kv=4096,nb=512' 2>&1 | grep -a -e hsk=256 -e error -e failed | sed -e 's/^/  B /'
    echo "  endrep $tag rep=$rep"
  done
done
echo M5_SWEEP_DONE
