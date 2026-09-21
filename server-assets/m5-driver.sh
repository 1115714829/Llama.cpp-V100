#!/bin/bash
# M5 FA-Volta config sweep driver v2. Usage: bash /root/m5-driver.sh <listfile>
LIST=$1
ROOT=/root/llm/test/v100-opt/llama.cpp
SRC=$ROOT/ggml/src/ggml-cuda/fattn-mma-f16.cuh
LOG=/root/m5-log
LIB=/root/m5-lib
JOBS=24
mkdir -p $LOG $LIB
export CUDA_VISIBLE_DEVICES=3

build_it () {
  cd $ROOT
  export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
  export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
  local try
  for try in 1 2 3 4; do
    cmake --build build-instr --config Release -j$JOBS
    if [ $? -eq 0 ]; then return 0; fi
    echo "build attempt $try failed, retry in 20s"
    sleep 20
  done
  return 1
}

for tag in $(cat $LIST); do
  echo "########## $tag"
  cp /root/m5-cand/$tag.cuh $SRC
  want=$(md5sum /root/m5-cand/$tag.cuh | cut -d' ' -f1)
  have=$(md5sum $SRC | cut -d' ' -f1)
  echo "  want=$want have=$have"
  ( build_it ) > $LOG/$tag.build.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  $tag BUILD_FAIL rc=$rc"
    grep -a -e error -e Error $LOG/$tag.build.log | head -20
    continue
  fi
  src2=$(md5sum $SRC | cut -d' ' -f1)
  cp -a $ROOT/build-instr/bin/test-backend-ops $ROOT/build-instr/bin/libggml-cuda.so* $ROOT/build-instr/bin/libggml-base.so* $LIB/ 2>/dev/null
  echo "  src_after=$src2 lib_md5=$(md5sum $LIB/libggml-cuda.so.0.24.0 | cut -d' ' -f1)"
  if [ "$src2" != "$want" ]; then echo "  WARN source changed during build"; fi
  for rep in 1 2 3; do
    LD_LIBRARY_PATH=$LIB $LIB/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 -p 'nr23=\[6,1\],kv=65536,nb=512' 2>&1 | grep -a -e hsk=256 | sed -e 's/^/  A /'
    LD_LIBRARY_PATH=$LIB $LIB/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 -p 'nr23=\[6,1\],kv=4096,nb=512' 2>&1 | grep -a -e hsk=256 | sed -e 's/^/  B /'
    echo "  endrep $tag rep=$rep"
  done
done
echo M5_DRIVER_DONE
