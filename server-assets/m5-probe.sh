#!/bin/bash
TAG=$1
TREE=/root/llm/test/v100-opt/llama.cpp
BD=$TREE/build-instr/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
GCUDA=$TREE/ggml/src/ggml-cuda
LIB=/root/m5-lib
OBJ=/root/m5-obj
LOG=/root/m5-log
PTXAS=$2
export CUDA_VISIBLE_DEVICES=3
DEFS=$(sed -n 's/^CUDA_DEFINES = //p' $BD/flags.make | sed -e 's/\\"/"/g')
FLAGS="-O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_70,code=[sm_70] -Xcompiler=-fPIC -use_fast_math -extended-lambda"
INC="--options-file $BD/includes_CUDA.rsp -I$GCUDA"
cp /root/m5-cand/$TAG.cuh /root/m5-src/fattn-mma-f16.cuh
rm -f $OBJ/*.cu.o
t0=$(date +%s)
/usr/local/cuda-12.4/bin/nvcc -forward-unknown-to-host-compiler $DEFS $INC $FLAGS -x cu -c /root/m5-src/fattn.cu -o $OBJ/fattn.cu.o > $LOG/probe-$TAG.cc1.log 2>&1 || { echo CC1_FAIL; tail -12 $LOG/probe-$TAG.cc1.log; exit 1; }
for TI in ncols1_8 ncols1_16 ncols1_32; do
  /usr/local/cuda-12.4/bin/nvcc -forward-unknown-to-host-compiler $DEFS $INC $FLAGS $PTXAS -x cu -c /root/m5-src/ti/fattn-mma-f16-instance-$TI-ncols2_2.cu -o $OBJ/fattn-mma-f16-instance-$TI-ncols2_2.cu.o > $LOG/probe-$TAG.$TI.log 2>&1 || { echo CC_FAIL_$TI; tail -12 $LOG/probe-$TAG.$TI.log; exit 1; }
done
/usr/bin/g++ -fPIC -shared -Wl,-soname,libggml-cuda.so.0 -o $LIB/libggml-cuda.so.0.24.0 @$OBJ/objects.rsp @$OBJ/linkLibs.rsp -L/usr/local/cuda-12.4/targets/ppc64le-linux/lib/stubs -L/usr/local/cuda-12.4/targets/ppc64le-linux/lib > $LOG/probe-$TAG.ld.log 2>&1 || { echo LINK_FAIL; exit 1; }
ln -sf libggml-cuda.so.0.24.0 $LIB/libggml-cuda.so.0
t1=$(date +%s)
echo "lib_md5=$(md5sum $LIB/libggml-cuda.so.0.24.0 | cut -d' ' -f1) build_s=$((t1-t0))"
grep -a -e registers -e spill /root/m5-log/probe-$TAG.ncols1_32.log | head -6
run_p () {
  LD_LIBRARY_PATH=$LIB $LIB/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 -p "$1" 2>&1 | grep -a -e TFLOPS -e CUDA | sed -e "s/^/  $2 /"
}
for rep in 1 2 3; do
  for nb in 8 16 64; do
    run_p "nr23=\[6,1\],kv=8192,nb=$nb" "kv8192_nb$nb"
  done
  run_p "nr23=\[6,1\],kv=131072,nb=8" "kv131072_nb8"
  run_p "nr23=\[6,1\],kv=65536,nb=512" "kv65536_nb512"
  echo "  endrep $TAG rep=$rep"
done
echo M5_PROBE_DONE
