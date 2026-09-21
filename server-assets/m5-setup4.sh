#!/bin/bash
set -e
TREE=/root/llm/test/v100-opt/llama.cpp
BD=$TREE/build-instr/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
mkdir -p /root/m5-src/ti
for TI in ncols1_8 ncols1_16 ncols1_32; do
  cp -a $TREE/ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-$TI-ncols2_2.cu /root/m5-src/ti/
done
sed -e "s|\"CMakeFiles/ggml-cuda.dir/|\"$BD/|g" /root/m5-obj/objects_base.rsp > /root/m5-obj/objects.rsp
sed -i -e "s|\"$BD/fattn.cu.o\"|\"/root/m5-obj/fattn.cu.o\"|" /root/m5-obj/objects.rsp
for TI in ncols1_8 ncols1_16 ncols1_32; do
  sed -i -e "s|\"$BD/template-instances/fattn-mma-f16-instance-$TI-ncols2_2.cu.o\"|\"/root/m5-obj/fattn-mma-f16-instance-$TI-ncols2_2.cu.o\"|" /root/m5-obj/objects.rsp
done
echo "mine_count=$(grep -o -e /root/m5-obj/fattn /root/m5-obj/objects.rsp | wc -l)"
ls /root/m5-src/ti/
echo SETUP4_OK
