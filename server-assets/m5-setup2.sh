#!/bin/bash
set -e
TREE=/root/llm/test/v100-opt/llama.cpp
BD=$TREE/build-instr/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
TI=fattn-mma-f16-instance-ncols1_32-ncols2_2
mkdir -p /root/m5-src/ti
cp -a $TREE/ggml/src/ggml-cuda/template-instances/$TI.cu /root/m5-src/ti/$TI.cu
cp -a $BD/objects1.rsp /root/m5-obj/objects_base.rsp
sed -e "s|\"CMakeFiles/ggml-cuda.dir/|\"$BD/|g" /root/m5-obj/objects_base.rsp > /root/m5-obj/objects.rsp
sed -i -e "s|\"$BD/fattn.cu.o\"|\"/root/m5-obj/fattn.cu.o\"|" /root/m5-obj/objects.rsp
sed -i -e "s|\"$BD/template-instances/$TI.cu.o\"|\"/root/m5-obj/$TI.cu.o\"|" /root/m5-obj/objects.rsp
echo "mine_fattn=$(grep -o -e /root/m5-obj/fattn.cu.o /root/m5-obj/objects.rsp | wc -l)"
echo "mine_ti=$(grep -o -e /root/m5-obj/$TI.cu.o /root/m5-obj/objects.rsp | wc -l)"
echo "stale_ti=$(grep -o -e $BD/template-instances/$TI.cu.o /root/m5-obj/objects.rsp | wc -l)"
echo "total_objs=$(grep -o -e .cu.o /root/m5-obj/objects.rsp | wc -l)"
ls -la /root/m5-src/ti/
echo SETUP2_OK
