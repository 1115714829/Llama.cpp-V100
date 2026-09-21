#!/bin/bash
set -e
TREE=/root/llm/test/v100-opt/llama.cpp
BD=$TREE/build-instr/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir
mkdir -p /root/m5-src /root/m5-obj /root/m5-lib
cp -a $TREE/ggml/src/ggml-cuda/fattn.cu /root/m5-src/fattn.cu
cp -a $TREE/build-instr/bin/test-backend-ops $TREE/build-instr/bin/libggml-cuda.so.0.24.0 $TREE/build-instr/bin/libggml-base.so.0.24.0 $TREE/build-instr/bin/libggml.so.0.24.0 /root/m5-lib/ 2>/dev/null || true
sed -e "s|\"CMakeFiles/ggml-cuda.dir/|\"$BD/|g" $BD/objects1.rsp > /root/m5-obj/objects.rsp
sed -i -e "s|\"$BD/fattn.cu.o\"|\"/root/m5-obj/fattn.cu.o\"|" /root/m5-obj/objects.rsp
cat > /root/m5-obj/linkLibs.rsp <<'EOF'
 -Wl,-rpath,/root/m5-lib:/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/lib "/root/m5-lib/libggml-base.so.0.24.0" "/usr/local/cuda-12.4/targets/ppc64le-linux/lib/libcudart.so" "/usr/local/cuda-12.4/targets/ppc64le-linux/lib/libcublas.so" /usr/lib64/libcuda.so "/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/lib/libnccl.so.2" "/usr/local/cuda-12.4/targets/ppc64le-linux/lib/libcublasLt.so" "/usr/local/cuda-12.4/targets/ppc64le-linux/lib/libculibos.a" -lcudadevrt -lcudart_static -lrt -lpthread -ldl 
EOF
echo "objects_with_fattn=$(grep -o -e /root/m5-obj/fattn.cu.o /root/m5-obj/objects.rsp | wc -l)"
echo "objects_total=$(grep -o -e .cu.o /root/m5-obj/objects.rsp | wc -l)"
cp -a /root/m5-pristine.cuh /root/m5-cand/base.cuh
ls -la /root/m5-lib/ | head -12
echo SETUP_OK
