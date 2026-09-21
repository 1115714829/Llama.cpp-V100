#!/bin/bash
NCCL=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime
export LD_LIBRARY_PATH=$NCCL/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2
echo '=== 164 KB (in-situ size) x 1000 ==='
/root/ar_push_test 40960 1000
echo '=== 1 MB x 500 ==='
/root/ar_push_test 262144 500
echo '=== 10 MB x 100 ==='
/root/ar_push_test 2500000 100
echo AR_RUN_DONE