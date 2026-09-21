#!/bin/bash
# Wait for the TP-degree arms to finish, then run the device-side push allreduce test alone.
while ! grep -q CHAIN_ALL_DONE /tmp/lc-chain.log 2>/dev/null; do sleep 30; done
echo AR_TEST_START
echo '=== 164 KB x 1000 (in-situ size, stress slot reuse) ==='
CUDA_VISIBLE_DEVICES=0,1,2 /root/ar_push_test 40960 1000
echo '=== 1 MB x 500 ==='
CUDA_VISIBLE_DEVICES=0,1,2 /root/ar_push_test 262144 500
echo '=== 10 MB x 100 ==='
CUDA_VISIBLE_DEVICES=0,1,2 /root/ar_push_test 2500000 100
echo AR_TEST_DONE