@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\map.txt
scp -o BatchMode=yes hmma_map.cu root@192.168.50.235:/root/
%SSHH% "/usr/local/cuda-12.4/bin/nvcc -arch=sm_70 -O2 -std=c++17 -o /root/hmma_map /root/hmma_map.cu 2>&1 | head -10; CUDA_VISIBLE_DEVICES=0 /root/hmma_map" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
