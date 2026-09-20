@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\wmma.txt
scp -o BatchMode=yes wmma_q8_bench.cu root@192.168.50.235:/root/
%SSHH% "/usr/local/cuda-12.4/bin/nvcc -arch=sm_70 -O2 -std=c++17 -o /root/wmma_bench /root/wmma_q8_bench.cu 2>&1 | head -20; echo ---; CUDA_VISIBLE_DEVICES=0 /root/wmma_bench 64; CUDA_VISIBLE_DEVICES=0 /root/wmma_bench 5120" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
