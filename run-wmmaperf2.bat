@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\wmmaperf2.txt
scp -o BatchMode=yes wmma_perf2.cu root@192.168.50.235:/root/
%SSHH% "/usr/local/cuda-12.4/bin/nvcc -arch=sm_70 -O3 -std=c++17 --ptxas-options=-v -o /root/wmma_perf2 /root/wmma_perf2.cu 2>&1 | grep -E 'error|registers|spill' | head -10; echo ---; CUDA_VISIBLE_DEVICES=0 /root/wmma_perf2 5120 17408 20; echo ---; CUDA_VISIBLE_DEVICES=0 /root/wmma_perf2 5120 17408 50" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
