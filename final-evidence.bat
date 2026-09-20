@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\final-evidence.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === A) our quant type (q8_0) at n=1 vs n=8, and q4_K for contrast === > "%OUT%"
%SSHH% "grep -a -A1 'type_a=q8_0,type_b=f32,m=4096,n=1,k=14336\|type_a=q8_0,type_b=f32,m=4096,n=8,k=14336\|type_a=q4_K,type_b=f32,m=14336,n=1,k=4096\|type_a=q4_K,type_b=f32,m=14336,n=8,k=4096' /tmp/mb-mulmat.log" >> "%OUT%" 2>&1
echo === B) my best HMMA prototype (N=17408 K=5120 8 tokens) === >> "%OUT%"
%SSHH% "CUDA_VISIBLE_DEVICES=0 /root/wmma_perf3 5120 17408 20" >> "%OUT%" 2>&1
echo === C) WMMA correctness probe === >> "%OUT%"
%SSHH% "CUDA_VISIBLE_DEVICES=0 /root/wmma_bench 64 2>&1 | grep -E 'probe|GEMM'" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
cd /d F:\vllm+llama.cpp\llama.cpp
echo === D) git status --porcelain (repo root = llama.cpp) === >> "%OUT%"
git status --porcelain >> "%OUT%" 2>&1
echo DONE2 >> "%OUT%"
