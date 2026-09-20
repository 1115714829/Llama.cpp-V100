@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes hmma_layout.cu run-hmma-layout.sh root@192.168.50.235:/root/
%SSHH% "bash /root/run-hmma-layout.sh" > F:\vllm+llama.cpp\1cat-vllm-v100-study\layout1.txt 2>&1
echo DONE >> F:\vllm+llama.cpp\1cat-vllm-v100-study\layout1.txt
