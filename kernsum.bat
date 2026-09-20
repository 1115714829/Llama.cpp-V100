@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes nsys-stats-qdstrm.sh root@192.168.50.235:/root/nsys-stats-qdstrm.sh
%SSHH% "bash /root/nsys-stats-qdstrm.sh" > F:\vllm+llama.cpp\1cat-vllm-v100-study\kernsum.txt 2>&1
echo DONE >> F:\vllm+llama.cpp\1cat-vllm-v100-study\kernsum.txt
