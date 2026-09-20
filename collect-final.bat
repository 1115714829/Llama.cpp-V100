@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes collect-final.sh root@192.168.50.235:/root/collect-final.sh
%SSHH% "bash /root/collect-final.sh" > F:\vllm+llama.cpp\1cat-vllm-v100-study\collect-final.txt 2>&1
echo DONE >> F:\vllm+llama.cpp\1cat-vllm-v100-study\collect-final.txt
