@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes revert-q8.sh root@192.168.50.235:/root/revert-q8.sh
%SSHH% "bash /root/revert-q8.sh" > F:\vllm+llama.cpp\1cat-vllm-v100-study\revert.txt 2>&1
echo DONE >> F:\vllm+llama.cpp\1cat-vllm-v100-study\revert.txt
