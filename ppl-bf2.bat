@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes ppl-bf2.sh root@192.168.50.235:/root/ppl-bf2.sh
%SSHH% "bash /root/ppl-bf2.sh" > F:\vllm+llama.cpp\1cat-vllm-v100-study\ppl-bf2.txt 2>&1
echo DONE >> F:\vllm+llama.cpp\1cat-vllm-v100-study\ppl-bf2.txt
