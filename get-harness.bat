@echo off
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "cat /root/p60-ab-harness.sh" > F:\vllm+llama.cpp\1cat-vllm-v100-study\harness.txt 2>&1
echo DONE >> F:\vllm+llama.cpp\1cat-vllm-v100-study\harness.txt
