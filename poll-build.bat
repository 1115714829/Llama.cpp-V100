@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes p60-ab-harness2.sh root@192.168.50.235:/root/p60-ab-harness2.sh
echo === build log so far === > F:\vllm+llama.cpp\1cat-vllm-v100-study\build-nccl-progress.txt
%SSHH% "cat /tmp/build-nccl.log" >> F:\vllm+llama.cpp\1cat-vllm-v100-study\build-nccl-progress.txt 2>&1
echo === still running? === >> F:\vllm+llama.cpp\1cat-vllm-v100-study\build-nccl-progress.txt
%SSHH% "pgrep -af build-nccl-inner" >> F:\vllm+llama.cpp\1cat-vllm-v100-study\build-nccl-progress.txt 2>&1
echo DONE >> F:\vllm+llama.cpp\1cat-vllm-v100-study\build-nccl-progress.txt
