@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes nccl-ab.sh nccl-ab-launch.sh root@192.168.50.235:/root/
%SSHH% "bash /root/nccl-ab-launch.sh"
%SSHH% "sleep 5; pgrep -af nccl-ab"
