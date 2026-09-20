@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes final-nccl.sh final-nccl-launch.sh root@192.168.50.235:/root/
%SSHH% "bash /root/final-nccl-launch.sh"
%SSHH% "sleep 5; pgrep -af 'final-nccl|nccl-env-sweep'"
%SSHH% "tail -6 /tmp/nccl-env-sweep.log"
