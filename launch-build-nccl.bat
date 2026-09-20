@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes build-nccl-inner.sh build-nccl.sh root@192.168.50.235:/root/
%SSHH% "bash /root/build-nccl.sh"
%SSHH% "sleep 8; pgrep -af build-nccl-inner; ls -l /tmp/build-nccl.log"
