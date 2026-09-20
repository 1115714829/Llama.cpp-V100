@echo off
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "sleep 240; tail -6 /tmp/build-nccl.log; echo ---; pgrep -af build-nccl-inner; echo ---; ls -l /root/libdir-nccl/llama-server 2>&1; echo ---; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"
