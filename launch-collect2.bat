@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes collect2.sh root@192.168.50.235:/root/collect2.sh
%SSHH% "nohup bash /root/collect2.sh > /root/collect2.out 2>&1 < /dev/null & echo LAUNCHED"
%SSHH% "sleep 3; tail -1 /root/collect2.out"
