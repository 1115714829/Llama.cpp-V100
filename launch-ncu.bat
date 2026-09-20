@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes ncu-tgt.sh root@192.168.50.235:/root/
%SSHH% "nohup bash /root/ncu-tgt.sh > /tmp/ncu-tgt.log 2>&1 < /dev/null & echo LAUNCHED"
%SSHH% "sleep 20; tail -4 /tmp/ncu-tgt.log; pgrep -af 'ncu' | head -3"
