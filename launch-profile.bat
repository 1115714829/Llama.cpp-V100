@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes profile-tgt.sh root@192.168.50.235:/root/profile-tgt.sh
%SSHH% "nohup bash /root/profile-tgt.sh > /tmp/profile-tgt.log 2>&1 < /dev/null & echo LAUNCHED"
%SSHH% "sleep 5; pgrep -af 'profile-tgt|nsys'"
