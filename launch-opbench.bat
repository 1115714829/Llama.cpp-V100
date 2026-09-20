@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
scp -o BatchMode=yes opbench.sh root@192.168.50.235:/root/
%SSHH% "nohup bash /root/opbench.sh > /tmp/opbench.log 2>&1 < /dev/null & echo LAUNCHED"
%SSHH% "sleep 30; cat /tmp/opbench.log; echo ---; pgrep -af 'test-backend-op[s]' | head -3"
