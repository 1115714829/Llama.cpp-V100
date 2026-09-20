@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\ncu-diag.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
%SSHH% "echo ===UPTIME===; uptime; echo ===PROCS===; pgrep -af 'ncu|llama-server' | head -6; echo ===TMPLS===; ls -la /tmp/ | grep -aE 'ncu|p60' | head -20; echo ===SCRIPT===; cat /root/ncu-tgt.sh 2>/dev/null || echo NO_SCRIPT; echo ===SERVERLOG===; cat /tmp/ncu-tgt-server.log" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
