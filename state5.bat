@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\state5.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === clock === > "%OUT%"
%SSHH% "date" >> "%OUT%" 2>&1
echo === server uptime/etime === >> "%OUT%"
%SSHH% "ps -o pid,etime,stat,cmd -p 1183335" >> "%OUT%" 2>&1
echo === harness procs === >> "%OUT%"
%SSHH% "ps -eo pid,etime,cmd" >> "%OUT%" 2>&1
echo === server log tail 12 === >> "%OUT%"
%SSHH% "tail -12 /tmp/p60-final-bf-tp3-server.log" >> "%OUT%" 2>&1
echo === server log size === >> "%OUT%"
%SSHH% "wc -c /tmp/p60-final-bf-tp3-server.log" >> "%OUT%" 2>&1
echo === health probe === >> "%OUT%"
%SSHH% "curl -s -m 5 http://127.0.0.1:8160/health" >> "%OUT%" 2>&1
echo "" >> "%OUT%"
echo === gpu === >> "%OUT%"
%SSHH% "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
