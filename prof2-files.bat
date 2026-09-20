@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\prof2-files.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === /tmp/prof2* === > "%OUT%"
%SSHH% "ls -l /tmp/prof2*" >> "%OUT%" 2>&1
echo === nsys messages in server log === >> "%OUT%"
%SSHH% "grep -aiE 'nsys|Generating|Generated|report|\.qdstrm|\.nsys-rep|error' /tmp/prof2-server.log | tail -20" >> "%OUT%" 2>&1
echo === server log last 5 === >> "%OUT%"
%SSHH% "tail -5 /tmp/prof2-server.log" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
