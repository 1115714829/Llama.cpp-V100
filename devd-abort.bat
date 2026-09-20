@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\devd-abort.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === devd-cuda0 log (full, short) === > "%OUT%"
%SSHH% "cat /tmp/p60-devd-cuda0-server.log" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
