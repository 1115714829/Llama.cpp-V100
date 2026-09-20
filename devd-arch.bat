@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\devd-archaeology.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === p60 logs mentioning draft device / abort === > "%OUT%"
%SSHH% "grep -al 'spec-draft-device' /tmp/p60-*.log /tmp/*.log" >> "%OUT%" 2>&1
echo === abort context === >> "%OUT%"
%SSHH% "grep -aE 'ggml_abort|Aborted|GGML_ASSERT|spec-draft-device|device-draft|SPLIT_MODE|n_devices|split_mode' /tmp/p60-*-server.log" >> "%OUT%" 2>&1
echo === list p60 logs === >> "%OUT%"
%SSHH% "ls -lt /tmp/p60-*.log" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
