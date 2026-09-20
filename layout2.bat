@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\layout2.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === A-probe lanes 8..31 === > "%OUT%"
%SSHH% "sed -n '29,136p' /tmp/hmma_layout.out" >> "%OUT%" 2>&1
echo === B-probe section === >> "%OUT%"
%SSHH% "sed -n '137,264p' /tmp/hmma_layout.out" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
