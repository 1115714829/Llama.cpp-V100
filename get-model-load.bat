@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\model-load.txt
ssh -o BatchMode=yes root@192.168.50.235 "grep -a -m1 -n 'load_model' /tmp/p60-ab-nccl-tp3-server.log; sed -n '1,55p' /tmp/p60-ab-nccl-tp3-server.log | cut -c1-150" > "%OUT%" 2>&1
echo DONE >> "%OUT%"
