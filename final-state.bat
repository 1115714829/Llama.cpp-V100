@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\final-state.txt
echo === llama.cpp === > "%OUT%"
cd /d F:\vllm+llama.cpp\llama.cpp
git log --oneline -2 >> "%OUT%" 2>&1
git status --porcelain >> "%OUT%" 2>&1
echo === archive === >> "%OUT%"
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
git log --oneline -3 >> "%OUT%" 2>&1
git status --porcelain >> "%OUT%" 2>&1
echo === server === >> "%OUT%"
ssh -o BatchMode=yes root@192.168.50.235 "systemctl is-active vllm-1cat llmscope new-api; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
