@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\state-before-handoff.txt
echo === llama.cpp repo === > "%OUT%"
cd /d F:\vllm+llama.cpp\llama.cpp
git log --oneline -2 >> "%OUT%" 2>&1
git status --porcelain >> "%OUT%" 2>&1
echo (empty above = clean) >> "%OUT%"
echo === study archive repo === >> "%OUT%"
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
git log --oneline -2 >> "%OUT%" 2>&1
git status --porcelain >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
