@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\archive-commit2.txt
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
echo === status before === > "%OUT%"
git status --porcelain >> "%OUT%" 2>&1
git add -A >> "%OUT%" 2>&1
git commit -m "archive : add the directory listing and commit logs" -m "Assisted-by: Qwen Code" >> "%OUT%" 2>&1
echo === log === >> "%OUT%"
git log --oneline -3 >> "%OUT%" 2>&1
echo === status after === >> "%OUT%"
git status --porcelain >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
