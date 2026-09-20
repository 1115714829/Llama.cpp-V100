@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\goal-git-status2.txt
cd /d F:\vllm+llama.cpp
echo === git status --porcelain === > "%OUT%"
git status --porcelain >> "%OUT%" 2>&1
echo. >> "%OUT%"
echo === git diff --stat === >> "%OUT%"
git diff --stat >> "%OUT%" 2>&1
echo. >> "%OUT%"
echo === entries NOT under llama.cpp/ === >> "%OUT%"
git status --porcelain 2>&1 | findstr /V /C:"llama.cpp/" >> "%OUT%"
echo DONE >> "%OUT%"
