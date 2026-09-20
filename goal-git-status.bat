@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\goal-git-status.txt
cd /d F:\vllm+llama.cpp\llama.cpp
echo === git status --porcelain === > "%OUT%"
git status --porcelain >> "%OUT%" 2>&1
echo. >> "%OUT%"
echo === git diff --stat === >> "%OUT%"
git diff --stat >> "%OUT%" 2>&1
echo. >> "%OUT%"
echo === files outside llama.cpp/ (should be empty) === >> "%OUT%"
cd /d F:\vllm+llama.cpp
git status --porcelain 2>>"%OUT%" | findstr /V "^ M llama.cpp/" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
