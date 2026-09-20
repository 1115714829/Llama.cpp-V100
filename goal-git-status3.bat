@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\goal-git-status3.txt
cd /d F:\vllm+llama.cpp\llama.cpp
echo === git status --porcelain (repo root = llama.cpp) === > "%OUT%"
git status --porcelain >> "%OUT%" 2>&1
echo. >> "%OUT%"
echo === git diff --stat === >> "%OUT%"
git diff --stat >> "%OUT%" 2>&1
echo. >> "%OUT%"
echo === untracked / new files === >> "%OUT%"
git ls-files --others --exclude-standard >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
