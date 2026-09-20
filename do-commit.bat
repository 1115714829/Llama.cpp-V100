@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\commit-report.txt
cd /d F:\vllm+llama.cpp\llama.cpp
echo === identity config === > "%OUT%"
git config user.name >> "%OUT%" 2>&1
git config user.email >> "%OUT%" 2>&1
echo === staged before commit === >> "%OUT%"
git status --porcelain >> "%OUT%" 2>&1
echo === commit === >> "%OUT%"
git commit -F F:\vllm+llama.cpp\1cat-vllm-v100-study\COMMIT-MSG.txt >> "%OUT%" 2>&1
echo === log === >> "%OUT%"
git log --oneline -3 >> "%OUT%" 2>&1
echo === show --stat HEAD === >> "%OUT%"
git show --stat --oneline HEAD >> "%OUT%" 2>&1
echo === worktree after commit === >> "%OUT%"
git status --porcelain >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
