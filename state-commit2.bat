@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\state-commit2.txt
cd /d F:\vllm+llama.cpp\llama.cpp
echo === HEAD === > "%OUT%"
git log --oneline -3 >> "%OUT%" 2>&1
git describe --tags >> "%OUT%" 2>&1
echo === status --porcelain (empty = nothing to commit) === >> "%OUT%"
git status --porcelain >> "%OUT%" 2>&1
echo === diff --stat (unstaged) === >> "%OUT%"
git diff --stat >> "%OUT%" 2>&1
echo === diff --stat --cached (staged) === >> "%OUT%"
git diff --cached --stat >> "%OUT%" 2>&1
echo === untracked === >> "%OUT%"
git ls-files --others --exclude-standard >> "%OUT%" 2>&1
echo === last commit stat === >> "%OUT%"
git show --stat --oneline HEAD >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
