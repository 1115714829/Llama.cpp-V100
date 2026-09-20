@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\stage-report.txt
cd /d F:\vllm+llama.cpp\llama.cpp
echo === HEAD (base) === > "%OUT%"
git log --oneline -1 >> "%OUT%" 2>&1
git describe --tags >> "%OUT%" 2>&1
echo === stage all === >> "%OUT%"
git add -A >> "%OUT%" 2>&1
echo === git status --porcelain (staged) === >> "%OUT%"
git status --porcelain >> "%OUT%" 2>&1
echo === git diff --cached --stat === >> "%OUT%"
git diff --cached --stat >> "%OUT%" 2>&1
echo === untracked left === >> "%OUT%"
git ls-files --others --exclude-standard >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
