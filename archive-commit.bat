@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\archive-commit.txt
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
echo === init === > "%OUT%"
git init >> "%OUT%" 2>&1
git config user.name >> "%OUT%" 2>&1
git config user.email >> "%OUT%" 2>&1
echo === add === >> "%OUT%"
git add -A >> "%OUT%" 2>&1
echo === staged count + size sanity === >> "%OUT%"
git diff --cached --shortstat >> "%OUT%" 2>&1
git diff --cached --name-only | find /c /v "" >> "%OUT%" 2>&1
echo === confirm the big blob is excluded === >> "%OUT%"
git status --porcelain --ignored | findstr /C:"tmp-src.tgz" >> "%OUT%" 2>&1
echo === commit === >> "%OUT%"
git commit -F F:\vllm+llama.cpp\ARCHIVE-COMMIT-MSG.txt >> "%OUT%" 2>&1
echo === log === >> "%OUT%"
git log --oneline -2 >> "%OUT%" 2>&1
git show --stat --oneline HEAD | tail -6 >> "%OUT%" 2>&1
echo === status after === >> "%OUT%"
git status --porcelain >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
