@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\dirlist.txt
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
echo === files by size (largest first) === > "%OUT%"
dir /a-d /o-s >> "%OUT%" 2>&1
echo === subdirs === >> "%OUT%"
dir /ad >> "%OUT%" 2>&1
echo === total === >> "%OUT%"
dir /s /-c | findstr /C:"File(s)" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
