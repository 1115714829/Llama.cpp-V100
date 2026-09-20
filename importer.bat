@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\importer.txt
set SSHH=ssh -o BatchMode=yes root@192.168.50.235
echo === QdstrmImporter? === > "%OUT%"
%SSHH% "find /usr/local/cuda-12.4/nsight-systems-2023.4.4 -maxdepth 3 -name '*Qdstrm*' -o -maxdepth 3 -name '*qdstrm*'" >> "%OUT%" 2>&1
echo === nsys help export === >> "%OUT%"
%SSHH% "/usr/local/cuda-12.4/bin/nsys export --help" >> "%OUT%" 2>&1
echo DONE >> "%OUT%"
