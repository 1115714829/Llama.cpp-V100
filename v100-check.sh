echo "=== NSIGHT CHECK ==="
which ncu
which nsys
ls -la /usr/local/cuda*/bin/ncu 2>/dev/null
ls -la /usr/local/cuda*/bin/ncu-cli 2>/dev/null
ls -d /opt/nvidia/nsight-compute* 2>/dev/null
ls -d /opt/nvidia/nsight-systems* 2>/dev/null
ls /usr/local/cuda-12.4/nsight-compute-*/ 2>/dev/null
echo "=== BUILD STATE ==="
cd /root/llm/test/v100-opt/llama.cpp
ls -la build/bin/llama-bench 2>/dev/null
md5sum ggml/src/ggml-cuda/fattn.cu
grep -c "v100-opt exp1" ggml/src/ggml-cuda/fattn.cu
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "CHECK_DONE"
