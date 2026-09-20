source /opt/rh/gcc-toolset-12/enable
cd /root/llm/test/v100-opt/llama.cpp || exit 1
echo "INCR_BUILD_START"
cmake --build build --config Release -j16 --target llama-bench
echo "INCR_BUILD_DONE"
echo "BENCH_START"
CUDA_VISIBLE_DEVICES=2 ./build/bin/llama-bench -m /root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf -p 512 -n 128
echo "BENCH_DONE"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "ALL_DONE"
