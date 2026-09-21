source /opt/rh/gcc-toolset-12/enable
cd /root/llm/test/v100-opt/llama.cpp || exit 1
export CUDA_VISIBLE_DEVICES=2
echo "NCU_START"
timeout 600 /usr/local/cuda-12.4/bin/ncu --csv --metrics gpu__time_duration.sum \
  --launch-count 1500 \
  ./build/bin/llama-bench -m /root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf -p 1 -n 32 > /root/ncu-decode.csv 2>/root/ncu-decode.err
echo "NCU_RC=$?"
echo "=== csv rows ==="
wc -l /root/ncu-decode.csv
echo "=== csv head ==="
head -3 /root/ncu-decode.csv
echo "=== err tail ==="
tail -5 /root/ncu-decode.err
echo "PROFILE_DONE"
