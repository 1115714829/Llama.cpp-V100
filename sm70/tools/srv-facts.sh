#!/bin/bash
# srv-facts.sh - read-only: launch parameters, draft models, test tools. Changes nothing.
echo "== vllm_batched_tokens"
grep -o -e '--max-num-batched-tokens [0-9]*' /root/llm/systemd/vllm-1cat.service
echo "== vllm_runtime_env_names"
sed -E 's/=.*/=.../' /root/llm/systemd/1cat-runtime.env 2>&1 | head -60
echo "== vllm_models"
ls -la /root/llm/models/ 2>&1 | head -30
ls -la /root/llm/models/Qwen3.8-27B-DFlash2 2>&1 | head -20
grep -A12 -e quantization_config /root/llm/models/Qwen3.8-27B-FP8/config.json 2>&1 | head -20
cat /root/llm/models/Qwen3.8-27B-FP8/generation_config.json 2>&1
echo "== draft_gguf"
ls -la /mnt/3.84t/llm-models/Qwen3.8-27B-DFlash2-GGUF/ 2>&1
echo "== test_tools"
for d in /root/libdir-gb /root/llm/test/v100-opt/llama.cpp/build-instr/bin /root/llm/test/v100-opt/llama.cpp/build-nccl/bin; do
  for b in llama-server test-backend-ops test-export-graph-ops llama-perplexity llama-bench; do
    if [ -e "$d/$b" ]; then echo "$d/$b $(stat -c %y "$d/$b" | cut -c1-19) $(md5sum "$d/$b" | cut -c1-12)"; fi
  done
done
echo "== ldd_server"
LD_LIBRARY_PATH=/root/libdir-gb ldd /root/libdir-gb/llama-server 2>&1 | grep -e ggml -e llama
echo "== prompt_file"
ls -la /root/llm/test/bl-prompt90.txt*; cat /root/llm/test/bl-prompt90.txt.meta 2>&1
echo "== py"; python3 --version
echo FACTS_DONE
