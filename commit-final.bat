@echo off
cd /d F:\vllm+llama.cpp\1cat-vllm-v100-study
git add -A
git commit -m "archive : final state snapshot at hand-off" -m "Records the hand-off-time state of both repos and the AC922 (llama.cpp HEAD 79504e72b clean; archive HEAD 5402fc1; vllm-1cat and llmscope inactive, all six V100s free)." -m "Assisted-by: Qwen Code"
git log --oneline -1
git status --porcelain
echo DONE
