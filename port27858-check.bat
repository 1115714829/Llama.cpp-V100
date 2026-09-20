@echo off
set OUT=F:\vllm+llama.cpp\1cat-vllm-v100-study\port27858-status.txt
cd /d F:\vllm+llama.cpp\llama.cpp
echo === git status --porcelain === > "%OUT%"
git status --porcelain >> "%OUT%" 2>&1
echo. >> "%OUT%"
echo === git diff --stat === >> "%OUT%"
git diff --stat >> "%OUT%" 2>&1
echo. >> "%OUT%"
echo === grep port symbols (should be >0) === >> "%OUT%"
findstr /N /C:"use_cpu_selector" /C:"load_dflash2_selector" /C:"build_dflash2_selector_cpu" /C:"is_dflash2_cpu" /C:"selector_rank" /C:"common_speculative_is_block_draft" /C:"common_speculative_block_draft_n_ubatch" /C:"llama_model_get_split_mode" src\models\dflash.cpp common\speculative.cpp common\speculative.h common\common.cpp src\llama-ext.h src\llama-model.cpp >> "%OUT%" 2>&1
echo. >> "%OUT%"
echo === meta.cpp diagnostic must be absent === >> "%OUT%"
findstr /C:"per-row op does not support" ggml\src\ggml-backend-meta.cpp >> "%OUT%" 2>&1
echo EXITCODE=%ERRORLEVEL% >> "%OUT%"
echo DONE >> "%OUT%"
