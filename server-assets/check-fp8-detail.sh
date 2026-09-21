#!/bin/bash
set -uo pipefail
echo "=== vllm-1cat.service (FP8 model, 1cat-vLLM) ==="
cat /root/llm/systemd/vllm-1cat.service 2>/dev/null
echo ""
echo "=== 1cat-runtime.env ==="
cat /root/llm/systemd/1cat-runtime.env 2>/dev/null | head -30
echo ""
echo "=== FP8 config.json (model_type / quant / layers) ==="
python3 -c "
import json
d=json.load(open('/root/llm/models/Qwen3.8-27B-FP8/config.json'))
for k in ['model_type','architectures','num_hidden_layers','hidden_size','num_attention_heads','num_key_value_heads','torch_dtype','quantization_config']:
    v=d.get(k)
    if k=='quantization_config' and v: v={kk:v[kk] for kk in list(v)[:6]}
    print(k,'=',v)
" 2>&1 | head -20
echo ""
echo "=== is 1cat-vLLM currently running? (GPU 0/1/3/4) ==="
systemctl is-active vllm-1cat 2>/dev/null
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 0
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 1
echo FP8DETAIL_DONE
