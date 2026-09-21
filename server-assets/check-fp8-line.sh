#!/bin/bash
set -uo pipefail
echo "=== vllm-1cat.service (FP8 model) ==="
cat /root/llm/systemd/vllm-1cat.service 2>/dev/null | grep -E "ExecStart|--model|--served|tensor|quant|dtype|enforce|gpu-mem|kv-cache|max-model|port|spec|mtp" | head -25
echo ""
echo "=== 1cat-runtime.env ==="
cat /root/llm/systemd/1cat-runtime.env 2>/dev/null | grep -vE "^#|^$" | head -25
echo ""
echo "=== FP8 config.json key fields ==="
python3 -c "
import json
d=json.load(open('/root/llm/models/Qwen3.8-27B-FP8/config.json'))
for k in ['model_type','architectures','num_hidden_layers','hidden_size','num_attention_heads','num_key_value_heads','max_position_embeddings','torch_dtype']:
    print(k,'=',d.get(k))
q=d.get('quantization_config',{})
print('quant_method =',q.get('quant_method'),' fmt =',q.get('fmt'),' weight_block_size =',q.get('weight_block_size'))
" 2>&1 | head -15
echo ""
echo "=== 1cat-vLLM log dir? (benchmarks / results) ==="
ls -la /root/llm/1cat* 2>/dev/null | head -10
ls -la /root/1cat* 2>/dev/null | head -10
find /root -maxdepth 3 -name "*.json" -path "*1cat*" 2>/dev/null | head -8
echo FP8LINE_DONE
