#!/bin/bash
# T4 gate re-establish: decomp=1 greedy sha x2 (must self-reproduce) vs the
# old gate f3edac19 (decomp=0 arm, unchanged path). Records the new gate value.
set -u
M=/mnt/3.84t/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
D=/mnt/3.84t/llm-models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
KEY=$(head -1 /etc/llama-server/api-keys)
PROMPT=$(python3 -c "print('The quick brown fox jumps over the lazy dog. '*10)")

arm() {
  tag=$1
  dec=$2
  echo "=== GATE ARM $tag decomp=$dec $(date) ==="
  env GGML_KV_BUCKET_RATIO=1.08 LLAMA_SM70_FA_DECOMP=$dec \
    LD_LIBRARY_PATH=/root/libdir-gb CUDA_VISIBLE_DEVICES=0,1,2 GGML_GALLOCR_SLOTS=3 \
    nohup /root/libdir-gb/llama-server --model "$M" --model-draft "$D" \
      --spec-type draft-dflash --spec-draft-n-max 7 \
      --split-mode tensor --tensor-split 1,1,1 --ctx-size 8192 -ngl 999 \
      --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
      --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.0 \
      --host 127.0.0.1 --port 8331 --api-key "$KEY" \
      > /tmp/t4g-$tag-server.log 2>&1 &
  SPID=$!
  for i in $(seq 1 40); do
    code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" http://127.0.0.1:8331/health)
    if [ "$code" = "200" ]; then break; fi
    sleep 5
  done
  curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"$PROMPT\",\"n_predict\":128,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"repeat_penalty\":1.0,\"seed\":4321}" \
    http://127.0.0.1:8331/completion > /tmp/t4g-$tag-resp.json 2>&1
  sleep 3
  kill $SPID 2>/dev/null
  sleep 4
  python3 - <<PYEOF
import json, hashlib
c = json.load(open('/tmp/t4g-$tag-resp.json'))['content']
print('sha=%s len=%d' % (hashlib.sha256(c.encode()).hexdigest(), len(c)))
PYEOF
}
arm d0 0
arm d1 1
arm d1b 1
arm d0b 0
echo T4G_DONE
