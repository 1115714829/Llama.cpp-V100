#!/bin/bash
# profile-tgt.sh -- kernel-level breakdown of the target's M=8 verify forward.
# Whole-run kernel sum is dominated by decode here (prompt ~30 tok vs 512 generated).
set -u
L=/root/libdir-nccl
PKGLIB=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/lib
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
KEY=$(head -1 /etc/llama-server/api-keys)
NSYS=/usr/local/cuda-12.4/bin/nsys
PORT=8190
OUT=/tmp/prof-tgt
rm -rf "$OUT".*

echo "=== launch server under nsys ==="
env CUDA_VISIBLE_DEVICES=0,1,2 LD_LIBRARY_PATH="$L:$PKGLIB" GGML_CUDA_P2P=1 \
  "$NSYS" profile -o "$OUT" --force-overwrite true --trace cuda,nvtx \
  "$L/llama-server" --model "$M" --model-draft "$D" \
    --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-ngl 999 \
    --split-mode tensor --tensor-split 1,1,1 --ctx-size 8192 -ngl 999 \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
    --host 127.0.0.1 --port "$PORT" --api-key-file /etc/llama-server/api-keys \
    > /tmp/prof-tgt-server.log 2>&1 &
pid=$!

n=0
while [ "$n" -lt 600 ]; do
  curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
  kill -0 "$pid" 2>/dev/null || { echo "server died"; tail -5 /tmp/prof-tgt-server.log; exit 1; }
  sleep 15; n=$((n+15))
done
echo "health after ${n}s"

P1="Explain, step by step, how to compute the sum of the first n integers and prove the formula by induction."
body="{\"messages\":[{\"role\":\"user\",\"content\":\"$P1\"}],\"n_predict\":512,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"presence_penalty\":0.0,\"repetition_penalty\":1.0,\"seed\":42,\"cache_prompt\":false,\"chat_template_kwargs\":{\"enable_thinking\":true,\"preserve_thinking\":true,\"reasoning_effort\":\"xhigh\"}}"
echo "=== one 512-token request ==="
curl -s -m 900 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "$body" "http://127.0.0.1:$PORT/v1/chat/completions" > /tmp/prof-tgt-resp.json
grep -a -oE "[0-9]+\.[0-9]+ tokens per second" /tmp/prof-tgt-server.log | tail -1

echo "=== graceful stop (nsys needs it) ==="
kill -TERM "$pid" 2>/dev/null
for i in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 3; done
kill -0 "$pid" 2>/dev/null && { echo "still up; SIGINT"; kill -INT "$pid"; sleep 20; }
wait "$pid" 2>/dev/null
ls -l "$OUT".nsys-rep 2>&1

echo "=== kernel summary ==="
"$NSYS" stats --report cuda_gpu_kern_sum --force-export true "$OUT".nsys-rep 2>&1 | head -40
echo PROF_TGT_DONE
