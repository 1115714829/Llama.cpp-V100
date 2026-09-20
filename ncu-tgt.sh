#!/bin/bash
# ncu-tgt.sh -- decide where the target's M=8 verify forward actually spends its time.
# Sample steady-state decode kernels only (skip the load/first tokens).
set -u
L=/root/libdir-nccl
PKGLIB=/root/llm/ac922env/1cat-20260907/venv/lib/python3.11/site-packages/ac922_nccl_runtime/lib
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
D=/root/llm/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
KEY=$(head -1 /etc/llama-server/api-keys)
NCU=/usr/local/cuda-12.4/bin/ncu
PORT=8192
OUT=/tmp/ncu-tgt

echo "=== ncu version ==="
"$NCU" --version 2>&1 | head -4
rm -rf "$OUT".*

echo "=== launch server under ncu (sample 60 kernels well into decode) ==="
env CUDA_VISIBLE_DEVICES=0,1,2 LD_LIBRARY_PATH="$L:$PKGLIB" GGML_CUDA_P2P=1 \
  "$NCU" --target-processes all --kernel-name-base demangled \
    --launch-skip 4000 --launch-count 60 \
    --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,smsp__inst_executed.sum \
    --force-overwrite -o "$OUT" \
  "$L/llama-server" --model "$M" --model-draft "$D" \
    --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-ngl 999 \
    --split-mode tensor --tensor-split 1,1,1 --ctx-size 8192 -ngl 999 \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
    --host 127.0.0.1 --port "$PORT" --api-key-file /etc/llama-server/api-keys \
    > /tmp/ncu-tgt-server.log 2>&1 &
PID=$!

n=0
while [ "$n" -lt 1200 ]; do
  curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
  kill -0 "$PID" 2>/dev/null || { echo "server died after ${n}s"; tail -20 /tmp/ncu-tgt-server.log; exit 1; }
  sleep 20; n=$((n+20))
done
echo "health after ${n}s"

P1="Explain, step by step, how to compute the sum of the first n integers and prove the formula by induction."
body="{\"messages\":[{\"role\":\"user\",\"content\":\"$P1\"}],\"n_predict\":256,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"presence_penalty\":0.0,\"repetition_penalty\":1.0,\"seed\":42,\"cache_prompt\":false,\"chat_template_kwargs\":{\"enable_thinking\":true,\"preserve_thinking\":true,\"reasoning_effort\":\"xhigh\"}}"
echo "=== request (256 tokens) ==="
curl -s -m 1800 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "$body" "http://127.0.0.1:$PORT/v1/chat/completions" > /tmp/ncu-tgt-resp.json
grep -a -oE "[0-9]+\.[0-9]+ tokens per second" /tmp/ncu-tgt-server.log | tail -1

echo "=== stop app so ncu finalizes ==="
APP=$(pgrep -f "libdir-nccl/llama-server" | head -1)
kill -TERM "$APP" 2>/dev/null
wait "$PID" 2>/dev/null
sleep 5
ls -l /tmp/ncu-tgt.ncu-rep 2>&1

echo "=== kernel time summary ==="
"$NCU" --import /tmp/ncu-tgt.ncu-rep --page details 2>&1 | head -5
"$NCU" --import /tmp/ncu-tgt.ncu-rep --csv --page raw 2>&1 | head -5
echo NCU_TGT_DONE
