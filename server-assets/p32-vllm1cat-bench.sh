#!/bin/bash
# Premise check: measure the REAL vllm-1cat service on this box.
# unit = 1Cat-vLLM Qwen3.8-27B-FP8 DFlash2 256K, TP4 on GPU 0/1/3/4, port 8000, max-num-seqs 1.
set -uo pipefail
LOG=/tmp/vllm1cat-bench.log
: > "$LOG"
exec >> "$LOG" 2>&1
U=/root/llm/systemd/vllm-1cat.service
KEY=$(grep -o 'sk-1cat-[A-Za-z0-9]*' "$U" | head -1)
PORT=8000

echo "=== before: GPU state ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "api key found: $([ -n "$KEY" ] && echo yes || echo NO)"

echo "=== starting vllm-1cat ==="
systemctl start vllm-1cat
echo "start rc=$?"

echo "=== wait for API on port $PORT (load may take many minutes on ppc64le) ==="
n=0; ok=0
while [ "$n" -lt 1800 ]; do
  if curl -s -m 5 -H "Authorization: Bearer $KEY" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q 'Qwen3.8-27B-FP8'; then ok=1; break; fi
  sleep 15; n=$((n+15))
  if [ $((n % 120)) -eq 0 ]; then echo "  ...still loading (${n}s)"; journalctl -u vllm-1cat -n 2 --no-pager | tail -2; fi
done
echo "api ok=$ok after ${n}s"
if [ "$ok" != "1" ]; then
  echo "active=$(systemctl is-active vllm-1cat)"
  journalctl -u vllm-1cat -n 40 --no-pager | tail -25
  echo VLLMBENCH_DONE
  exit 0
fi

echo "=== after load: GPU state ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

P1="A train leaves Station A at 60 km/h. Another leaves Station B, 300 km away, at 40 km/h toward A. They start at the same time. Work through this step by step and explain when and where they meet."
P2="Explain, step by step, how to compute the sum of the first n integers and prove the formula by induction."
P3="Solve step by step: if 3x + 7 = 22, what is x? Then explain in detail how you checked your answer."

i=0
for p in "$P1" "$P2" "$P3"; do
  i=$((i+1))
  echo ""
  echo "########## vllm-1cat request $i (max_tokens 512, temp 1.0 / top_p 0.95 / top_k 20, seed 42) ##########"
  t0=$(date +%s%N)
  curl -s -m 900 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d "{\"model\":\"Qwen3.8-27B-FP8\",\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":512,\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"seed\":42}" \
    "http://127.0.0.1:$PORT/v1/chat/completions" > /tmp/vllm-r$i.json 2>&1
  t1=$(date +%s%N)
  echo "wall_ms=$(( (t1 - t0) / 1000000 ))"
  echo "usage: $(grep -o '"usage":{[^}]*}' /tmp/vllm-r$i.json | head -1)"
  echo "finish/cap: $(grep -o '"finish_reason":"[a-z]*"' /tmp/vllm-r$i.json | head -1)"
done

echo ""
echo "=== vLLM own throughput lines (generation throughput excludes prefill) ==="
journalctl -u vllm-1cat --no-pager -n 600 | grep -a "Avg generation throughput" | tail -15
echo "=== acceptance / speculative lines, if logged ==="
journalctl -u vllm-1cat --no-pager -n 3000 | grep -aiE "acceptance rate|accepted tokens|spec_decode|num_accepted" | tail -12
echo "=== dtype/engine summary ==="
journalctl -u vllm-1cat --no-pager | grep -aiE "dtype|kv_cache_dtype|speculative|dflash|model loading took|graph capturing" | tail -18
echo VLLMBENCH_DONE
