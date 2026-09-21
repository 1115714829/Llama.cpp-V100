#!/bin/bash
# Interleaved same-source A/B: pristine vs C4, two KV/FA configs, alternating order.
set -uo pipefail
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
export CUDA_VISIBLE_DEVICES=0

run_one() {
  local v="$1" tag="$2" flags="$3" extra="$4"
  local B=/root/bin-b11053-$v-bench
  local out
  out=$("$B" -m "$M" -p 512 -n 128 -ngl -1 $flags -r 3 2>&1)
  local pp tg
  pp=$(echo "$out" | grep -E "\| *pp512" | tail -1)
  tg=$(echo "$out" | grep -E "\| *tg128" | tail -1)
  echo "RESULT variant=$v cfg=$tag pp=\"$(echo "$pp" | awk -F'|' '{print $NF}' | tr -d ' ')\" tg=\"$(echo "$tg" | awk -F'|' '{print $NF}' | tr -d ' ')\""
}

echo "=== config PLAIN (llama-bench defaults, matches historical baseline.md command) ==="
for round in 1 2; do
  run_one pristine plain "" ""
  run_one c4       plain "" ""
done

echo ""
echo "=== config FAQ8 (-fa on -ctk q8_0 -ctv q8_0, matches today's 256k server config) ==="
for round in 1 2; do
  run_one pristine faq8 "-fa on -ctk q8_0 -ctv q8_0" ""
  run_one c4       faq8 "-fa on -ctk q8_0 -ctv q8_0" ""
done

echo ""
echo "=== GPU state ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo P4_DONE
