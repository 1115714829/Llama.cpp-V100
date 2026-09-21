#!/bin/bash
# Interleaved same-source A/B, raw rows captured to logs (robust parsing).
set -uo pipefail
M=/root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf
export CUDA_VISIBLE_DEVICES=0

run_one() {
  v="$1"; tag="$2"; shift 2
  B=/root/bin-b11053-$v-bench
  L=/tmp/ab-${v}-${tag}.log
  echo "### variant=$v cfg=$tag"
  "$B" -m "$M" -p 512 -n 128 -ngl -1 "$@" -r 3 > "$L" 2>&1
  echo "exit=$?"
  grep -F -e "pp512" -e "tg128" "$L"
  echo ""
}

echo "=========== config PLAIN (defaults, same as historical baseline.md) ==========="
run_one pristine plain
run_one c4       plain
run_one pristine plain
run_one c4       plain

echo "=========== config FAQ8 (-fa on -ctk q8_0 -ctv q8_0) ==========="
run_one pristine faq8 -fa on -ctk q8_0 -ctv q8_0
run_one c4       faq8 -fa on -ctk q8_0 -ctv q8_0
run_one pristine faq8 -fa on -ctk q8_0 -ctv q8_0
run_one c4       faq8 -fa on -ctk q8_0 -ctv q8_0

echo "=========== summary lines ==========="
for f in /tmp/ab-*.log; do
  echo "--- $f"
  grep -F -e "pp512" -e "tg128" "$f" | awk -F'|' '{gsub(/ /,"",$0); print $0}'
done
echo P5_DONE
