#!/bin/bash
# Phase 0.1: verify the Q8_0 download (size + sha256) and record disk state.
set -uo pipefail
D=/root/llm/models/Qwen3.8-27B-GGUF
M=$D/Qwen3.8-27B-Q8_0.gguf
L=/tmp/sha-q8.log
: > "$L"

echo "expected size   : 29047086048"        | tee -a "$L"
echo "expected sha256 : a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348" | tee -a "$L"
echo "--- actual ---" | tee -a "$L"
stat -c "size=%s bytes" "$M" 2>/dev/null | tee -a "$L"
df -h / | tail -1 | tee -a "$L"

if [ "$(stat -c %s "$M" 2>/dev/null)" != "29047086048" ]; then
  echo "SIZE MISMATCH - aborting sha256" | tee -a "$L"
  echo SHA_DONE | tee -a "$L"
  exit 1
fi

echo "size OK; starting sha256 (reads 29 GB, backgrounded)" | tee -a "$L"
nohup bash -c "sha256sum $M >> $L 2>&1; echo SHA_DONE >> $L" > /dev/null 2>&1 &
echo "sha256 launched"
