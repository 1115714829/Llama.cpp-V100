#!/bin/bash
# Phase P2: if the pristine build finished, save the pristine binaries and restore C4 source.
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp

echo "=== build log tail ==="
tail -5 /tmp/build-pristine.log
echo ""
if pgrep -f "[c]make --build" >/dev/null 2>&1; then
  echo "BUILD_STILL_RUNNING"
  echo P2_DONE
  exit 0
fi
echo "=== build finished; error scan ==="
grep -iE "error:|Error [0-9]|FAILED" /tmp/build-pristine.log | head -10 || true
echo "(no lines above = no errors)"
echo ""
echo "=== pristine binary versions (expect 0.4.1-dev) ==="
"$SRC/build/bin/llama-server" --version 2>&1 | grep -iE "version|commit" | head -2
"$SRC/build/bin/llama-bench"  --version 2>&1 | grep -iE "version|commit" | head -2
echo ""
echo "=== save pristine binaries aside ==="
cp -a "$SRC/build/bin/llama-server" /root/bin-b11053-pristine-server
cp -a "$SRC/build/bin/llama-bench"  /root/bin-b11053-pristine-bench
ls -la /root/bin-b11053-pristine-server /root/bin-b11053-pristine-bench /root/bin-b11053-c4-server /root/bin-b11053-c4-bench
echo ""
echo "=== restore C4 source (so the tree matches the C4 patch) ==="
cp /root/mmvq-c4-backup.cu "$SRC/ggml/src/ggml-cuda/mmvq.cu"
echo "C4 refs in source now: $(grep -c MMVQ_PARAMETERS_VOLTA "$SRC/ggml/src/ggml-cuda/mmvq.cu")  (expect 5)"
echo ""
echo "=== quick sanity: both binaries run + report arch ==="
"$SRC/build/bin/llama-bench" -m /root/llm/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf -p 16 -n 8 -ngl -1 -fa on -ctk q8_0 -ctv q8_0 2>&1 | tail -6
echo P2_DONE
