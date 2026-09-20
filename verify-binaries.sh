#!/bin/bash
# Verify the provenance of the binaries used in the A/B tests (read-only).
set -uo pipefail

echo "=== [1] installed prefix binary versions ==="
for b in llama-server llama-bench llama-cli; do
  echo -n "$b: "
  /root/llm/llama.cpp/bin/$b --version 2>&1 | grep -iE "^version|commit|build" | head -2 | tr '\n' ' '
  echo ""
done
echo ""
echo "=== [2] C4 build binary versions ==="
for b in llama-server llama-bench; do
  echo -n "$b: "
  /root/llm/test/v100-opt/llama.cpp/build/bin/$b --version 2>&1 | grep -iE "^version|commit|build" | head -2 | tr '\n' ' '
  echo ""
done
echo ""
echo "=== [3] is /root/llm/llama.cpp a git repo? ==="
git -C /root/llm/llama.cpp rev-parse HEAD 2>&1 | head -2
git -C /root/llm/llama.cpp log -1 --format="%h %ci %s" 2>&1 | head -2
echo ""
echo "=== [4] /root/llm/test/v100-opt/llama.cpp git ==="
git -C /root/llm/test/v100-opt/llama.cpp rev-parse HEAD 2>&1 | head -2
git -C /root/llm/test/v100-opt/llama.cpp log -1 --format="%h %ci %s" 2>&1 | head -2
git -C /root/llm/test/v100-opt/llama.cpp status --short 2>&1 | head -5
echo ""
echo "=== [5] build dirs under v100-opt ==="
ls -la /root/llm/test/v100-opt/llama.cpp/ 2>/dev/null | head -25
echo ""
echo "=== [6] all llama-server binaries on the box ==="
find /root -maxdepth 6 -type f -name "llama-server" 2>/dev/null
echo ""
echo "=== [7] top-level of /root/llm and /root/llm/test ==="
ls -la /root/llm/ 2>/dev/null
echo "--- test ---"
ls -la /root/llm/test/ 2>/dev/null
echo ""
echo "=== [8] disk free ==="
df -h /root / 2>/dev/null
echo ""
echo "=== [9] mmvq.cu current state in C4 tree (VOLTA count) ==="
grep -c VOLTA /root/llm/test/v100-opt/llama.cpp/ggml/src/ggml-cuda/mmvq.cu 2>/dev/null
echo "=== [10] backups ==="
ls -la /root/mmvq-*.cu 2>/dev/null
echo DONE
