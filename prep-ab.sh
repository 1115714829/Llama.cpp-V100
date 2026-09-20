#!/bin/bash
# Check llama-server flags for spec-type and free GPU 2/5 by stopping our own C4 test server.
set -uo pipefail

echo "=== [1] --spec-type values (b11053 original binary) ==="
/root/llm/llama.cpp/bin/llama-server --help 2>&1 | grep -B2 -A 14 -- "--spec-type"
echo ""
echo "=== [2] spec / draft-mtp / mtp related flags ==="
/root/llm/llama.cpp/bin/llama-server --help 2>&1 | grep -iE "spec-|draft|mtp|nextn" | head -30
echo ""
echo "=== [3] stop OUR C4 test server (frees GPU 2/5) ==="
systemctl stop llama-server-c4
sleep 3
echo -n "llama-server-c4 now: "
systemctl is-active llama-server-c4
echo ""
echo "=== [4] GPU state after stop ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo ""
echo "=== [5] C4 build also has same flags? (sanity) ==="
/root/llm/test/v100-opt/llama.cpp/build/bin/llama-server --help 2>&1 | grep -cE "spec-type"
echo ""
echo "=== [6] version strings ==="
/root/llm/llama.cpp/bin/llama-cli --version 2>&1 | grep -iE "version|build" | head -3
echo DONE
