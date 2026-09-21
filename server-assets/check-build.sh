#!/bin/bash
set -uo pipefail
echo "=== build log (last 8) ==="
tail -8 /tmp/build-c4.log 2>/dev/null
echo "=== build process (still running?) ==="
if pgrep -f "cmake --build.*llama-server" >/dev/null 2>&1; then echo "BUILD_RUNNING"; else echo "BUILD_DONE"; fi
echo "=== llama-server present? ==="
ls -la /root/llm/test/v100-opt/llama.cpp/build/bin/llama-server 2>/dev/null
echo "=== errors? ==="
grep -iE "error|Error|FAILED" /tmp/build-c4.log 2>/dev/null | tail -4
echo BUILD_CHECK_DONE
