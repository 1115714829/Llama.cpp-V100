#!/bin/bash
# Poll the PR #27858 port build.
set -uo pipefail
WAIT=${WAIT_S:-600}
B=/tmp/build-p27858.log
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  pgrep -f "[c]make --build" >/dev/null 2>&1 || break
  sleep 15
done

echo "=== TEST27858 script output ==="
cat /tmp/test27858.log 2>/dev/null

echo ""
echo "=== build state ==="
if pgrep -f "[c]make --build" >/dev/null 2>&1; then echo STATE=BUILDING; else echo STATE=BUILD_STOPPED; fi
echo "--- errors (grep) ---"
grep -inE "error:|Error [0-9]|FAILED|undefined reference" "$B" 2>/dev/null | head -20 || echo "(none)"
echo "--- build log tail ---"
tail -8 "$B" 2>/dev/null

echo ""
echo "=== artifacts ==="
ls -la /root/llm/test/v100-opt/llama.cpp/build/bin/llama-server \
       /root/llm/test/v100-opt/llama.cpp/build/bin/llama-bench 2>/dev/null

echo ""
echo "=== lib timestamps (link happened?) ==="
ls -la --time-style=+%H:%M:%S /root/llm/test/v100-opt/llama.cpp/build/bin/libggml-base.so* \
                    /root/llm/test/v100-opt/llama.cpp/build/bin/libllama.so* 2>/dev/null
echo WAIT27858_DONE
