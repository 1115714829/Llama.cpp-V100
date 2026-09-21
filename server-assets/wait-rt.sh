#!/bin/bash
# Wait for the instrument build and the Q8_0 sha256.
set -uo pipefail
WAIT=${WAIT_S:-600}
D=$(( $(date +%s) + WAIT ))
while [ "$(date +%s)" -lt "$D" ]; do
  pgrep -f "[c]make --build" >/dev/null 2>&1 || break
  sleep 15
done

echo "=== build state ==="
if pgrep -f "[c]make --build" >/dev/null 2>&1; then echo STATE=BUILDING; else echo STATE=BUILD_DONE; fi
echo "--- errors ---"
grep -inE "error:|Error [0-9]|FAILED|undefined reference" /tmp/build-rt.log 2>/dev/null | head -20 || echo "(none)"
echo "--- tail ---"
tail -8 /tmp/build-rt.log 2>/dev/null

echo ""
echo "=== sha256 state ==="
cat /tmp/sha-q8.log 2>/dev/null

echo ""
echo "=== artifacts ==="
ls -l --time-style=+%H:%M:%S /root/llm/test/v100-opt/llama.cpp/build/bin/llama-server \
                       /root/llm/test/v100-opt/llama.cpp/build/bin/llama-bench \
                       /root/llm/test/v100-opt/llama.cpp/build/bin/libllama.so.0.4.1 2>/dev/null
echo WAITRT_DONE
