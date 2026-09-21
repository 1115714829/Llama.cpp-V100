#!/bin/bash
# Which shared object really contains the llama core code that the server executes?
set -uo pipefail
cd /root/libdir-rt

for f in libllama.so.0.4.1 libllama-server-impl.so libllama-common.so.0.4.1 libmtmd.so.0.4.1 ; do
  echo "--- $f"
  echo -n "   llama-context marker 'failed to allocate graph' : "; grep -c -a "failed to allocate graph" "$f" 2>/dev/null
  echo -n "   llama-context marker 'removing memory module'   : "; grep -c -a "removing memory module entries" "$f" 2>/dev/null
  echo -n "   new round-timing marker                          : "; grep -c -a "round timing (LLAMA_ROUND_TIMING)" "$f" 2>/dev/null
  echo -n "   exported llama_decode                            : "; nm -D --defined-only "$f" 2>/dev/null | grep -c "llama_decode"
done

echo ""
echo "=== which lib DEFINES llama_decode (T = text/defined) ==="
for f in libllama.so.0.4.1 libllama-server-impl.so ; do
  echo "--- $f"
  nm -D --defined-only "$f" 2>/dev/null | grep -e "llama_decode" -e "llama_new_context" | head -8
done

echo ""
echo "=== does server-impl reference libllama? ==="
nm -D -u libllama-server-impl.so 2>/dev/null | grep -c "llama_"
echo LOC2_DONE
