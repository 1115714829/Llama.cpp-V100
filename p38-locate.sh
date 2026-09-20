#!/bin/bash
# Locate which shared object actually contains the llama-context code that the server runs.
set -uo pipefail
L=/root/libdir-rt
echo "=== libraries in the snapshot ==="
ls -l "$L"/*.so* 2>/dev/null

echo ""
echo "=== which of them contain the new instrumentation marker? ==="
for f in "$L"/*.so* ; do
  c=$(grep -c -a llama_round_timing "$f" 2>/dev/null)
  echo "$c  $f"
done

echo ""
echo "=== which of them contain an older llama-context-only marker? ==="
for f in "$L"/*.so* ; do
  c=$(grep -c -a LLAMA_GRAPH_REUSE_DISABLE "$f" 2>/dev/null)
  echo "$c  $f"
done

echo ""
echo "=== what does the server actually link? ==="
LD_LIBRARY_PATH="$L" ldd "$L/llama-server" 2>/dev/null | grep -i -e llama -e ggml

echo ""
echo "=== does llama-server load an impl lib? ==="
nm -D "$L/llama-server" 2>/dev/null | grep -c -i -e llama_decode -e llama_new_context
echo "--- strings hint ---"
grep -a -o -e "libllama[a-z-]*\.so[a-z0-9.]*" "$L/llama-server" 2>/dev/null | sort -u | head -10
echo LOCATE_DONE
