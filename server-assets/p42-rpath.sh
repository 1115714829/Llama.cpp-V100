#!/bin/bash
# Is the runtime loading a DIFFERENT libllama.so than the one we patch/snapshot?
set -uo pipefail
L=/root/libdir-rt

echo "=== dynamic tags of the launcher stub ==="
readelf -d "$L/llama-server" 2>/dev/null | grep -e RPATH -e RUNPATH -e NEEDED

echo ""
echo "=== does an INSTALLED copy exist at the recipe's CMAKE_INSTALL_RPATH? ==="
ls -l --time-style=full-iso /root/llm/llama.cpp/lib64/ 2>/dev/null | head -20

echo ""
echo "=== marker (new code) in each candidate libllama ==="
for f in /root/libdir-rt/libllama.so.0.4.1 \
         /root/llm/test/v100-opt/llama.cpp/build/bin/libllama.so.0.4.1 \
         /root/llm/llama.cpp/lib64/libllama.so.0.4.1 \
         /root/llm/llama.cpp/lib64/libllama.so ; do
  if [ -e "$f" ]; then
    echo "$(grep -c -a 'round timing (LLAMA_ROUND_TIMING)' "$f" 2>/dev/null)  $f"
  else
    echo "(absent)  $f"
  fi
done

echo ""
echo "=== what ldd resolves libllama.so.0 to, WITH and WITHOUT LD_LIBRARY_PATH ==="
echo "-- without LD_LIBRARY_PATH --"
ldd "$L/llama-server" 2>/dev/null | grep -i "libllama\.so"
echo "-- with LD_LIBRARY_PATH=$L --"
LD_LIBRARY_PATH="$L" ldd "$L/llama-server" 2>/dev/null | grep -i "libllama\.so"
echo --with LD_LIBRARY_PATH=$L -- > /dev/null
echo LOC3_DONE
