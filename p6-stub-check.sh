#!/bin/bash
# Is llama-bench/llama-server a thin stub + shared libs? If so, copying the exe alone is meaningless.
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp

echo "=== [1] build/bin contents ==="
ls -la "$SRC/build/bin/" | head -40
echo ""
echo "=== [2] md5 of the 4 saved 'variant' binaries ==="
md5sum /root/bin-b11053-c4-server /root/bin-b11053-c4-bench \
       /root/bin-b11053-pristine-server /root/bin-b11053-pristine-bench \
       /root/llm/llama.cpp/bin/llama-bench 2>/dev/null
echo ""
echo "=== [3] ldd of a copied stub (where do libs come from?) ==="
ldd /root/bin-b11053-pristine-bench 2>&1 | head -15
echo ""
echo "=== [4] RPATH / RUNPATH / NEEDED ==="
readelf -d /root/bin-b11053-pristine-bench 2>/dev/null | grep -iE "rpath|runpath|needed|origin"
echo ""
echo "=== [5] lib sizes: build tree vs install prefix ==="
ls -la "$SRC/build/bin/"*.so* 2>/dev/null
echo "--- install prefix ---"
ls -la /root/llm/llama.cpp/lib64/*.so* 2>/dev/null | head -12
echo ""
echo "=== [6] md5 build-tree libs vs install-prefix libs ==="
for l in libllama.so libggml-cuda.so libggml-base.so; do
  a=$(md5sum "$SRC/build/bin/$l" 2>/dev/null | awk '{print $1}')
  b=$(md5sum "/root/llm/llama.cpp/lib64/$l" 2>/dev/null | awk '{print $1}')
  echo "$l  build=$a  install=$b"
done
echo ""
echo "=== [7] is the exe actually a wrapper? (symbol/main) ==="
file /root/bin-b11053-pristine-bench
echo ""
echo "=== [8] does the bench impl lib carry the real code? ==="
ls -la "$SRC/build/bin/libllama-bench-impl.so" 2>/dev/null
nm -D "$SRC/build/bin/libllama-bench-impl.so" 2>/dev/null | head -5
echo DONE
