#!/bin/bash
# Full-tree source manifest of the build tree, CR-normalized so the Windows
# working copy can be compared byte for byte.
#   bash /tmp/tree-md5.sh [tree-root] [out-file]
set -u
T=${1:-/root/llm/test/v100-opt/llama.cpp}
OUT=${2:-/tmp/tree-md5-server.txt}
cd "$T" || { echo "NO_TREE $T"; exit 2; }
find . \( -path ./build-instr -o -path ./build -o -path ./.git \) -prune -o -type f \
  \( -name '*.c' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.hpp' \) -print \
  | sed 's|^\./||' | sort > /tmp/files-server.txt
: > "$OUT"
n=0
while IFS= read -r f; do
  h=$(tr -d '\r' < "$f" | md5sum | cut -d' ' -f1)
  printf '%s  %s\n' "$h" "$f" >> "$OUT"
  n=$((n+1))
done < /tmp/files-server.txt
echo "TREE=$T"
echo "COUNT=$n"
echo "OUT=$OUT"
