#!/bin/bash
# p4-build.sh - manifest-gated build for the V100 decode/prefill line.
#
#   1. scan the build tree and compare it with the workstation source manifest
#      (/tmp/tree-md5-local.txt, CR-normalized md5 of every .c/.cpp/.cu/.cuh/.h/.hpp)
#   2. refuse to build when the tree drifts (fail closed) - this is what makes the
#      measured binary traceable to an exact source snapshot
#   3. build, install to LIBS, write BUILD_MANIFEST.txt next to the binary
#
# Usage: bash /tmp/p4-build.sh            (env: TREE, LIBS, MAN, ALLOW)
set -uo pipefail
TREE=${TREE:-/root/llm/test/v100-opt/llama.cpp}
LIBS=${LIBS:-/root/libdir-gb}
MAN=${MAN:-/tmp/tree-md5-local.txt}
ALLOW=${ALLOW:-'^build-nccl/|^common/build-info.h$'}
LOG=${LOG:-/tmp/p4-build.log}
exec > "$LOG" 2>&1

echo "BUILD_START $(date)"
if [ -f /tmp/LLAMA_BUILD_LOCK ]; then echo LOCK_HELD; echo BUILD_DONE; exit 2; fi
if pgrep -f 'llama-serve[r] --model' >/dev/null; then echo BUSY_SERVER; echo BUILD_DONE; exit 2; fi
if pgrep -f 'cmake --buil[d]' >/dev/null; then echo BUSY_BUILD; echo BUILD_DONE; exit 2; fi
touch /tmp/LLAMA_BUILD_LOCK
trap 'rm -f /tmp/LLAMA_BUILD_LOCK' EXIT

if [ ! -f "$MAN" ]; then echo "NO_MANIFEST $MAN"; echo BUILD_DONE; exit 2; fi
if [ ! -f "$TREE/build-instr/CMakeCache.txt" ]; then echo "NO_BUILD_DIR $TREE/build-instr"; echo BUILD_DONE; exit 2; fi

bash /tmp/tree-md5.sh "$TREE" /tmp/tree-md5-now.txt || { echo TREE_SCAN_FAILED; echo BUILD_DONE; exit 2; }

tr -d '\r' < "$MAN" | LC_ALL=C sort > /tmp/p4-man-sorted.txt
LC_ALL=C sort /tmp/tree-md5-now.txt > /tmp/p4-now-sorted.txt
diff -u /tmp/p4-man-sorted.txt /tmp/p4-now-sorted.txt > /tmp/p4-diff.txt
awk -v allow="$ALLOW" 'BEGIN{n=0}
  /^[-+]/ && !/^[-+][-+][-+]/ {
    line = substr($0, 2); path = line; sub(/^[0-9a-f]+  /, "", path);
    if (path !~ allow) { print substr($0,1,1) " " path; n++ }
  }
  END { print "UNALLOWED_DIFFS=" n > "/tmp/p4-diff-count.txt" }
' /tmp/p4-diff.txt > /tmp/p4-diff-filtered.txt
N=$(sed -n 's/^UNALLOWED_DIFFS=//p' /tmp/p4-diff-count.txt)
echo "MANIFEST_FILES=$(wc -l < "$MAN")"
echo "MANIFEST_DIFFS=$N"
if [ -z "$N" ] || [ "$N" != "0" ]; then
  echo "REFUSE_BUILD_TREE_DRIFT (-/<md5> = manifest only, +/<md5> = tree only)"
  head -60 /tmp/p4-diff-filtered.txt
  echo BUILD_DONE
  exit 3
fi

cd "$TREE" || { echo NO_TREE; echo BUILD_DONE; exit 2; }
cmake --build build-instr -j176 --target llama-server llama-bench > /tmp/p4-make.log 2>&1
BUILD_RC=$?
echo "BUILD_RC=$BUILD_RC"
echo "ERROR_LINES=$(grep -c 'error:' /tmp/p4-make.log || true)"
if [ "$BUILD_RC" != "0" ]; then
  grep -m 30 -a 'error' /tmp/p4-make.log
  echo BUILD_DONE
  exit 1
fi

mkdir -p "$LIBS"
cp -a build-instr/bin/. "$LIBS/"
{
  echo "BUILD_TIME=$(date -Iseconds)"
  echo "TREE=$TREE"
  echo "LIBS=$LIBS"
  echo "MANIFEST_SHA256=$(sha256sum "$MAN" | cut -d' ' -f1)"
  echo "MANIFEST_FILES=$(wc -l < "$MAN")"
  echo "LIBGGML_CUDA_MD5=$(md5sum "$LIBS/libggml-cuda.so.0.24.0" | cut -d' ' -f1)"
  echo "LLAMA_SERVER_MD5=$(md5sum "$LIBS/llama-server" | cut -d' ' -f1)"
  echo "BUILD_RC=$BUILD_RC"
} > "$LIBS/BUILD_MANIFEST.txt"
cat "$LIBS/BUILD_MANIFEST.txt"
echo BUILD_DONE
