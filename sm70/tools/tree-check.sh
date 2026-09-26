#!/bin/bash
# tree-check.sh - compare the server build tree with the uploaded workstation manifest. Exit 0 = identical.
# env: TREE (build tree), MAN (uploaded manifest), ALLOW (regex of paths ignored on both sides),
#      NEWER_THAN (optional file, e.g. a built library: lists tree files modified after it = provenance check)
set -u
TREE=${TREE:-/root/llm/test/v100-opt/llama.cpp}
MAN=${MAN:-/root/llm/test/sm70/manifest-local.txt}
ALLOW=${ALLOW:-'^common/build-info[.]h$'}
W=/root/llm/test/sm70
[ -f "$MAN" ] || { echo "TREE_CHECK_REFUSED no_manifest $MAN"; exit 2; }
bash $W/tools/tree-md5.sh "$TREE" $W/manifest-server.txt > /dev/null || { echo TREE_CHECK_FAILED scan; exit 2; }
tr -d '\r' < "$MAN" | awk -v a="$ALLOW" '{ p = substr($0, 35); if (p !~ a) print }' > $W/manifest-local.cmp
awk -v a="$ALLOW" '{ p = substr($0, 35); if (p !~ a) print }' $W/manifest-server.txt > $W/manifest-server.cmp

if [ -n "${NEWER_THAN:-}" ]; then
  awk '{ print substr($0, 35) }' $W/manifest-server.cmp | while IFS= read -r f; do
    [ "$TREE/$f" -nt "$NEWER_THAN" ] && echo "$f"
  done > $W/newer-than.txt
  echo "NEWER_THAN_REF count=$(wc -l < $W/newer-than.txt) ref=$NEWER_THAN"
  head -20 $W/newer-than.txt
fi

if diff $W/manifest-local.cmp $W/manifest-server.cmp > $W/manifest-diff.txt; then
  echo "TREE_MATCH files=$(wc -l < $W/manifest-server.cmp) sha256=$(sha256sum $W/manifest-server.cmp | cut -c1-16)"
  exit 0
fi
echo "TREE_DRIFT lines=$(grep -c '^[<>]' $W/manifest-diff.txt) (< = workstation only or different, > = server only or different)"
grep '^[<>]' $W/manifest-diff.txt | head -60
exit 3
