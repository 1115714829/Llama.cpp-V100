#!/bin/bash
# Sweep the Volta nwarps for ncols_dst = 5..8 (the speculative verify shape) and keep the best.
# The table is compile-time, so each candidate needs a rebuild. Base file is restored each round.
set -uo pipefail
SRC=/root/llm/test/v100-opt/llama.cpp
F=$SRC/ggml/src/ggml-cuda/mmvq.cu
BAK=/root/mmvq.sweep-base.cu

cp -a "$F" "$BAK"
echo "=== baseline VOLTA ncols 5..8 setting ==="
grep -A 14 'MMVQ_PARAMETERS_VOLTA) {' "$BAK" | head -18

for NW in 4 8 ; do
  echo ""
  echo "################ nwarps=$NW for ncols 5..8 ################"
  cp -a "$BAK" "$F"
  python3 - "$F" "$NW" <<'PY'
import re, sys
path, nw = sys.argv[1], sys.argv[2]
src = open(path).read()
key = "table_id == MMVQ_PARAMETERS_VOLTA) {"
i = src.index(key)
j = src.index("return 1;", i)          # end of the VOLTA block
block = src[i:j]
new = block.replace("""            case 5:
            case 6:
            case 7:
            case 8:
                return 2;""", """            case 5:
            case 6:
            case 7:
            case 8:
                return %s;""" % nw, 1)
assert new != block, "VOLTA ncols 5..8 pattern not found"
open(path, "w").write(src[:i] + new + src[j:])
print("patched VOLTA ncols 5..8 ->", nw)
PY
  sed -i 's/\r$//' "$F"

  nohup bash -c "cd $SRC && source /opt/rh/gcc-toolset-12/enable && cmake --build build --config Release -j82 --target llama-server" > /tmp/build-sweep-$NW.log 2>&1 &
  n=0
  while pgrep -f "[c]make --build" >/dev/null 2>&1; do sleep 10; n=$((n+10)); [ "$n" -gt 800 ] && break; done
  echo -n "build errors: "; grep -icE "error:|FAILED" /tmp/build-sweep-$NW.log
  cp -a "$SRC/build/bin/." /root/libdir-rt/

  NODROP=1 P2P=1 CARDS=0,1,2 SPLIT=tensor TAG=nw$NW bash /root/p60-ab-harness.sh || true
  echo "--- result nw=$NW"
  grep -a -e "^prompt" -e "^MEDIAN_TG" -e "spec timing:" /tmp/p60-nw$NW.log 2>/dev/null
done

echo ""
echo "=== restore the base source ==="
cp -a "$BAK" "$F"
echo "=== sweep summary (ncols_dst=8 verify shape) ==="
for t in nw4 nw8; do
  echo -n "$t median: "; grep -a "^MEDIAN_TG" /tmp/p60-$t.log 2>/dev/null | cut -d= -f2
done
echo -n "reference nw=2 (current code) median: "; grep -a "^MEDIAN_TG" /tmp/p60-ckpt-tensor.log 2>/dev/null | cut -d= -f2
echo P76_DONE
