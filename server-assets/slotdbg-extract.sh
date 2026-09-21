#!/bin/bash
SLOG=/tmp/p60-slotdbg-server.log
echo "=== SLOT total lines ==="
grep -a -c -F '[SLOT]' ${SLOG}
echo "=== SLOT calls per ctx ==="
grep -a -F '[SLOT]' ${SLOG} | sed -e 's/ call=.*//' | sort | uniq -c
echo "=== DRAFT (DFlash2) first 12 ==="
grep -a -F '[SLOT]' ${SLOG} | grep -a DFlash2 | head -12
echo "=== TARGET first 6 ==="
grep -a -F '[SLOT]' ${SLOG} | grep -a -v DFlash2 | head -6
echo "=== RT perf lines ==="
grep -a -h -e 'RT. perf' ${SLOG}
echo "=== DRAFT slot sequence (all) ==="
grep -a -F '[SLOT]' ${SLOG} | grep -a DFlash2 | sed -e 's/.*call=/call=/' | awk '{print $1, $2, $3}'
echo "=== RUN LOG ==="
cat /tmp/p60-slotdbg-run.log
echo "=== SERVER LOG TAIL ==="
tail -6 ${SLOG}
