#!/bin/bash
SLOG=/tmp/p60-dirprobe-server.log
echo "=== A: grep -a -e DIRECT SLOG | head -30 ==="
grep -a -e DIRECT "$SLOG" | head -30
echo "=== B: grep -a -c props changed ==="
grep -a -c 'props changed' "$SLOG"
echo "=== C: grep -a -e calls= | tail -1 ==="
grep -a -e calls= "$SLOG" | tail -1
echo "=== D: DIRECT_PROBE summary lines (last 20) ==="
grep -a -e 'DIRECT_PROBE]' "$SLOG" | tail -20
echo "=== E: DIRECT_PROBE detail lines ==="
grep -a -e 'DIRECT_PROBE-DETAIL' "$SLOG" | head -6
echo "=== F: driver log tail ==="
tail -14 /tmp/dirprobe-driver.log
echo "=== G: harness result lines ==="
grep -a -e P60_DONE -e 'tokens per second' -e 'mean len' -e health /tmp/p60-dirprobe.log | tail -20
echo "REPORT_DONE"
