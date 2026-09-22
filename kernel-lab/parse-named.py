#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse a test-backend-ops perf console dump: the case name and the timing share one line."""
import re
import sys

path = sys.argv[1]
rows = []
for ln in open(path, errors="replace"):
    s = ln.strip()
    m = re.match(r"^(.+?):\s+(\d+)\s+runs\s+-\s+([\d.]+)\s+(us|ms)/run", s)
    if not m:
        continue
    v = float(m.group(3)) * (1000.0 if m.group(4) == "ms" else 1.0)
    rows.append((v, m.group(1)))

rows.sort(reverse=True)
tot = sum(v for v, _ in rows)
print("=== %d timed ops, total %.1f ms ===" % (len(rows), tot / 1000.0))
for v, n in rows[:25]:
    print("%12.1f us  %5.1f%%  %s" % (v, 100.0 * v / tot, n[:118]))
