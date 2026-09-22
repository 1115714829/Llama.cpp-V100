#!/bin/bash
# Named per-op table for the prefill batch: keep the case-name lines so times can be attributed.
set -u
BIN=/root/llm/test/v100-opt/llama.cpp/build-instr/bin
LAB=/mnt/3.84t/v100-opt/kernel-lab
export CUDA_VISIBLE_DEVICES=0
export GGML_CUDA_DISABLE_GRAPHS=1

echo "=== perf with names, -ub 2048 $(date -Is)"
"$BIN/test-backend-ops" perf --test-file /tmp/ops-ub2048.txt -b CUDA0 2>&1 \
    | sed 's/\x1b\[[0-9;]*m//g' > "$LAB/perf-named-ub2048.txt"

python3 - "$LAB/perf-named-ub2048.txt" <<'PY'
import re, sys
lines = open(sys.argv[1], errors="replace").read().splitlines()
rows = []
pending = None
for ln in lines:
    s = ln.strip()
    m = re.match(r"^(\d+)\s+runs\s+-\s+([\d.]+)\s+(us|ms)/run", s)
    if m:
        v = float(m.group(2)) * (1000.0 if m.group(3) == "ms" else 1.0)
        rows.append((v, pending or "?"))
        pending = None
        continue
    if s and not s.startswith(("Backend", "Device", "ggml_", "Testing", "2/2", "OK", "Skipping")):
        pending = s[:110]
rows.sort(reverse=True)
print("=== top 25 ops by time (-ub 2048, single card) ===")
for v, name in rows[:25]:
    print("%12.1f us  %s" % (v, name))
tot = sum(v for v, _ in rows)
print("=== total timed: %.1f ms over %d ops ===" % (tot / 1000.0, len(rows)))
PY
echo "=== DONE $(date -Is)"
