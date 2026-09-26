#!/usr/bin/env python3
"""op-perf.py - L1 op timing (sm70/BENCH.md): test-backend-ops on a shape file, N independent processes.

usage: LIBS=<lib dir> [GPU=2] [A/B env ...] op-perf.py <shape_file> <arm> [reps=3] [mode=perf|test]
perf: prints a median table (one row per timed case, in output order) and OP_DONE.
test: runs the correctness check once per rep and prints OK/FAIL counts (OP_TEST ...).
Raw output: /root/llm/test/sm70/op/<arm>/rep<N>.txt ([FAK] kernel lines included).
"""
import os
import re
import statistics
import subprocess
import sys


def main():
    shape, arm = sys.argv[1], sys.argv[2]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    mode = sys.argv[4] if len(sys.argv) > 4 else "perf"
    libs = os.environ["LIBS"]
    gpu = os.environ.get("GPU", "2")
    out = "/root/llm/test/sm70/op/%s" % arm
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
               LD_LIBRARY_PATH=libs + ":/usr/local/cuda/lib64", GGML_CUDA_FA_KERNEL_DEBUG="1")
    print("OP_BEGIN arm=%s mode=%s shape=%s gpu=%s libs=%s" % (arm, mode, shape, gpu, libs))
    rows = {}
    for r in range(1, reps + 1):
        p = subprocess.run([libs + "/test-backend-ops", mode, "--test-file", shape, "-b", "CUDA0"],
                           env=env, capture_output=True, text=True)
        txt = p.stdout + p.stderr
        with open("%s/rep%d.txt" % (out, r), "w", encoding="utf-8") as f:
            f.write(txt)
        if mode == "test":
            m = re.search(r"(\d+)/(\d+) tests passed", txt)
            passed = "%s/%s" % m.groups() if m else "-"
            n_fail = len(re.findall(r"\bFAIL\b", txt))
            print("OP_TEST arm=%s rep=%d rc=%d passed=%s fail_lines=%d" % (arm, r, p.returncode, passed, n_fail))
            continue
        print("OP_REP arm=%s rep=%d rc=%d" % (arm, r, p.returncode))
        k = 0
        for line in txt.splitlines():
            m = re.search(r"([\d.]+)\s+us/run", line)
            if not m:
                continue
            k += 1
            g = re.search(r"([\d.]+)\s+GB/s", line)
            row = rows.setdefault(k, {"name": line.strip().split(":")[0][:110], "us": [], "gb": []})
            row["us"].append(float(m.group(1)))
            if g:
                row["gb"].append(float(g.group(1)))
    if mode == "perf":
        print("| # | case | median us | min | max | median GB/s | n |")
        print("|---|---|---|---|---|---|---|")
        for k in sorted(rows):
            v = rows[k]
            gb = "%.1f" % statistics.median(v["gb"]) if v["gb"] else "-"
            print("| %d | %s | %.1f | %.1f | %.1f | %s | %d |" % (
                k, v["name"], statistics.median(v["us"]), min(v["us"]), max(v["us"]), gb, len(v["us"])))
    print("OP_DONE arm=%s out=%s" % (arm, out))


if __name__ == "__main__":
    main()
