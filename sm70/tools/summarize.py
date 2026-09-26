#!/usr/bin/env python3
"""summarize.py <run_dir> [nmax] - L3 table from stress.txt + gpu.csv, formulas as in sm70/BENCH.md.

TPOT = 1000 / tg (tg = (completion_tokens - 1) / decode_seconds, client side).
llama only (response timings): rounds = draft_n / nmax, AL = predicted_n / rounds, ms/round = predicted_ms / rounds.
"""
import json
import os
import re
import statistics
import sys


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, nd=2):
    return "-" if x is None else ("%." + str(nd) + "f") % x


def main():
    d = sys.argv[1]
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    rows = []
    errors = []
    for line in open(os.path.join(d, "stress.txt"), encoding="utf-8", errors="replace"):
        if not line.startswith("STRESS tag="):
            continue
        if " ERROR " in line:
            errors.append(line.strip()[:300])
            continue
        head, _, tjson = line.partition(" timings=")
        kv = dict(re.findall(r"(\w+)=(\S+)", head))
        timings = {}
        if tjson:
            try:
                timings = json.loads(tjson)
            except ValueError:
                pass
        tg = float(kv["tg_tps"])
        r = {
            "rep": kv["rep"],
            "ptok": int(kv["prompt_tokens"]),
            "ctok": int(kv["completion_tokens"]),
            "ttft": float(kv["ttft_s"]),
            "pp": float(kv["pp_tps"]),
            "tg": tg,
            "tpot": 1000.0 / tg if tg > 0 else None,
            "al": None, "msr": None, "acc": None,
        }
        dn = timings.get("draft_n")
        pn = timings.get("predicted_n")
        pm = timings.get("predicted_ms")
        if dn and pn and pm:
            rounds = dn / float(nmax)
            r["al"] = pn / rounds
            r["msr"] = pm / rounds
            r["acc"] = (timings.get("draft_n_accepted") or 0) / float(dn)
        rows.append(r)

    print("| rep | prompt_tok | TTFT s | pp t/s | tg t/s | TPOT ms | AL | ms/round | accept |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s | %d | %.2f | %.1f | %.2f | %s | %s | %s | %s |" % (
            r["rep"], r["ptok"], r["ttft"], r["pp"], r["tg"], fmt(r["tpot"]),
            fmt(r["al"]), fmt(r["msr"], 1), fmt(r["acc"], 3)))
    if rows:
        print("| **median** | %d | %s | %s | %s | %s | %s | %s | %s |" % (
            rows[0]["ptok"], fmt(med([r["ttft"] for r in rows])), fmt(med([r["pp"] for r in rows]), 1),
            fmt(med([r["tg"] for r in rows])), fmt(med([r["tpot"] for r in rows])),
            fmt(med([r["al"] for r in rows])), fmt(med([r["msr"] for r in rows]), 1),
            fmt(med([r["acc"] for r in rows]), 3)))
        ttfts = [r["ttft"] for r in rows]
        tpots = [r["tpot"] for r in rows if r["tpot"]]
        if len(rows) > 1:
            print("SPREAD ttft=%.1f%% tpot=%s" % (
                100.0 * (max(ttfts) - min(ttfts)) / statistics.median(ttfts),
                ("%.1f%%" % (100.0 * (max(tpots) - min(tpots)) / statistics.median(tpots))) if len(tpots) > 1 else "-"))
    for e in errors:
        print("RUN_ERROR " + e)

    peak = {}
    gpath = os.path.join(d, "gpu.csv")
    if os.path.exists(gpath):
        for line in open(gpath, encoding="utf-8", errors="replace"):
            p = [x.strip() for x in line.split(",")]
            if len(p) < 3:
                continue
            try:
                mib = int(p[2].split()[0])
            except ValueError:
                continue
            peak[p[1]] = max(peak.get(p[1], 0), mib)
    for g in sorted(peak):
        print("MEM_PEAK gpu=%s mib=%d" % (g, peak[g]))


if __name__ == "__main__":
    main()
