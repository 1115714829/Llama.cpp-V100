#!/usr/bin/env python3
# Summarise /root/p60-<tag>-p{1,2,3}.json into tg / AL / ms-per-round.
# Usage: ab_summary.py [--nmax 7] TAG [TAG ...]

import json
import os
import sys


def load(tag, nmax):
    rows = []
    for p in (1, 2, 3):
        path = '/tmp/p60-%s-p%d.json' % (tag, p)
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            d = json.load(fh)
        t = d.get('timings', d)
        pn = float(t['predicted_n'])
        pm = float(t['predicted_ms'])
        dn = float(t.get('draft_n') or 0)
        rounds = dn / nmax if dn else pn
        rows.append((pn * 1e3 / pm, pn / rounds, pm / rounds))
    return rows


def main():
    args = sys.argv[1:]
    nmax = 7.0
    if args and args[0] == '--nmax':
        nmax = float(args[1])
        args = args[2:]
    if not args:
        print('usage: ab_summary.py [--nmax 7] TAG [TAG ...]')
        return 1
    for tag in args:
        rows = load(tag, nmax)
        if rows is None:
            print('%s MISSING response json' % tag)
            continue
        tg = ' '.join('%.2f' % r[0] for r in rows)
        al = ' '.join('%.2f' % r[1] for r in rows)
        msr = ' '.join('%.2f' % r[2] for r in rows)
        print('%s tg=%s | AL=%s | ms_per_round=%s | mean_msr=%.2f' % (tag, tg, al, msr, sum(r[2] for r in rows) / 3.0))
    return 0


if __name__ == '__main__':
    sys.exit(main())