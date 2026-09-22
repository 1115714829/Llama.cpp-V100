#!/usr/bin/env python3
"""Patch MMVQ's rows_per_block for Volta on wide batches (ncols_dst 5..8).

Why: at n_dst=8 the CTA reads the whole 8-token activation panel but only covers 2
weight rows, so the activation read traffic is ~4x the weight traffic. Raising
rows_per_block amortises that panel over more rows.

Usage: rpb-patch.py <path/to/mmvq.cu> <value>
  value > 0 : raise the Volta ncols 5..8 rows_per_block to <value>
  value = 0 : restore the pristine copy (<path>.orig-rpb)

The pristine copy is kept at <path>.orig-rpb. Patching is idempotent: the second
and later calls only rewrite the value macro.
"""
import hashlib
import os
import sys

ANCHOR_OLD = (
    "            case 2:\r\n"
    "            case 3:\r\n"
    "            case 4:\r\n"
    "            case 5:\r\n"
    "            case 6:\r\n"
    "            case 7:\r\n"
    "            case 8:\r\n"
    "                return 2;\r\n"
)
ANCHOR_NEW = (
    "            case 2:\r\n"
    "            case 3:\r\n"
    "            case 4:\r\n"
    "                return 2;\r\n"
    "            case 5:\r\n"
    "            case 6:\r\n"
    "            case 7:\r\n"
    "            case 8:\r\n"
    "                return table_id == MMVQ_PARAMETERS_VOLTA ? MMVQ_VOLTA_RPB_WIDE : 2;\r\n"
)
MACRO = "#define MMVQ_VOLTA_RPB_WIDE "
MACRO_ANCHOR = "static constexpr __host__ __device__ int calc_rows_per_block("


def md5_of(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    path = sys.argv[1]
    value = int(sys.argv[2])
    orig = path + ".orig-rpb"

    if value == 0:
        if not os.path.exists(orig):
            sys.exit("FAIL no pristine copy at " + orig)
        with open(orig, "rb") as f:
            data = f.read()
        with open(path, "wb") as f:
            f.write(data)
        print("RESTORED md5=" + md5_of(path))
        return

    if not os.path.exists(orig):
        with open(path, "rb") as f:
            data = f.read()
        with open(orig, "wb") as f:
            f.write(data)

    with open(path, "rb") as f:
        text = f.read().decode("utf-8")

    if MACRO in text:
        lines = text.split("\r\n")
        hits = 0
        for i, line in enumerate(lines):
            if line.startswith(MACRO):
                lines[i] = MACRO + str(value)
                hits += 1
        if hits != 1:
            sys.exit("FAIL macro occurrences = %d" % hits)
        text = "\r\n".join(lines)
    else:
        if text.count(ANCHOR_OLD) != 1:
            sys.exit("FAIL anchor count = %d" % text.count(ANCHOR_OLD))
        text = text.replace(ANCHOR_OLD, ANCHOR_NEW, 1)
        if text.count(MACRO_ANCHOR) != 1:
            sys.exit("FAIL function anchor count = %d" % text.count(MACRO_ANCHOR))
        text = text.replace(MACRO_ANCHOR, MACRO + str(value) + "\r\n" + MACRO_ANCHOR, 1)

    with open(path, "wb") as f:
        f.write(text.encode("utf-8"))
    print("PATCHED rpb=%d md5=%s" % (value, md5_of(path)))


if __name__ == "__main__":
    main()
