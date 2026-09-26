#!/usr/bin/env python3
"""gate.py - L2 greedy gate for llama-server (sm70/BENCH.md).

Sends each fixed prompt to /completion with temperature 0 and prints one line per prompt:
  GATE tag=<tag> p=<i> text_sha=<16 hex> tok_sha=<16 hex> n_tok=<n> n_chars=<n>
With --ref <file> (a previous gate output) it also prints GATE_MATCH / GATE_DIFF per prompt and a verdict.
"""
import argparse
import hashlib
import json
import os
import re
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--prompts", default="/root/llm/test/sm70/tools/gate-prompts.txt")
    ap.add_argument("--out", default="/root/llm/test/sm70/gate")
    ap.add_argument("--ref", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prompts = [l.rstrip("\n") for l in open(a.prompts, encoding="utf-8") if l.strip()]

    got = {}
    for i, p in enumerate(prompts, 1):
        body = {"prompt": p, "n_predict": a.n, "temperature": 0.0, "top_k": 1, "seed": 0,
                "cache_prompt": False, "return_tokens": True}
        req = urllib.request.Request("http://127.0.0.1:%d/completion" % a.port,
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
        except Exception as e:  # noqa: BLE001 - report and continue
            print("GATE_ERROR tag=%s p=%d %s" % (a.tag, i, str(e)[:200]))
            continue
        with open(os.path.join(a.out, "%s-p%d.json" % (a.tag, i)), "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False)
        text = r.get("content", "")
        toks = r.get("tokens") or []
        ts = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        ks = hashlib.sha256(json.dumps(toks).encode("utf-8")).hexdigest()[:16] if toks else "-"
        got[i] = (ts, ks)
        print("GATE tag=%s p=%d text_sha=%s tok_sha=%s n_tok=%d n_chars=%d" % (a.tag, i, ts, ks, len(toks), len(text)))

    if a.ref:
        ref = {}
        for line in open(a.ref, encoding="utf-8"):
            m = re.match(r"GATE tag=\S+ p=(\d+) text_sha=(\S+) tok_sha=(\S+)", line)
            if m:
                ref[int(m.group(1))] = (m.group(2), m.group(3))
        ok = bool(ref) and len(got) == len(prompts)
        for i in range(1, len(prompts) + 1):
            same = i in got and i in ref and got[i] == ref[i]
            ok = ok and same
            print("%s p=%d" % ("GATE_MATCH" if same else "GATE_DIFF", i))
        print("GATE_VERDICT %s tag=%s ref=%s" % ("PASS" if ok else "FAIL", a.tag, a.ref))
    print("GATE_DONE tag=%s" % a.tag)


if __name__ == "__main__":
    main()
