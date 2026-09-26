#!/usr/bin/env python3
"""stress-256k client: one identical streaming request shape for vLLM and llama-server.

Stdlib only (urllib). Measures TTFT / prefill t/s / decode t/s over SSE chat
completions. Prompt is generated once and cached so every engine sees the same
bytes. Usage token counts come from the API when present (authoritative).
"""
import argparse
import json
import os
import sys
import time
import urllib.request

FILLER = (
    "In a large software system, the coordinator keeps a table of pending "
    "tasks, assigns each task to a worker, and checks the results when the "
    "workers finish. Some tasks depend on earlier tasks, so the coordinator "
    "keeps the dependency order in a list and only starts a task after its "
    "predecessors are done. When a worker reports a failure, the coordinator "
    "retries the task once and then marks it for review. "
)


def build_prompt(target_tokens):
    # Indexed sentences defeat speculative-decode gaming on pure repetition
    # (skill warning: degenerate output inflates tg). Calibration 4.86
    # chars/token accounts for the "[i] " index tokens added per sentence.
    n = target_tokens // 45 + 2
    body = "".join("[%d] %s" % (i, FILLER) for i in range(n))
    return body[: int(target_tokens * 4.86)]


def load_prompt(args):
    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    prompt = build_prompt(args.prompt_tokens)
    if args.prompt_file:
        with open(args.prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)
    return prompt


def one_run(args, prompt, rep):
    # Per-rep salt defeats engine-side prefix caching so prefill is real work.
    # The salt depends only on rep => byte-identical across engines for the
    # same rep index (cross-engine comparability preserved).
    content = "run %d\n%s" % (rep, prompt)
    base = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.gen,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
        "repeat_penalty": 1.05,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # vLLM accepts the full shape; llama-server 400s on unknown fields
    # (stream_options / repetition_penalty). Fall back stepwise and record
    # which variant ran so cross-engine calls stay auditable.
    variants = [("full", []), ("no_streamopt", ["stream_options"]),
                ("llama_min", ["stream_options", "repetition_penalty"]),
                ("bare", ["stream_options", "repetition_penalty",
                          "repeat_penalty", "top_k"])]
    hdrs = {"Content-Type": "application/json"}
    if args.api_key:
        hdrs["Authorization"] = "Bearer " + args.api_key
    resp = None
    used = ""
    err = ""
    for label, drops in variants:
        b = dict(base)
        for k in drops:
            b.pop(k, None)
        req = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(b).encode("utf-8"), headers=hdrs)
        t0 = time.time()
        try:
            resp = urllib.request.urlopen(req, timeout=args.timeout)
            used = label
            break
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", "replace")[:300]
            if e.code != 400:
                print("STRESS tag=%s rep=%d ERROR http %d body=%s"
                      % (args.tag, rep, e.code, err))
                return
    if resp is None:
        print("STRESS tag=%s rep=%d ERROR all variants 400 body=%s"
              % (args.tag, rep, err))
        return
    t_first = None
    n_chars = 0
    fields_seen = set()
    finish = ""
    usage = {}
    timings = {}
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("timings"):
                timings = chunk["timings"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                texts = [delta.get("content") or "", delta.get("reasoning_content") or "",
                         delta.get("reasoning") or "", delta.get("text") or ""]
                for tc in (delta.get("tool_calls") or []):
                    texts.append(((tc.get("function") or {}).get("arguments")) or "")
                for k, v in delta.items():
                    if v:
                        fields_seen.add(k)
                got = "".join(texts)
                if got:
                    n_chars += len(got)
                    if t_first is None:
                        t_first = time.time()
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    t_end = time.time()
    if t_first is None:
        print("STRESS tag=%s rep=%d ERROR no tokens (finish=%s usage=%s fields=%s)"
              % (args.tag, rep, finish, json.dumps(usage, separators=(",", ":")),
                 ",".join(sorted(fields_seen))))
        return
    ttft = t_first - t0
    dec = t_end - t_first
    ptoks = usage.get("prompt_tokens", 0)
    ctoks = usage.get("completion_tokens", 0)
    pp = ptoks / ttft if ptoks else 0.0
    tg = (ctoks - 1) / dec if ctoks > 1 and dec > 0 else 0.0
    extra = ""
    if timings:
        extra = " timings=" + json.dumps(timings, separators=(",", ":"))
    print("STRESS tag=%s rep=%d v=%s ttft_s=%.3f pp_tps=%.2f tg_tps=%.2f "
          "prompt_tokens=%d completion_tokens=%d gen_chars=%d fields=%s%s"
          % (args.tag, rep, used, ttft, pp, tg, ptoks, ctoks, n_chars,
             ",".join(sorted(fields_seen)), extra))
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=235930)
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()
    prompt = load_prompt(args)
    if args.prompt_file:
        with open(args.prompt_file + ".meta", "w", encoding="utf-8") as f:
            f.write("base_chars=%d gen=%d reps=%d\n" % (len(prompt), args.gen, args.reps))
    print("STRESS-BEGIN tag=%s prompt_chars=%d gen=%d reps=%d url=%s"
          % (args.tag, len(prompt), args.gen, args.reps, args.base_url))
    sys.stdout.flush()
    for rep in range(1, args.reps + 1):
        one_run(args, prompt, rep)
    print("STRESS-DONE tag=%s" % args.tag)


if __name__ == "__main__":
    main()
