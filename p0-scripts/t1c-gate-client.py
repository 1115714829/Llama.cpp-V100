#!/usr/bin/env python3
# T1-C end-to-end gate client: one greedy request on a fixed prompt, prints the
# sha256 of the completion so both arms (engine ON / OFF) can be compared.
import hashlib
import json
import sys
import urllib.request

tag = sys.argv[1] if len(sys.argv) > 1 else "g"
prompt = open("/root/llm/test/bl-prompt90.txt", encoding="utf-8").read()[:100000]
body = json.dumps({
    "prompt": prompt,
    "n_predict": 32,
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "seed": 1234,
}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8082/completion",
    data=body,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=1800) as resp:
    r = json.loads(resp.read())
content = r.get("content", "")
print("GATE tag=%s sha=%s len=%d predicted=%s" % (
    tag,
    hashlib.sha256(content.encode()).hexdigest(),
    len(content),
    r.get("tokens_predicted"),
))
