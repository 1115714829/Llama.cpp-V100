#!/bin/bash
set -uo pipefail
echo "=== full stress-orig.out (look for the OOM / context error) ==="
grep -iE "error|failed|out of memory|oom|alloc|context|kv|cache|not enough|insufficient" /tmp/stress-orig.out | head -40
echo "=== last 40 lines ==="
tail -40 /tmp/stress-orig.out
echo LOG_DONE
