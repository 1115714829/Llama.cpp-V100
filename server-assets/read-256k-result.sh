#!/bin/bash
set -uo pipefail
echo "=== current test result (journal, last 3 min) ==="
journalctl -u llama-server-test --since "3 min ago" --no-pager 2>/dev/null | grep -iE "prompt eval time|eval time|total time|draft acceptance|n_gen" | tail -8
echo "=== resp json (new, tail) ==="
tail -c 300 /tmp/resp256k.json 2>/dev/null
echo ""
echo R256_DONE
