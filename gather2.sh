#!/bin/bash
# Read-only: pull complete timing blocks + unit configs from AC922.
set -uo pipefail

for u in llama-server-c4 llama-server-test; do
  echo "############ $u : last 10 print_timing lines ############"
  journalctl -u "$u" --no-pager -o cat 2>/dev/null | grep -E "print_timing" | tail -10
  echo ""
done

echo "############ MTP / draft / spec warning lines ############"
for u in llama-server-c4 llama-server-test; do
  echo "--- $u ---"
  journalctl -u "$u" --no-pager -o cat 2>/dev/null | grep -iE "offload failed|CPU sampler|spec-type|draft-mtp|n_max|accepted" | tail -12
  echo ""
done

echo "############ unit file: llama-server-c4.service ############"
grep -vE "^#|^$" /etc/systemd/system/llama-server-c4.service 2>/dev/null
echo ""
echo "############ unit file: llama-server-test.service ############"
grep -vE "^#|^$" /etc/systemd/system/llama-server-test.service 2>/dev/null
echo ""

echo "############ /tmp/bench-256k-orig.log ############"
cat /tmp/bench-256k-orig.log 2>/dev/null
echo ""
echo "############ /tmp/resp256k.json (first 400 chars) ############"
head -c 400 /tmp/resp256k.json 2>/dev/null
echo ""
echo "############ /tmp/resp256k.json (last 400 chars) ############"
tail -c 400 /tmp/resp256k.json 2>/dev/null
echo ""
echo "############ C4 build-info / version ############"
/root/llm/test/v100-opt/llama.cpp/build/bin/llama-bench --version 2>&1 | head -5
echo ""
echo "############ orig build-info / version ############"
/root/llm/llama.cpp/bin/llama-bench --version 2>&1 | head -5
echo ""
echo GATHER2_DONE
