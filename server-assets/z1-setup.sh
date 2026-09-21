#!/bin/bash
# z1-setup.sh -- make a CTX-parameterized COPY of the p60 harness for the Z1 context scan.
# The original /root/p60-ab-harness.sh is left untouched so historical numbers stay comparable.
# Idempotent: always restarts from the original.
set -uo pipefail

cp -f /root/p60-ab-harness.sh /root/z1-harness.sh

# insert a CTX knob as its own line right after PORT (sed append, no backslash-n games)
sed -i '/^PORT=/a CTX=${CTX:-8192}' /root/z1-harness.sh

sed -i 's|--ctx-size 8192|--ctx-size $CTX|' /root/z1-harness.sh
sed -i 's|ctx=8192|ctx=$CTX|' /root/z1-harness.sh

echo === z1-harness.sh lines 15-22 ===
sed -n 15,22p /root/z1-harness.sh
echo === ctx-size line ===
grep -n -e 'ctx-size' /root/z1-harness.sh
echo === syntax check ===
bash -n /root/z1-harness.sh
echo SYNTAX_RC=$?
echo === default resolves to 8192 without CTX set ===
grep -n -e 'CTX=' /root/z1-harness.sh
echo === original untouched ===
grep -c -e 'ctx-size 8192' /root/p60-ab-harness.sh
