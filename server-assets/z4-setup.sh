#!/bin/bash
# z4-setup.sh -- make a CTKV-parameterized COPY of the p60 harness for the Z4 KV dtype A/B.
# The original /root/p60-ab-harness.sh is left untouched so historical numbers stay comparable.
# Idempotent: always restarts from the original.
set -uo pipefail

cp -f /root/p60-ab-harness.sh /root/z4-harness.sh

# add a CTKV knob as its own line right after the PORT line
sed -i '/^PORT=/a CTKV=${CTKV:-q8_0}' /root/z4-harness.sh

sed -i 's|--cache-type-k q8_0|--cache-type-k $CTKV|' /root/z4-harness.sh
sed -i 's|--cache-type-v q8_0|--cache-type-v $CTKV|' /root/z4-harness.sh
sed -i 's|cache-type-k q8_0|cache-type-k $CTKV|' /root/z4-harness.sh

echo === check ===
grep -n -e 'CTKV=' -e 'cache-type' /root/z4-harness.sh
echo === syntax ===
bash -n /root/z4-harness.sh
echo SYNTAX_RC=$?
echo === original untouched, cache-type count ===
grep -c -e 'cache-type-k q8_0' /root/p60-ab-harness.sh
echo === also report how many cache-type occurrences exist at all ===
grep -c -e 'cache-type' /root/p60-ab-harness.sh
