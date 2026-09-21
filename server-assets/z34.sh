#!/bin/bash
# Z3/Z4: FA on/off and mixed KV dtypes at 8K, using a parameterized copy of the main harness.
while ! grep -q TPSWEEP_DONE /tmp/tpsweep.log 2>/dev/null; do sleep 30; done
cd /root || exit 1
sed -e 's/--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0/--flash-attn ${FA:-on} --cache-type-k ${KVK:-q8_0} --cache-type-v ${KVV:-q8_0}/' /root/p60-ab-harness.sh > /root/p60-ab-harness-kv.sh
if ! grep -q 'FA:-on' /root/p60-ab-harness-kv.sh; then echo SED_FAILED; exit 1; fi
chmod +x /root/p60-ab-harness-kv.sh
echo Z34_START
echo '===== Z3: -fa off, q8_0 KV ====='
env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=z3foff FA=off NPRED=512 bash /root/p60-ab-harness-kv.sh 2>&1 | grep -a -e prompt -e MEDIAN -e greedy -e 'launch failure'
echo '===== Z4a: -ctk q8_0 -ctv f16 ====='
env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=z4kqvf KVK=q8_0 KVV=f16 NPRED=512 bash /root/p60-ab-harness-kv.sh 2>&1 | grep -a -e prompt -e MEDIAN -e greedy -e 'launch failure'
echo '===== Z4b: -ctk f16 -ctv q8_0 ====='
env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=z4kfvq KVK=f16 KVV=q8_0 NPRED=512 bash /root/p60-ab-harness-kv.sh 2>&1 | grep -a -e prompt -e MEDIAN -e greedy -e 'launch failure'
echo Z34_DONE