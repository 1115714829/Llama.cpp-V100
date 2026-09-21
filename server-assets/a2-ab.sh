#!/bin/bash
# A2 same-library A/B: 8K spec decode and 32K prefill, with and without index-based recurrent writes.
while ! grep -q A2_DONE /tmp/a2-run.log 2>/dev/null; do sleep 20; done
echo A2_SEEN
cd /root/llm/test/v100-opt/llama.cpp || exit 1
export CC=/opt/rh/gcc-toolset-12/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
find build-instr -name 'llama-graph.cpp.o' -o -name 'delta-net-base.cpp.o' -o -name 'llama-memory-recurrent.cpp.o' | xargs -r rm -f
cmake --build build-instr --config Release -j128 > /tmp/rs-build.log 2>&1
echo "RS_RC=$?"
echo -n 'error lines: '; grep -a -c -e error /tmp/rs-build.log
cp -a build-instr/bin/. /root/libdir-instr/
md5sum /root/libdir-instr/libllama.so.0.4.1
echo -n 'gate marker: '; strings /root/libdir-instr/libllama.so.0.4.1 | grep -a -c GGML_RS_INDEX_WRITE
cd /root
echo === ARM1 8K spec old path ====
env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=rsid_off NPRED=512 bash /root/p60-ab-harness.sh 2>&1 | grep -a -e prompt -e MEDIAN -e greedy
grep -a 'RT. perf' /tmp/p60-rsid_off-server.log 2>/dev/null | grep -a -e Qwen3.8-27B | tail -1
echo === ARM2 8K spec index writes ====
env CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=rsid_on NPRED=512 GGML_RS_INDEX_WRITE=1 bash /root/p60-ab-harness.sh 2>&1 | grep -a -e prompt -e MEDIAN -e greedy
grep -a 'RT. perf' /tmp/p60-rsid_on-server.log 2>/dev/null | grep -a -e Qwen3.8-27B | tail -1
echo === ARM3 32K prefill old path ====
env TAG=rsp_off CARDS=0,1,2 L=/root/libdir-instr KV=q8_0 CTX=32768 NPRED=8 bash /root/lc-spec.sh 2>&1 | grep -a -e prompt -e timings
echo === ARM4 32K prefill index writes ====
env GGML_RS_INDEX_WRITE=1 TAG=rsp_on CARDS=0,1,2 L=/root/libdir-instr KV=q8_0 CTX=32768 NPRED=8 bash /root/lc-spec.sh 2>&1 | grep -a -e prompt -e timings
echo --- target round probes ---
grep -a 'RT. perf' /tmp/lc-spec-rsp_off-server.log 2>/dev/null | grep -a -e Qwen3.8-27B | tail -1
grep -a 'RT. perf' /tmp/lc-spec-rsp_on-server.log 2>/dev/null | grep -a -e Qwen3.8-27B | tail -1
echo A2AB_DONE