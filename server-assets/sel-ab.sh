#!/bin/bash
# Build selector+sched changes (abort on build failure), then measure 2 arms.
cd /root/llm/test/v100-opt/llama.cpp
cmake --build build-instr --config Release -j128 > /tmp/buildsel.log 2>&1
if [ $? -ne 0 ]; then echo BUILD_FAILED; grep -a error /tmp/buildsel.log | head -5; exit 1; fi
echo BUILD_OK
cp -a build-instr/bin/. /root/libdir-instr/
md5sum /root/libdir-instr/libggml-cuda.so.0.24.0 /root/libdir-instr/libllama-common.so.0.4.1 /root/libdir-instr/libggml-base.so.0.9.1 2>/dev/null
SUM=/tmp/sel-ab.txt
: > $SUM
run_arm() {
  tag=$1; shift
  echo "=== ARM $tag : $* ===" >> $SUM
  env "$@" GGML_SCHED_SPLIT_TIMING=1 CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=$tag NPRED=384 NODROP=1 \
    bash /root/p60-ab-harness.sh > /tmp/$tag-driver.log 2>&1
  grep -a tg= /tmp/$tag-driver.log >> $SUM
  grep -a MEDIAN_TG /tmp/$tag-driver.log >> $SUM
  grep -a greedy /tmp/$tag-driver.log >> $SUM
  grep -a selector.cpu /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a SCHED /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a spec.timing /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a n_ctx=8192 /tmp/p60-$tag-server.log | tail -1 >> $SUM
}
run_arm selA
run_arm selB GGML_SCHED_SPLIT_CACHE=1
echo SEL_AB_DONE >> $SUM
