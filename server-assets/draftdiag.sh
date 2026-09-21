#!/bin/bash
# Draft diagnosis: inject vs block split, draft ctx graph reuse, and whether CUDA graphs matter.
cd /root
SUM=/tmp/draftdiag-summary.txt
: > $SUM
run_arm() {
  tag=$1; shift
  echo "=== ARM $tag : $* ===" >> $SUM
  env "$@" GGML_CUDA_AR_TIMING=1 CARDS=0,1,2 SPLIT=tensor L=/root/libdir-instr P2P=1 TAG=$tag NPRED=384 NODROP=1 \
    bash /root/p60-ab-harness.sh > /tmp/$tag-driver.log 2>&1
  grep -a tg= /tmp/$tag-driver.log >> $SUM
  grep -a MEDIAN_TG /tmp/$tag-driver.log >> $SUM
  grep -a inject.timing /tmp/p60-$tag-server.log | tail -2 >> $SUM
  grep -a spec.timing /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a draft.ctx /tmp/p60-$tag-server.log | tail -2 >> $SUM
  grep -a ar_us_avg /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a calls= /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a decode+sync /tmp/p60-$tag-server.log | tail -1 >> $SUM
  grep -a n_ctx=8192 /tmp/p60-$tag-server.log | tail -1 >> $SUM
  echo "--- done $tag" >> $SUM
}
run_arm dd1
run_arm dd2 NOGRAPH=1
echo DRAFTDIAG_DONE >> $SUM
