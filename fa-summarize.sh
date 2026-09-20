#!/bin/bash
# fa-summarize.sh -- compact view of test-backend-ops FLASH_ATTN_EXT perf rows
# for q8_0 K/V only (our servng口径: --cache-type-k q8_0 --cache-type-v q8_0).
awk '
/FLASH_ATTN_EXT\(/ {
  hdr=$0; getline res;
  if (hdr !~ /type_K=q8_0/ || hdr !~ /type_V=q8_0/) next;
  kv=""; nh=""; hsk=""; hsv="";
  n=split(hdr, a, ",");
  for (i=1;i<=n;i++) {
    if (a[i] ~ /kv=[0-9]/) { split(a[i],b,"="); kv=b[2] }
    if (a[i] ~ /nh=[0-9]/) { split(a[i],b,"="); nh=b[2] }
    if (a[i] ~ /hsk=[0-9]/) { split(a[i],b,"="); hsk=b[2] }
    if (a[i] ~ /hsv=[0-9]/) { split(a[i],b,"="); hsv=b[2] }
  }
  us=""; gf="";
  if (match(res, /[0-9.]+ us\/run/)) { us=substr(res,RSTART,RLENGTH-7) }
  if (match(res, /[0-9.]+ GFLOPS/)) { gf=substr(res,RSTART,RLENGTH-7) }
  printf "kv=%-6s nh=%-3s hsk=%-4s hsv=%-4s %10s us/run  %10s GFLOPS\n", kv, nh, hsk, hsv, us, gf
}' /tmp/opbench-fa.txt | sort -t= -k2 -n | head -70
echo "=== total q8_0 rows: $(grep -ac 'type_K=q8_0' /tmp/opbench-fa.txt)"
