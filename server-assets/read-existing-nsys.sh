#!/bin/bash
# read-existing-nsys.sh -- use the nsys reports that were successfully written earlier today.
set -u
NSYS=/usr/local/cuda-12.4/bin/nsys
for R in /tmp/nsys-pristine.nsys-rep /tmp/nsys-volta2.nsys-rep; do
  echo "########## $R"
  ls -l "$R" 2>&1
  echo "--- kernel sum ---"
  "$NSYS" stats --report cuda_gpu_kern_sum "$R" 2>&1 | tail -30
done
echo READ_NSYS_DONE
