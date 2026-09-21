#!/bin/bash
# convert-and-stats.sh -- QdstrmImporter turns the raw stream into an nsys-rep, then read it.
set -u
NSYS=/usr/local/cuda-12.4/bin/nsys
IMP=/usr/local/cuda-12.4/nsight-systems-2023.4.4/host-linux-ppc64le/QdstrmImporter
cd /tmp || exit 1
echo "=== importer help (first lines) ==="
"$IMP" --help 2>&1 | head -20
echo "=== convert ==="
"$IMP" -i /tmp/prof-tgt.qdstrm -o /tmp/prof-tgt.nsys-rep 2>&1 | tail -10
ls -l /tmp/prof-tgt.nsys-rep 2>&1
echo "=== kernel summary (cuda_gpu_kern_sum) ==="
"$NSYS" stats --report cuda_gpu_kern_sum /tmp/prof-tgt.nsys-rep 2>&1 | tail -40
echo CONVERT_DONE
