#!/bin/bash
LOG=/tmp/p60-fak1-server.log
echo "MKEYCOUNT=$(grep -a -c -e MKEY $LOG)"
echo "--- first 24 MKEY lines ---"
grep -a -e MKEY $LOG | head -24
echo "--- distinct n0 count per (i,j) pair ---"
grep -a -e MKEY $LOG | awk '{print $3, $4, $5}' | sort -u | awk '{print $1, $2}' | uniq -c
echo "--- n0 sequence i=0 j=0 ---"
grep -a -e MKEY $LOG | awk '$3 == "i=0" && $4 == "j=0" {print $2, $5, $6, $7}'
echo "--- n0 sequence i=1 j=0 ---"
grep -a -e MKEY $LOG | awk '$3 == "i=1" && $4 == "j=0" {print $2, $5, $6, $7}'
echo "--- all distinct n0 values overall ---"
grep -a -e MKEY $LOG | awk '{print $5}' | sort | uniq -c | sort -rn
echo "--- FAK ---"
grep -a -e FAK $LOG
echo "FAKCOUNT=$(grep -a -c -e FAK $LOG)"
