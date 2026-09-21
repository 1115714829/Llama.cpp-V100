#!/bin/bash
for i in $(seq 1 34); do
  if grep -q G4AB_DONE /tmp/g4-ab.log 2>/dev/null; then break; fi
  sleep 10
done
date +%H:%M:%S
tail -14 /tmp/g4-ab.log