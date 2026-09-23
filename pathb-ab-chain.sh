#!/bin/bash
# ABBA chain for Path B q8-direct vs staged. Control=0, Feature=1.
set -uo pipefail
bash /root/pathb-ab.sh a1 32768 2 0 1
bash /root/pathb-ab.sh a2 32768 2 1 1
bash /root/pathb-ab.sh a3 32768 2 1 1
bash /root/pathb-ab.sh a4 32768 2 0 1
echo CHAIN_DONE
