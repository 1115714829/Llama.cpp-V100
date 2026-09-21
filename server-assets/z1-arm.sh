#!/bin/bash
# z1-arm.sh -- Z1 context scan driver. Requires CTX, TAG, PORT in the environment.
# USESPEC=1 (default) sets SPEC to --spec-type none (no draft). USESPEC=0 leaves the harness default.
set -uo pipefail
U=${USESPEC:-1}
echo DRIVER start time=$(date +%H:%M:%S) CTX=$CTX TAG=$TAG PORT=$PORT USESPEC=$U
if [ $U = 1 ]; then
  export SPEC='--spec-type none'
else
  unset SPEC
fi
export CARDS=0,1,2
export SPLIT=tensor
export L=/root/libdir-instr
export P2P=1
export NPRED=128
echo SPEC_IS_SET=[${SPEC:-UNSET}]
if [ $U = 1 ] && [ "$SPEC" != '--spec-type none' ]; then echo SPEC_GUARD_FAIL; exit 2; fi
echo DRIVER env CARDS=$CARDS SPLIT=$SPLIT L=$L P2P=$P2P NPRED=$NPRED
bash /root/z1-harness.sh
echo DRIVER end time=$(date +%H:%M:%S)
