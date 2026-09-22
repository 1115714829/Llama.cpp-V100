#!/bin/bash
# A/B: does retrying the gallocr plan reuse (GGML_SCHED_RETRY_ALLOC) remove the per-call
# device sync + reserve, and does that show up in wall clock?
#
# Four arms ABBA on ONE library (the only variable is the env var, so the library md5 is the same
# on both sides). Baseline config = the adopted B3 invocation: CARDS=0,1,2 SPLIT=tensor P2P=1
# L=<libdir> NPRED=512 NODROP=1 GGML_META_FULLGRAPH=1.
set -u

LAB=/mnt/3.84t/v100-opt/kernel-lab
LOG=$LAB/schedretry-ab.log
MARK=$LAB/schedretry-ab.done
LIB=/root/libdir-retry
BUILT=/root/llm/test/v100-opt/llama.cpp/build-instr/bin

rm -f "$MARK"
exec >"$LOG" 2>&1
echo "=== START $(date -Is)"

fail() {
    echo "ABORT_$1"
    pkill -9 -x llama-server 2>/dev/null
    rm -f /tmp/LLAMA_BUILD_LOCK
    touch "$MARK"
    exit 1
}

# preflight: nothing else may own the GPUs or the machine
pgrep -f 'llama-benc[h] -m' >/dev/null && fail BUSY_BENCH
pgrep -f 'cmake --buil[d]'    >/dev/null && fail BUSY_CMAKE
pkill -9 -x llama-server 2>/dev/null
sleep 2
touch /tmp/LLAMA_BUILD_LOCK

# library under test = the adopted baseline with only libggml-base rebuilt
if [ ! -d "$LIB" ]; then
    cp -a /root/libdir-instr "$LIB"
fi
cp -f "$BUILT/libggml-base.so.0.24.0" "$LIB/libggml-base.so.0.24.0"
echo "lib under test:"
md5sum "$LIB/libggml-base.so.0.24.0" "$LIB/libggml-cuda.so.0.24.0" "$LIB/libllama.so.0.4.1" "$LIB/libllama-common.so.0.4.1"
echo "baseline libdir (/root/libdir-instr) for comparison:"
md5sum /root/libdir-instr/libggml-base.so.0.24.0 /root/libdir-instr/libggml-cuda.so.0.24.0

arm() {  # $1 = tag, $2 = port, $3 = retry (0/1)
    pkill -9 -x llama-server 2>/dev/null
    sleep 2
    for i in $(seq 1 60); do
        (exec 3<>/dev/tcp/127.0.0.1/"$2") 2>/dev/null || break
        sleep 1
    done
    if [ "$3" = "1" ]; then
        export GGML_SCHED_RETRY_ALLOC=1
    else
        unset GGML_SCHED_RETRY_ALLOC
    fi
    echo "=== ARM $1 port=$2 retry=$3 $(date -Is)"
    CARDS=0,1,2 SPLIT=tensor L="$LIB" P2P=1 TAG="$1" NPRED=512 NODROP=1 PORT="$2" \
        GGML_META_FULLGRAPH=1 bash /root/p60-ab-harness.sh
    echo "--- ARM $1 rc=$? $(date -Is)"
    pkill -9 -x llama-server 2>/dev/null
    sleep 2
}

arm c1 8331 0
arm r1 8332 1
arm r2 8333 1
arm c2 8334 0

echo ""
echo "=== SUMMARIES ==="
for t in c1 r1 r2 c2; do
    echo "--- $t"
    grep -a -e MEDIAN_TG -e "prompt1:" -e "prompt2:" -e "prompt3:" -e "greedy:" /tmp/p60-$t.log 2>/dev/null
    grep -a -F "[SCHED_RETRY]" /tmp/p60-$t-server.log 2>/dev/null | tail -2
    grep -a -F "[RT] perf:" /tmp/p60-$t-server.log 2>/dev/null | tail -1
done

pkill -9 -x llama-server 2>/dev/null
rm -f /tmp/LLAMA_BUILD_LOCK
echo "=== DONE $(date -Is)"
touch "$MARK"
