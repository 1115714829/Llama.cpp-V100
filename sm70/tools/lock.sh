#!/bin/bash
# lock.sh <task_id> [cards] - pre-flight "three checks" plus the machine lock, in one step.
# cards: comma list of physical GPU indices that must be idle (< 500 MiB and no compute process).
# Prints LOCK_OK, or LOCK_REFUSED <reason> and exits 1 (then stop and report, do not retry in a loop).
ID=${1:?usage: lock.sh <task_id> [cards]}
CARDS=${2:-}
L=/root/llm/test/AGENT_LOCK
if [ -e "$L" ]; then echo "LOCK_REFUSED held_by $(cat "$L")"; exit 1; fi
if [ -e /tmp/LLAMA_BUILD_LOCK ]; then echo "LOCK_REFUSED build_lock"; exit 1; fi
P=$(pgrep -af 'llama-serve[r]|vllm.entrypoint[s]|test-backend-op[s] |llama-benc[h] |cmake --buil[d]' | cut -c1-120)
if [ -n "$P" ]; then echo "LOCK_REFUSED busy_process"; echo "$P"; exit 1; fi
if [ -n "$CARDS" ]; then
  for g in $(echo "$CARDS" | tr ',' ' '); do
    m=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    a=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -c .)
    if [ -z "$m" ] || [ "$m" -ge 500 ] || [ "$a" -gt 0 ]; then echo "LOCK_REFUSED gpu$g mem=${m}MiB apps=$a"; exit 1; fi
  done
fi
echo "$ID $(date -Iseconds)" > "$L"
echo "LOCK_OK id=$ID cards=${CARDS:-none}"
