#!/bin/bash
# unlock.sh <task_id> - release the machine lock if (and only if) this task holds it.
ID=${1:?usage: unlock.sh <task_id>}
L=/root/llm/test/AGENT_LOCK
if [ ! -e "$L" ]; then echo "UNLOCK_NOOP no_lock"; exit 0; fi
if [ "$(cut -d' ' -f1 "$L")" != "$ID" ]; then echo "UNLOCK_REFUSED held_by $(cat "$L")"; exit 1; fi
rm -f "$L"
echo "UNLOCK_OK id=$ID"
