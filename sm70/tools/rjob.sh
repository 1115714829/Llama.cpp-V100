#!/bin/bash
# rjob.sh <name> <command string>
# Run <command string> in a tmux session named <name> and return immediately.
# When the command exits, "JOB_DONE_<name> rc=<rc>" is appended to the log; rwait.sh blocks on it.
# Log: /root/llm/test/sm70/logs/<name>.log (LOG=/path overrides).
set -u
NAME=${1:?usage: rjob.sh <name> <command string>}
shift
CMD="$*"
LOG=${LOG:-/root/llm/test/sm70/logs/$NAME.log}
MARK="JOB_DONE_$NAME"
WRAP=/root/llm/test/sm70/logs/rjob-$NAME.sh

mkdir -p "$(dirname "$LOG")"
if tmux has-session -t "$NAME" 2>/dev/null; then echo "RJOB_REFUSED session $NAME exists"; exit 2; fi
: > "$LOG"
printf '#!/bin/bash\n%s\nrc=$?\necho "%s rc=$rc" >> %s\n' "$CMD" "$MARK" "$LOG" > "$WRAP"
tmux new-session -d -s "$NAME" "bash $WRAP >> $LOG 2>&1"
echo "LAUNCHED name=$NAME log=$LOG mark=$MARK"
