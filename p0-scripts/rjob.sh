#!/bin/bash
# rjob.sh <name> <command string>
#
# Launch <command string> in a tmux session named <name> and return IMMEDIATELY.
# When the command exits the wrapper appends "JOB_DONE_<name> rc=<rc>" to
# /root/llm/test/<name>.log, which is the terminator rwait.sh blocks on.
#
#   rjob.sh build "bash /tmp/p3-build.sh"
#   rwait.sh build 1200            # 10 s sleep loop until the marker shows
#
# LOG=/path overrides the log file. Never run the payload through tmux send-keys
# quoting: it is written to /tmp/rjob-<name>.sh first.
set -u
NAME=${1:?usage: rjob.sh <name> <command string>}
shift
CMD="$*"
LOG=${LOG:-/root/llm/test/$NAME.log}
MARK="JOB_DONE_$NAME"
WRAP=/tmp/rjob-$NAME.sh

mkdir -p "$(dirname "$LOG")"
: > "$LOG"
tmux kill-session -t "$NAME" 2>/dev/null
printf '#!/bin/bash\n%s\nrc=$?\necho "%s rc=$rc" >> %s\n' "$CMD" "$MARK" "$LOG" > "$WRAP"
tmux new-session -d -s "$NAME" "bash $WRAP > $LOG 2>&1"
echo "LAUNCHED name=$NAME log=$LOG mark=$MARK"
