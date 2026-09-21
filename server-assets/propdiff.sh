#!/bin/bash
LOG=/tmp/p60-fsA-server.log
echo "=== top shape changes (new -> old) ==="
grep -a "prop diff" $LOG | sed -e 's/.*new_ne=//' -e 's/ new_data=.*//' | sort | uniq -c | sort -rn | head -14
echo "=== top node names ==="
grep -a "prop diff" $LOG | sed -e 's/.*node=//' -e 's/ op=.*//' | sort | uniq -c | sort -rn | head -14
