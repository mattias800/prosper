#!/bin/bash
# attach probe232w to the live stalled DOLL (pid in /root/pid232.txt), keep target alive
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-a9e9a0302ab3fc359/prosper
PID=$(cat /root/pid232.txt)
kill -0 "$PID" || { echo TARGET-DEAD; exit 1; }
timeout 240 gdb -p "$PID" -batch -x "$WT/tools/dbg/$1" 2>&1 \
  | grep -v "^\[New\|libthread_db\|debuginfod\|Debuginfod\|warning:\|^\[Thread\|^Downloading\|^Reading\|futex-internal\|clock_nanosleep\|^[0-9]*\Win \|^#"
echo PROBE-DONE
