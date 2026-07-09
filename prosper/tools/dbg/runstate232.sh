#!/bin/bash
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-a2d51084866d8d045/prosper
PID=$(cat /root/pid232.txt)
kill -0 $PID 2>/dev/null || { echo "DEAD pid=$PID"; exit 1; }
echo "probing pid=$PID"
timeout 150 gdb -p $PID -batch -x "$WT/tools/dbg/state232.py" 2>/dev/null \
  | grep -vE "New LWP|Thread debugging|libthread|warning|Download|debuginfod|answered|No such file"
echo STATE232-DONE
