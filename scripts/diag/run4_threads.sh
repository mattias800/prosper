#!/bin/bash
# DOLL progression probe, run 4: boot no-render to steady state, then gdb-attach twice (60 s apart)
# and dump every thread's name/pc/guest-RAs — is anything blocked, and does the picture CHANGE?
set -u
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-aa922b2bc59375888
BT="$WT/prosper/build-linux/boot_trace"
DUMP=/root/PPSA17942-app0
LOG=/root/doll_r4.log
rm -f "$LOG"

env PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_PROGRESS=5 "$BT" "$DUMP" > "$LOG" 2>&1 &
sleep 3
PID=""
for p in $(pgrep -f "$DUMP"); do
    if [ "$(cat /proc/$p/comm 2>/dev/null)" = "boot_trace" ]; then PID=$p; fi
done
echo "boot pid: ${PID:-NONE}"
[ -z "$PID" ] && exit 1

sleep 180
echo "=== THREAD DUMP 1 (t=180s) ==="
gdb -p "$PID" -batch -x "$WT/prosper/tools/dbg/thr232.py" 2>/dev/null | grep -v "^\[" | head -120
sleep 60
echo "=== THREAD DUMP 2 (t=240s) ==="
gdb -p "$PID" -batch -x "$WT/prosper/tools/dbg/thr232.py" 2>/dev/null | grep -v "^\[" | head -120
echo "=== heartbeat tail ==="
grep -a '\[progress\]' "$LOG" | tail -8
kill -9 "$PID" 2>/dev/null
echo RUN4_DONE
