#!/bin/bash
# Diagnostic: run DOLL to steady state, then sample all-thread backtraces with gdb.
# Usage: sample_bt.sh <sleep_seconds> <nsamples>
SLEEP=${1:-150}
N=${2:-5}
cd /mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-ae005ba42de93cd40/prosper
PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1 \
  ./build-linux/boot_trace /root/PPSA17942-app0 > /root/scene1.log 2>&1 &
PID=$!
echo "boot_trace pid=$PID"
sleep "$SLEEP"
echo "=== thread cpu usage (utime+stime, tid, comm) ===" > /root/bt_samples.txt
for t in /proc/$PID/task/*; do
  tid=$(basename "$t")
  read -r -a f < "$t/stat" 2>/dev/null || continue
  # comm may contain spaces; stat fields after comm shift. comm is field 2 in parens.
  utime=$(awk '{ for(i=1;i<=NF;i++) if ($i ~ /\)$/) { print $(i+12)+$(i+13); exit } }' "$t/stat" 2>/dev/null)
  name=$(cat "$t/comm" 2>/dev/null)
  echo "$utime $tid $name"
done | sort -rn | head -14 >> /root/bt_samples.txt
for i in $(seq 1 "$N"); do
  echo "=== SAMPLE $i ===" >> /root/bt_samples.txt
  gdb -p "$PID" -batch -ex "set pagination off" -ex "thread apply all bt 24" >> /root/bt_samples.txt 2>&1
  sleep 3
done
kill -9 "$PID" 2>/dev/null
echo "=== log tail ==="
tail -30 /root/scene1.log
