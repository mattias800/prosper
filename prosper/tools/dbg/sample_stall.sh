#!/bin/bash
# Run DOLL; when GpuFlip count stops advancing for 30s, gdb-sample all threads (the RenderThread stall).
cd "${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-ae005ba42de93cd40/prosper"
PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1 \
  ./build-linux/boot_trace /root/PPSA17942-app0 > /root/scene3.log 2>&1 &
PID=$!
echo "boot_trace pid=$PID"
last=0; same=0
while kill -0 $PID 2>/dev/null; do
  sleep 10
  cur=$(grep -c GpuFlip /root/scene3.log 2>/dev/null)
  if [ "$cur" -gt 100 ] && [ "$cur" -eq "$last" ]; then
    same=$((same+1))
  else
    same=0
  fi
  last=$cur
  if [ "$same" -ge 3 ]; then break; fi
done
if ! kill -0 $PID 2>/dev/null; then echo "process died before stall detection"; exit 1; fi
echo "stall detected at flips=$last — sampling"
echo "=== thread cpu (utime+stime tid comm) ===" > /root/bt_samples.txt
for t in /proc/$PID/task/*; do
  tid=$(basename "$t")
  utime=$(awk '{ for(i=1;i<=NF;i++) if ($i ~ /\)$/) { print $(i+12)+$(i+13); exit } }' "$t/stat" 2>/dev/null)
  name=$(cat "$t/comm" 2>/dev/null)
  echo "$utime $tid $name"
done | sort -rn | head -16 >> /root/bt_samples.txt
for i in 1 2; do
  echo "=== SAMPLE $i ===" >> /root/bt_samples.txt
  gdb -p "$PID" -batch -ex "set pagination off" -ex "thread apply all bt 14" >> /root/bt_samples.txt 2>&1
  sleep 2
done
kill -9 "$PID"
echo SAMPLING-DONE
