#!/bin/bash
# Run DOLL to steady state, then attach gdb and probe the #232 poll wall.
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-aa1d2a853141bdab4/prosper"
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
./build-linux/boot_trace /root/PPSA17942-app0 > /root/poll1.log 2>&1 &
PID=$!
echo "boot_trace pid=$PID"
last=0; same=0; elapsed=0
while kill -0 $PID 2>/dev/null; do
  sleep 10
  elapsed=$((elapsed+10))
  cur=$(grep -c GpuFlip /root/poll1.log 2>/dev/null)
  echo "t=$elapsed flips=$cur"
  if [ "$cur" -gt 100 ] && [ "$cur" -eq "$last" ]; then same=$((same+1)); else same=0; fi
  last=$cur
  if [ "$same" -ge 2 ]; then break; fi
  if [ "$elapsed" -ge 300 ] && [ "$cur" -gt 50 ]; then echo "timeout-steady"; break; fi
done
if ! kill -0 $PID 2>/dev/null; then echo "process died"; tail -30 /root/poll1.log; exit 1; fi
echo "steady at flips=$last — attaching gdb"
gdb -p $PID -batch -x "$WT/tools/dbg/poll_probe.gdb" > /root/poll_probe.txt 2>&1
kill -9 $PID 2>/dev/null
echo PROBE-DONE
wc -l /root/poll_probe.txt
