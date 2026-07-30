#!/bin/bash
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-aa1d2a853141bdab4/prosper"
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
./build-linux/boot_trace /root/PPSA17942-app0 > /root/sbt2.log 2>&1 &
PID=$!
echo "pid=$PID"
last=0; same=0; el=0
while kill -0 $PID 2>/dev/null; do
  sleep 8; el=$((el+8))
  cur=$(grep -c GpuFlip /root/sbt2.log 2>/dev/null)
  echo "t=$el flips=$cur"
  if [ "$cur" -gt 100 ] && [ "$cur" -eq "$last" ]; then same=$((same+1)); else same=0; fi
  last=$cur
  if [ "$same" -ge 2 ]; then echo "FROZEN"; break; fi
  if [ "$el" -ge 300 ]; then echo giveup; break; fi
done
kill -0 $PID 2>/dev/null || { echo died; exit 1; }
gdb -p $PID -batch \
  -ex "set pagination off" \
  -ex "handle all nostop pass noprint" \
  -ex "thread apply all bt 6" \
  > /root/sbt2.txt 2>&1
kill -9 $PID 2>/dev/null
echo SBT2-DONE
# Reduce: for each thread show its innermost eboot frame (frame #0/#1)
awk '/^Thread /{t=$0} /eboot|0x000000040/{if(t){print t; t=""} print}' /root/sbt2.txt | head -200 > /root/sbt2_reduced.txt
wc -l /root/sbt2.txt
