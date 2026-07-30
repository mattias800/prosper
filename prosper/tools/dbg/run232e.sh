#!/bin/bash
# Issue #232: boot DOLL (with STUBDUMP so import indices are mappable), reach the plateau,
# then count unimplemented-import calls per thread for 60 s.
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-af9660f2dbcb26e7d/prosper"
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1 PROSPER_STUBDUMP=1
./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232e.log 2>&1 &
PID=$!
echo "pid=$PID"
last=0; same=0; el=0
while kill -0 $PID 2>/dev/null; do
  sleep 10; el=$((el+10))
  cur=$(grep -c GpuFlip /root/d232e.log 2>/dev/null)
  echo "t=$el flips=$cur"
  if [ "$cur" -gt 100 ] && [ "$cur" -eq "$last" ]; then same=$((same+1)); else same=0; fi
  last=$cur
  if [ "$same" -ge 2 ]; then echo PLATEAU; break; fi
  if [ "$el" -ge 240 ]; then echo giveup; break; fi
done
kill -0 $PID 2>/dev/null || { echo DIED; tail -20 /root/d232e.log; exit 1; }
gdb -p $PID -batch -x "$WT/tools/dbg/unimpl232.py" > /root/unimpl232.txt 2>&1
grep "UNIMPL\|unimpl trace done" /root/unimpl232.txt
echo "=== map indices to NIDs (stubdump) ==="
grep "UNIMPL idx=" /root/unimpl232.txt | grep -o "idx=[0-9]*" | sort -u | while read x; do
  i=${x#idx=}
  grep -m1 "^\[stub\] #$i " /root/d232e.log
done
kill -9 $PID 2>/dev/null
echo RUN232E-DONE
