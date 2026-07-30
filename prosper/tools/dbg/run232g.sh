#!/bin/bash
# Issue #232: attach EARLY (flips>90, before the ~174 warmup submit) and arm the w1KFAHVqpaU probe.
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-af9660f2dbcb26e7d/prosper"
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232g.log 2>&1 &
PID=$!
echo "pid=$PID"
el=0
while kill -0 $PID 2>/dev/null; do
  sleep 5; el=$((el+5))
  cur=$(grep -c GpuFlip /root/d232g.log 2>/dev/null)
  if [ "$cur" -ge 90 ]; then echo "attach at flips=$cur t=$el"; break; fi
  if [ "$el" -ge 180 ]; then echo "never reached 90 flips"; break; fi
done
kill -0 $PID 2>/dev/null || { echo DIED; tail -20 /root/d232g.log; exit 1; }
timeout 150 gdb -p $PID -batch -x "$WT/tools/dbg/w1k232.py" > /root/w1k232.txt 2>&1
grep -E "CALLSITE|rsp\+|buffer_entry|base\+|w1k probe" /root/w1k232.txt
kill -9 $PID 2>/dev/null
echo RUN232G-DONE
