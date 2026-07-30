#!/bin/bash
# Issue #232: boot with AMPRLOG, wait for the read stall, then locate + dump the dispatcher.
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-a36ca69a115e7c61a/prosper"
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1 PROSPER_AMPRLOG=1
PID=
for try in 1 2 3 4 5 6 7 8; do
  ./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232s.log 2>&1 &
  PID=$!
  echo "try=$try pid=$PID"
  sleep 60
  kill -0 $PID 2>/dev/null && break
  echo "DIED (try $try)"
  PID=
done
[ -n "$PID" ] || { echo "ALL TRIES DIED"; exit 1; }
# wait for the stall: reads frozen for 40 s
prev=-1
for i in 1 2 3 4 5 6 7 8 9 10; do
  cur=$(grep -ac "read-submit id" /root/d232s.log)
  echo "reads=$cur"
  if [ "$cur" = "$prev" ] && [ "$cur" -gt 1000 ]; then echo STALLED; break; fi
  prev=$cur
  sleep 40
  kill -0 $PID 2>/dev/null || { echo DIED-WAITING; exit 1; }
done
# the stalled batch: last H896 bind's a0 is the cb; tag a3 = the batch object
grep -a "H896Pt-yB4I(CbSetEqueue)" /root/d232s.log | tail -1
LASTTAG=$(grep -a "H896Pt-yB4I(CbSetEqueue)" /root/d232s.log | tail -1 | grep -o "a3=0x[0-9a-f]*" | cut -d= -f2)
echo "$LASTTAG" > /root/batch.txt
echo "batch=$LASTTAG"
timeout 240 gdb -p $PID -batch -x "$WT/tools/dbg/scan_disp.py" 2>&1 | grep -v "^\[New\|libthread_db\|debuginfod\|Debuginfod\|warning:\|answered N" | tail -70
kill -9 $PID 2>/dev/null
echo RUN232S-DONE
