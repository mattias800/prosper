#!/bin/bash
# DOLL progression probe, run 7: THE DECISIVE EXPERIMENT — deliver the NetCtl DISCONNECTED
# callback (PROSPER_NETCTL_CB=1) and watch whether the boot flow advances (new file IO after the
# t=6s wall, pad open, sms_opening movie open, new NIDs).
set -u
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-aa922b2bc59375888"
cp -f "$WT/prosper/build-linux/boot_trace" /root/bt_diag
DUMP=/root/PPSA17942-app0
DUR=${DUR:-300}
LOG=/root/doll_r7.log
rm -f "$LOG"

( timeout "$DUR" env PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_FILELOG=1 PROSPER_PADLOG=1 \
      PROSPER_PROGRESS=5 PROSPER_PROGRESS_UNIMPL=1 PROSPER_NETCTL_CB=1 PROSPER_SVCLOG=1 \
      /root/bt_diag "$DUMP" 2>&1 | \
  gawk '{ if (!t0) t0 = systime(); printf "%4d %s\n", systime()-t0, $0; fflush() }' > "$LOG" ) &
wait

echo "=================== ANALYSIS (NetCtl delivery) ==================="
wc -l "$LOG"
echo "--- DELIVERED line:"
grep -a 'DELIVERED' "$LOG"
echo "--- heartbeats (every 6th):"
grep -a '\[progress\]' "$LOG" | awk 'NR % 6 == 1'
echo "--- file opens per 30s bucket:"
grep -a '\[file\] open' "$LOG" | awk '{ b = int($1/30); n[b]++ } END { for (i in n) printf "t%03d: %d opens\n", i*30, n[i] }' | sort
echo "--- APR reads per 30s bucket:"
grep -a 'read-submit\|read-direct' "$LOG" | awk '{ b = int($1/30); n[b]++ } END { for (i in n) printf "t%03d: %d reads\n", i*30, n[i] }' | sort
echo "--- pad lines:"
grep -a '\[pad\]' "$LOG" | head -10
echo "--- usm/movie activity:"
grep -a 'usm' "$LOG" | tail -8
echo "--- file activity after t=20:"
grep -a '\[file\] open' "$LOG" | awk '$1 > 20' | head -20
echo "--- last unimpl dump:"
grep -a 'x  lib' "$LOG" | tail -35
echo "--- svc lines (last 25):"
grep -a '\[svc\]' "$LOG" | tail -25
echo RUN7_DONE
