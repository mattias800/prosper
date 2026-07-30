#!/bin/bash
# DOLL progression probe, run 8: pad-connected matrix. 8a = PAD_PRESS only; 8b = PAD_PRESS+NETCTL_CB.
# Usage: run8_matrix.sh a|b
set -u
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-aa922b2bc59375888"
V=$1
cp -f "$WT/prosper/build-linux/boot_trace" /root/bt_diag_$V
DUMP=/root/PPSA17942-app0
DUR=${DUR:-240}
LOG=/root/doll_r8$V.log
rm -f "$LOG"

EXTRA=""
if [ "$V" = "b" ]; then EXTRA="PROSPER_NETCTL_CB=1"; fi

( timeout "$DUR" env PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_FILELOG=1 PROSPER_PADLOG=1 \
      PROSPER_PROGRESS=5 PROSPER_PAD_PRESS=1 $EXTRA \
      /root/bt_diag_$V "$DUMP" 2>&1 | \
  gawk '{ if (!t0) t0 = systime(); printf "%4d %s\n", systime()-t0, $0; fflush() }' > "$LOG" ) &
wait

echo "=================== ANALYSIS (8$V: PAD_PRESS ${EXTRA:-only}) ==================="
echo "--- file opens per 30s bucket:"
grep -a '\[file\] open' "$LOG" | awk '{ b = int($1/30); n[b]++ } END { for (i in n) printf "t%03d: %d opens\n", i*30, n[i] }' | sort
echo "--- APR reads per 30s bucket:"
grep -a 'read-submit\|read-direct' "$LOG" | awk '{ b = int($1/30); n[b]++ } END { for (i in n) printf "t%03d: %d reads\n", i*30, n[i] }' | sort
echo "--- file activity after t=20 (first 15):"
grep -a '\[file\]' "$LOG" | awk '$1 > 20' | head -15
echo "--- usm:"
grep -a 'usm' "$LOG" | tail -5
echo "--- heartbeats every 6th:"
grep -a '\[progress\]' "$LOG" | awk 'NR % 6 == 1' | tail -10
echo "--- log tail:"
tail -5 "$LOG"
echo RUN8${V}_DONE
