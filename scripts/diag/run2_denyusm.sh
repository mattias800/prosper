#!/bin/bash
# DOLL progression probe, run 2: NO renderer + PROSPER_DENY_SUBSTR=.usm (movie-denial A/B).
# If the front-end flow is wedged waiting on CRI movie playback that our missing sceVideodec2
# can't deliver, denying the .usm files should CHANGE progression (skip-movie path).
set -u
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-aa922b2bc59375888"
BT="$WT/prosper/build-linux/boot_trace"
DUMP=/root/PPSA17942-app0
DUR=${DUR:-420}
LOG=/root/doll_r2.log
rm -f "$LOG"

( timeout "$DUR" env PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1 \
      PROSPER_PROGRESS=5 PROSPER_PROGRESS_UNIMPL=1 PROSPER_DENY_SUBSTR=.usm "$BT" "$DUMP" 2>&1 | \
  gawk '{ if (!t0) t0 = systime(); printf "%4d %s\n", systime()-t0, $0; fflush() }' > "$LOG" ) &
wait

echo "=================== ANALYSIS (deny .usm) ==================="
wc -l "$LOG"
echo "--- [progress] heartbeats:"
grep -a '\[progress\]' "$LOG" | head -100
echo "--- DENIED lines:"
grep -a 'DENIED' "$LOG" | head -10
echo "--- APR reads per 60s bucket:"
grep -a 'read-submit\|read-direct' "$LOG" | awk '{ b = int($1/60); n[b]++ } END { for (i in n) printf "min %d: %d reads\n", i, n[i] }' | sort -t' ' -k2 -n
echo "--- file opens per 60s bucket:"
grep -a '\[file\] open' "$LOG" | awk '{ b = int($1/60); n[b]++ } END { for (i in n) printf "min %d: %d opens\n", i, n[i] }' | sort -t' ' -k2 -n
echo "--- last 15 file-layer lines:"
grep -a '\[file\]\|\[apr\]' "$LOG" | tail -15
echo "--- unimplemented (distinct, whole run):"
grep -a 'unimplemented:' "$LOG" | sed 's/^ *[0-9]* //' | sort | uniq -c | sort -rn | head -25
echo RUN2_DONE
