#!/bin/bash
# DOLL progression probe, run 1: NO renderer (isolates game-logic progression from llvmpipe cost).
# Boots for DUR seconds with file/event logging + the PROSPER_PROGRESS heartbeat, samples /proc
# every 20 s, then prints a bucketed progression analysis. Run inside WSL Ubuntu-24.04 as root.
set -u
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-aa922b2bc59375888"
BT="$WT/prosper/build-linux/boot_trace"
DUMP=/root/PPSA17942-app0
DUR=${DUR:-420}
LOG=/root/doll_r1.log
SAMP=/root/doll_r1.samples

rm -f "$LOG" "$SAMP"

# Boot with wall-clock timestamps on every line.
( timeout "$DUR" env PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1 \
      PROSPER_PROGRESS=5 "$BT" "$DUMP" 2>&1 | \
  gawk '{ if (!t0) t0 = systime(); printf "%4d %s\n", systime()-t0, $0; fflush() }' > "$LOG" ) &
RUNNER=$!

# Sampler: /proc io + cpu + thread count every 20 s.
sleep 5
PID=$(pgrep -f "boot_trace $DUMP" | head -1)
echo "boot pid: ${PID:-NONE}" >> "$SAMP"
T0=$(date +%s)
while kill -0 "${PID:-0}" 2>/dev/null; do
    NOW=$(( $(date +%s) - T0 ))
    RB=$(grep read_bytes /proc/$PID/io 2>/dev/null | awk '{print $2}')
    RC=$(grep rchar /proc/$PID/io 2>/dev/null | head -1 | awk '{print $2}')
    NT=$(ls /proc/$PID/task 2>/dev/null | wc -l)
    CPU=$(awk '{print $14+$15}' /proc/$PID/stat 2>/dev/null)
    echo "t=$NOW rchar=$RC read_bytes=$RB threads=$NT cpu_ticks=$CPU" >> "$SAMP"
    # top-3 busiest threads by recent utime (name + state)
    for t in /proc/$PID/task/*/stat; do
        awk '{printf "%s %s %s %s\n", $1, $2, $3, $14+$15}' "$t" 2>/dev/null
    done | sort -k4 -rn | head -4 | sed 's/^/    thr /' >> "$SAMP"
    sleep 20
done
wait $RUNNER

echo "=================== ANALYSIS ==================="
echo "--- log size:"; wc -l "$LOG"
echo "--- [progress] heartbeat (all lines):"
grep -a '\[progress\]' "$LOG" | head -100
echo "--- APR reads per 60s bucket:"
grep -a 'read-submit\|read-direct' "$LOG" | awk '{ b = int($1/60); n[b]++ } END { for (i in n) printf "min %d: %d reads\n", i, n[i] }' | sort -t' ' -k2 -n
echo "--- file opens per 60s bucket:"
grep -a '\[file\] open' "$LOG" | awk '{ b = int($1/60); n[b]++ } END { for (i in n) printf "min %d: %d opens\n", i, n[i] }' | sort -t' ' -k2 -n
echo "--- GpuFlip per 60s bucket:"
grep -a 'GpuFlip' "$LOG" | awk '{ b = int($1/60); n[b]++ } END { for (i in n) printf "min %d: %d flips\n", i, n[i] }' | sort -t' ' -k2 -n
echo "--- last 15 file-layer lines:"
grep -a '\[file\]\|\[apr\]' "$LOG" | tail -15
echo "--- last 12 distinct non-flip/non-wait event lines:"
grep -a '\[ev\]' "$LOG" | grep -av 'GpuFlip\|WaitEqueue\|delivered\|IsFlipPending\|WAIT.empty' | tail -12
echo "--- unimplemented NIDs called in LAST 120s of run:"
LAST=$(tail -1 "$LOG" | awk '{print $1}')
grep -a 'unimplemented\|unimpl' "$LOG" | awk -v c="$LAST" '$1 > c-120' | sed 's/^ *[0-9]* //' | sort | uniq -c | sort -rn | head -20
echo "--- samples:"
cat "$SAMP"
echo "--- log tail:"
tail -20 "$LOG"
echo RUN1_DONE
