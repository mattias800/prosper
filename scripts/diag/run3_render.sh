#!/bin/bash
# DOLL progression probe, run 3: WITH live renderer (llvmpipe) — quantify presented-frame
# throughput and whether render throttling changes guest progression. PADLOG shows input polling.
set -u
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-aa922b2bc59375888"
cp -f "$WT/prosper/build-linux/boot_trace" /root/bt_diag
BT=/root/bt_diag
DUMP=/root/PPSA17942-app0
DUR=${DUR:-180}
LOG=/root/doll_r3.log
SAMP=/root/doll_r3.samples
rm -f "$LOG" "$SAMP"
mkdir -p /root/doll_r3_frames && rm -f /root/doll_r3_frames/frame_*.bmp

( timeout "$DUR" env PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_RTT_PERTARGET=1 \
      PROSPER_FILELOG=1 PROSPER_PROGRESS=5 PROSPER_PROGRESS_UNIMPL=1 PROSPER_PADLOG=1 \
      PROSPER_FRAME_DIR=/root/doll_r3_frames PROSPER_DUMP_CONTENT=64 "$BT" "$DUMP" 2>&1 | \
  gawk '{ if (!t0) t0 = systime(); printf "%4d %s\n", systime()-t0, $0; fflush() }' > "$LOG" ) &

sleep 8
# find the real boot_trace PID (comm match, not the timeout wrapper)
PID=""
for p in $(pgrep -f "$DUMP"); do
    if [ "$(cat /proc/$p/comm 2>/dev/null)" = "bt_diag" ]; then PID=$p; fi
done
echo "boot pid: ${PID:-NONE}" >> "$SAMP"
T0=$(date +%s)
while [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; do
    NOW=$(( $(date +%s) - T0 ))
    IOL=$(tr '\n' ' ' < /proc/$PID/io 2>/dev/null)
    NT=$(ls /proc/$PID/task 2>/dev/null | wc -l)
    CPU=$(awk '{print $14+$15}' /proc/$PID/stat 2>/dev/null)
    echo "t=$NOW threads=$NT cpu_ticks=$CPU io: $IOL" >> "$SAMP"
    for t in /proc/$PID/task/*/stat; do
        awk '{printf "%s %s %s %s\n", $1, $2, $3, $14+$15}' "$t" 2>/dev/null
    done | sort -k4 -rn | head -5 | sed 's/^/    thr /' >> "$SAMP"
    sleep 30
done
wait

echo "=================== ANALYSIS (render) ==================="
wc -l "$LOG"
echo "--- [progress] heartbeats:"
grep -a '\[progress\]' "$LOG" | head -120
echo "--- pad lines:"
grep -a '\[pad\]' "$LOG" | head -20
echo "--- file opens per 60s bucket:"
grep -a '\[file\] open' "$LOG" | awk '{ b = int($1/60); n[b]++ } END { for (i in n) printf "min %d: %d opens\n", i, n[i] }' | sort -t' ' -k2 -n
echo "--- APR reads per 60s bucket:"
grep -a 'read-submit\|read-direct' "$LOG" | awk '{ b = int($1/60); n[b]++ } END { for (i in n) printf "min %d: %d reads\n", i, n[i] }' | sort -t' ' -k2 -n
echo "--- last 15 file-layer lines:"
grep -a '\[file\]\|\[apr\]' "$LOG" | tail -15
echo "--- samples:"
cat "$SAMP"
echo "--- log tail:"
tail -15 "$LOG"
echo RUN3_DONE
