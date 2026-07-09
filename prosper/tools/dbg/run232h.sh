#!/bin/bash
# Issue #232: does the w1KFAHVqpaU submit-fold advance DOLL past the RenderThread fence wall?
# Run 360 s, track flips/submits over time (should NOT plateau at ~178 now), and check for draws.
SECS=${1:-360}
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-af9660f2dbcb26e7d/prosper
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
timeout "$SECS" ./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232h.log 2>&1 &
PID=$!
echo "pid=$PID"
el=0
while kill -0 $PID 2>/dev/null; do
  sleep 20; el=$((el+20))
  f=$(grep -c GpuFlip /root/d232h.log 2>/dev/null)
  s=$(grep -c "SubmitDcb\|SubmitDcbFinal" /root/d232h.log 2>/dev/null)
  v=$(grep -c "SubmitDcbFinal" /root/d232h.log 2>/dev/null)
  d=$(grep -c "kind=DrawIndex" /root/d232h.log 2>/dev/null)
  echo "t=$el flips=$f submits=$s finalSubmits=$v drawpkts=$d"
done
echo "=== DONE rc ==="
echo "flips=$(grep -c GpuFlip /root/d232h.log)"
echo "final-submits (w1KFAHVqpaU)=$(grep -c SubmitDcbFinal /root/d232h.log)"
echo "EOP writes to 0x1180f0 (the RT fence region)=$(grep -c 'EOP write \[0x1180f0' /root/d232h.log)"
echo "DrawIndex packets=$(grep -c 'kind=DrawIndex' /root/d232h.log)"
echo "DrawIndexAuto packets=$(grep -c 'kind=DrawIndexAuto' /root/d232h.log)"
echo "draws>0 lines:"; grep -oE 'draws: [1-9][0-9]*' /root/d232h.log | tail -3
echo "presented frames=$(grep -c presented /root/d232h.log)"
echo "faults=$(grep -c 'WORKER-THREAD FAULT\|RUN ENDED' /root/d232h.log)"
echo "=== w1k refusals ==="; grep -c "w1KFAHVqpaU: no PM4" /root/d232h.log
echo "=== tail ==="; tail -6 /root/d232h.log
echo RUN232H-DONE
