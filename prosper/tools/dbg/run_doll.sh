#!/bin/bash
# Run DOLL for N seconds with full logging, then summarize draw evidence.
SECS=${1:-300}
LOG=${2:-/root/scene2.log}
cd "${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-ae005ba42de93cd40/prosper"
PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1 \
  timeout "$SECS" ./build-linux/boot_trace /root/PPSA17942-app0 > "$LOG" 2>&1
echo "run done rc=$?"
echo "=== flips ==="; grep -c "GpuFlip" "$LOG"
echo "=== last flip ==="; grep "GpuFlip" "$LOG" | tail -2
echo "=== submits ==="; grep -c "SubmitDcb" "$LOG"
echo "=== draws (SetCxRegistersIndirect / DrawIndex) ==="
grep -cE "kind=DrawIndex" "$LOG"
grep -E "draws so far: [1-9]" "$LOG" | tail -5
echo "=== ctxinit calls ==="; grep -c "CtxInit" "$LOG"
grep "CtxInit" "$LOG" | head -5
grep "CtxInit" "$LOG" | tail -3
echo "=== worker faults ==="; grep -c "WORKER-THREAD FAULT" "$LOG"
grep -A2 "WORKER-THREAD FAULT" "$LOG" | head -6
echo "=== presented ==="; grep -c "presented" "$LOG"
grep "presented" "$LOG" | tail -3
echo "=== tail ==="; tail -5 "$LOG"
