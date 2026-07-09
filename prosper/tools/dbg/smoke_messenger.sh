#!/bin/bash
# Messenger render smoke: must render frames (presented) with the live renderer.
cd /mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-ae005ba42de93cd40/prosper
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 PROSPER_GFXLOG=1 \
  timeout 240 ./build-linux/boot_trace /mnt/c/Users/matti/repos/ps5ys/PPSA24651-app0 > /root/smoke.log 2>&1
echo "rc=$?"
echo "=== presented frames ==="
grep -c "presented" /root/smoke.log
grep "presented" /root/smoke.log | tail -2
echo "=== draws ==="
grep -E "draws so far: [1-9]" /root/smoke.log | tail -2
echo "=== faults ==="
grep -c "WORKER-THREAD FAULT\|RUN ENDED" /root/smoke.log
tail -3 /root/smoke.log
