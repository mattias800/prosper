#!/bin/bash
# Messenger render smoke (this worktree): must reach 1000+ frames with 0 faults.
cd /mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-a36ca69a115e7c61a/prosper || exit 1
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 PROSPER_GFXLOG=1 \
  timeout 240 ./build-linux/boot_trace /mnt/c/Users/matti/repos/ps5ys/PPSA24651-app0 > /root/smoke2.log 2>&1
echo "rc=$?"
echo "flips=$(grep -ac GpuFlip /root/smoke2.log)"
echo "presented=$(grep -ac presented /root/smoke2.log)"
grep -a "presented" /root/smoke2.log | tail -2 | cut -c1-140
echo "faults=$(grep -ac 'WORKER-THREAD FAULT\|RUN ENDED' /root/smoke2.log)"
tail -3 /root/smoke2.log | cut -c1-140
echo SMOKE2-DONE
