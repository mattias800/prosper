#!/bin/bash
# Messenger render smoke (this worktree): must render frames (presented) with the live renderer.
cd "${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-a2d51084866d8d045/prosper" || exit 1
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 PROSPER_GFXLOG=1 \
  timeout 240 ./build-linux/boot_trace "${PROSPER_REPO_ROOT:?set to your checkout root}/PPSA24651-app0" > /root/smoke3.log 2>&1
echo "rc=$?"
echo "=== presented frames ==="
grep -ac "presented" /root/smoke3.log
grep -a "presented" /root/smoke3.log | tail -2
echo "=== draws ==="
grep -aE "draws so far: [1-9]" /root/smoke3.log | tail -2
echo "=== faults ==="
grep -ac "WORKER-THREAD FAULT\|RUN ENDED" /root/smoke3.log
tail -3 /root/smoke3.log
