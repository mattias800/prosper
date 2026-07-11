#!/bin/bash
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-a48ff9e03b626cc45/prosper
cd "$WT" || exit 1
pkill -9 boot_trace 2>/dev/null; sleep 1
L=/root/msg_reg.log
timeout 130 env PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
  ./build-linux/boot_trace /mnt/c/Users/matti/repos/ps5ys/PPSA24651-app0 > "$L" 2>&1
echo "rc=$?"
echo "lines=$(wc -l < $L)"
echo "vcount: $(grep -oE 'vcount=[0-9]+' $L | tail -1)"
echo "frames: $(grep -oiE 'frame[s]?[ =:]+[0-9]+' $L | tail -1)"
echo "faults: $(grep -icE 'WORKER-THREAD FAULT|SIGSEGV|LowLevelFatal|smashing' $L)"
echo "forge/mb3 leakage (should be 0): $(grep -cE 'REL1-FORGE|FORGE-STOMP|mb3watch' $L)"
tail -4 $L
