#!/bin/bash
cd /mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-ae005ba42de93cd40/prosper
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1
gdb -batch -x tools/dbg/drawnid2.gdb --args ./build-linux/boot_trace /root/PPSA17942-app0 > /root/drawnid2.log 2>&1
echo "gdb done rc=$?"
sed -n "/hit #3 /,/HIT DONE/p" /root/drawnid2.log | head -70
sed -n "/hit #40 /,/HIT DONE/p" /root/drawnid2.log | head -70
