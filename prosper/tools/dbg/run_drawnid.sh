#!/bin/bash
cd "${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-ae005ba42de93cd40/prosper"
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1
gdb -batch -x tools/dbg/drawnid.gdb --args ./build-linux/boot_trace /root/PPSA17942-app0 > /root/drawnid.log 2>&1
echo "gdb done rc=$?"
grep -n "HIT DONE\|hit #" /root/drawnid.log | head
sed -n "/hit #5 /,/HIT DONE/p" /root/drawnid.log | head -80
