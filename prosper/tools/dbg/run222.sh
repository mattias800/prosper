#!/bin/bash
# Run DOLL under gdb catching the #222 fault.
cd /mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-ae005ba42de93cd40/prosper
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
gdb -batch -x tools/dbg/catch222.gdb --args ./build-linux/boot_trace /root/PPSA17942-app0 > /root/catch222.log 2>&1
echo "gdb done, exit=$?"
grep -n "FATAL 222" /root/catch222.log | head -2
tail -120 /root/catch222.log
