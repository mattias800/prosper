#!/bin/bash
cd "${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-ae005ba42de93cd40/prosper"
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1
gdb -batch -x tools/dbg/vostatus.gdb --args ./build-linux/boot_trace /root/PPSA17942-app0 > /root/vostatus.log 2>&1
echo "rc=$?"
sed -n "/GetOutputStatus a0/,/==== DONE/p" /root/vostatus.log | head -40
