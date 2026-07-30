#!/bin/bash
# Messenger render smoke for issue #232 (shared GPU pm4/command_processor changed). Must still reach
# real scene draws (vcount=1044) with 0 faults. Runs under a private binary name (other agents pkill
# boot_trace on this shared WSL). Long run: the scene draws only appear deep in.
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-ac66b5fec5918f7aa/prosper"
cp -f "$WT/build-linux/boot_trace" /root/btmsg
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 PROSPER_GFXLOG=1 \
  timeout 260 /root/btmsg "${PROSPER_REPO_ROOT:?set to your checkout root}/PPSA24651-app0" > /root/smokemsg.log 2>&1
echo "rc=$?"
echo "=== presented frames ==="; grep -ac "presented" /root/smokemsg.log
echo "=== draws (max) ==="; grep -ao "draws so far: [0-9]*" /root/smokemsg.log | awk '{print $4}' | sort -n | tail -2
echo "=== vcount (scene geometry) ==="; grep -ao "vcount=[0-9]*" /root/smokemsg.log | sort | uniq -c | sort -rn | head -5
echo "=== DrawIndexOffset (should be 0 for Messenger) ==="; grep -ac "kind=DrawIndexOffset" /root/smokemsg.log
echo "=== faults ==="; grep -ac "WORKER-THREAD FAULT\|SIGSEGV\|Corruption\|LowLevelFatal" /root/smokemsg.log
tail -3 /root/smokemsg.log
echo SMOKE-DONE
