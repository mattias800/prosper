#!/bin/bash
# Issue #232: catch the stack-smash abort with a real core dump.
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-a36ca69a115e7c61a/prosper
cd "$WT" || exit 1
echo '/root/cores/core.%p' > /proc/sys/kernel/core_pattern
mkdir -p /root/cores; rm -f /root/cores/core.*
ulimit -c unlimited
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
for try in 1 2 3 4 5 6 7 8; do
  timeout 150 ./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232p.log 2>&1
  rc=$?
  echo "try=$try rc=$rc flips=$(grep -ac GpuFlip /root/d232p.log)"
  grep -a "WORKER-THREAD FAULT\|stack smashing" /root/d232p.log | head -2
  ls /root/cores/ 2>/dev/null | head -2
  if ls /root/cores/core.* >/dev/null 2>&1; then break; fi
  # a run that survives 180s+ is the steady loop; kill it and retry for a crash
done
CORE=$(ls -t /root/cores/core.* 2>/dev/null | head -1)
[ -n "$CORE" ] || { echo "NO CORE"; exit 1; }
echo "=== core: $CORE ==="
gdb "$WT/build-linux/boot_trace" "$CORE" -batch -ex "set pagination off" -ex "info threads" -ex "thread apply all bt 12" 2>/dev/null | grep -v "^\[New\|libthread_db" | head -120
echo RUN232P-DONE
