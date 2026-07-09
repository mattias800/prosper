#!/bin/bash
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-a36ca69a115e7c61a/prosper
CORE=$(ls -t /root/cores/core.* | head -1)
gdb "$WT/build-linux/boot_trace" "$CORE" -batch \
  -ex "set pagination off" \
  -ex "thread 1" -ex "bt 25" \
  -ex "thread 60" -ex "bt 25" \
  -ex "thread 1" -ex "x/32gx \$rsp" 2>/dev/null | grep -v "libthread_db\|^\[New"
