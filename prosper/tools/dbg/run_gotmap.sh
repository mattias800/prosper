#!/bin/bash
cd /mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-ae005ba42de93cd40/prosper
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_STUBDUMP=1
gdb -batch -x tools/dbg/gotmap.gdb --args ./build-linux/boot_trace /root/PPSA17942-app0 > /root/gotmap.log 2>&1
echo "rc=$?"
grep -E "GOT\[" /root/gotmap.log
# map the stub addresses to NIDs via the stub dump
for g in $(grep -oE "GOT\[0x(a9|aa)\]=0x[0-9a-f]+" /root/gotmap.log | cut -d= -f2); do
  off=$(( g - 0x600000000 ))
  printf "stub offset for %s = +0x%x\n" "$g" "$off"
  grep -iE "\+0x0*$(printf %x $off)\b" /root/gotmap.log | head -2
done
