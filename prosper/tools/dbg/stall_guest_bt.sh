#!/bin/bash
# Run DOLL; at flip stall, dump raw stacks of ALL threads and extract guest (0x4xxxxxxxx) RAs.
cd /mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-ae005ba42de93cd40/prosper
PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1 \
  ./build-linux/boot_trace /root/PPSA17942-app0 > /root/scene4.log 2>&1 &
PID=$!
echo "boot_trace pid=$PID"
last=0; same=0
while kill -0 $PID 2>/dev/null; do
  sleep 10
  cur=$(grep -c GpuFlip /root/scene4.log 2>/dev/null)
  if [ "$cur" -gt 100 ] && [ "$cur" -eq "$last" ]; then same=$((same+1)); else same=0; fi
  last=$cur
  if [ "$same" -ge 3 ]; then break; fi
done
if ! kill -0 $PID 2>/dev/null; then echo "process died before stall"; exit 1; fi
echo "stall at flips=$last — dumping guest stacks"
cat > /tmp/gbt.gdb << "GEOF"
set pagination off
define gstack
  printf "===== THREAD-STACK %d =====\n", $arg0
  thread $arg0
  set $p = $sp
  set $e = $sp + 4096
  while $p < $e
    set $v = *(unsigned long long*)$p
    if ($v >= 0x400000000 && $v < 0x40a000000) || ($v >= 0x600000000 && $v < 0x600100000)
      printf "sp+0x%04x: 0x%llx\n", (unsigned int)($p - $sp), $v
    end
    set $p = $p + 8
  end
end
GEOF
NT=$(ls /proc/$PID/task | wc -l)
echo "threads: $NT"
{
  echo "set pagination off"
  echo "source /tmp/gbt.gdb"
  i=1
  while [ $i -le $NT ]; do
    echo "gstack $i"
    i=$((i+1))
  done
} > /tmp/gbt_all.gdb
gdb -p $PID -batch -x /tmp/gbt_all.gdb > /root/guest_stacks.txt 2>&1
kill -9 $PID
echo GSTACK-DONE
wc -l /root/guest_stacks.txt
