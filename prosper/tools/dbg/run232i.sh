#!/bin/bash
# Issue #232 (post-fence): what is the GameThread doing in steady state?
# Runs DOLL to STEADY_T, then gdb-samples every thread (names now adopted) 3x,
# focusing on GameThread / RenderThread / RHIThread guest RAs.
SECS=${1:-420}
STEADY_T=${2:-180}
WT=/mnt/c/Users/matti/repos/ps5ys/.claude/worktrees/agent-a36ca69a115e7c61a/prosper
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232i.log 2>&1 &
PID=$!
echo "pid=$PID"
el=0
while kill -0 $PID 2>/dev/null && [ $el -lt $STEADY_T ]; do
  sleep 15; el=$((el+15))
  f=$(grep -c GpuFlip /root/d232i.log 2>/dev/null)
  s=$(grep -c "SubmitDcb" /root/d232i.log 2>/dev/null)
  d=$(grep -c "kind=DrawIndex" /root/d232i.log 2>/dev/null)
  c=$(grep -c "DcbSetCxRegistersIndirect" /root/d232i.log 2>/dev/null)
  echo "t=$el flips=$f submits=$s drawpkts=$d cxind=$c"
done
kill -0 $PID 2>/dev/null || { echo "DIED before steady state"; tail -40 /root/d232i.log; exit 1; }
cat > /root/gw232i.gdb << "GEOF"
set pagination off
handle all nostop pass noprint
define gs
  printf "  --host frames--\n"
  bt 10
  printf "  --guest RAs on stack--\n"
  set $p = $sp - 128
  set $e = $sp + 8192
  while $p < $e
    set $v = *(unsigned long long*)$p
    if ($v >= 0x400000000 && $v < 0x406700000)
      printf "    eboot+0x%llx\n", $v - 0x400000000
    end
    if ($v >= 0x600000000 && $v < 0x610000000)
      printf "    prx+0x%llx\n", $v - 0x600000000
    end
    set $p = $p + 8
  end
end
info threads
thread apply all gs
GEOF
for i in 1 2 3; do
  echo "=== SAMPLE $i (t=$el) ==="
  gdb -p $PID -batch -x /root/gw232i.gdb > /root/gw232i_$i.txt 2>&1
  echo "sample $i lines: $(wc -l < /root/gw232i_$i.txt)"
  sleep 12; el=$((el+12))
  f=$(grep -c GpuFlip /root/d232i.log 2>/dev/null)
  echo "t=$el flips=$f"
done
kill -9 $PID 2>/dev/null
echo "=== summary ==="
echo "flips=$(grep -c GpuFlip /root/d232i.log)"
echo "draw-pkts=$(grep -c 'kind=DrawIndex' /root/d232i.log)"
echo "cx-indirect=$(grep -c 'DcbSetCxRegistersIndirect' /root/d232i.log)"
echo "worker faults=$(grep -c 'WORKER-THREAD FAULT' /root/d232i.log)"
grep 'WORKER-THREAD FAULT' /root/d232i.log | head -3
echo "=== GameThread sample 1 ==="
awk '/Thread .*GameThread/{f=1} f&&/^Thread [0-9]+/{if(++n>1)f=0} f' /root/gw232i_1.txt | head -60
echo RUN232I-DONE
