#!/bin/bash
# Issue #232 (post-#236): characterize DOLL's steady state — is the GameThread advancing or parked?
# Runs DOLL, samples flip/submit counts every 15 s, then at STEADY_T attaches gdb and dumps every
# thread: host frame + guest return addresses found on the stack. Repeats the dump 3x (interrupt
# sampling) to see whether threads MOVE. Everything in ONE wsl call (WSL /tmp is wiped between calls).
SECS=${1:-300}
STEADY_T=${2:-180}
WT="${PROSPER_REPO_ROOT:?set to your checkout root}/.claude/worktrees/agent-af9660f2dbcb26e7d/prosper"
cd "$WT" || exit 1
export PROSPER_GUEST_FS=1 PROSPER_NULL_PAGE=1 PROSPER_RENDER=1 PROSPER_GFXLOG=1 PROSPER_FILELOG=1 PROSPER_EVLOG=1
./build-linux/boot_trace /root/PPSA17942-app0 > /root/d232.log 2>&1 &
PID=$!
echo "pid=$PID"
el=0
while kill -0 $PID 2>/dev/null && [ $el -lt $STEADY_T ]; do
  sleep 15; el=$((el+15))
  f=$(grep -c GpuFlip /root/d232.log 2>/dev/null)
  s=$(grep -c "SubmitDcb #" /root/d232.log 2>/dev/null)
  d=$(grep -c "kind=DrawIndex" /root/d232.log 2>/dev/null)
  echo "t=$el flips=$f submits=$s drawpkts=$d"
done
kill -0 $PID 2>/dev/null || { echo "DIED before steady state"; tail -30 /root/d232.log; exit 1; }
cat > /root/gw232.gdb << "GEOF"
set pagination off
handle all nostop pass noprint
define gs
  printf "  --host frames--\n"
  bt 8
  printf "  --guest RAs on stack--\n"
  set $p = $sp - 128
  set $e = $sp + 6144
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
thread apply all gs
GEOF
for i in 1 2 3; do
  echo "=== SAMPLE $i (t≈$el) ==="
  gdb -p $PID -batch -x /root/gw232.gdb > /root/gw232_$i.txt 2>&1
  echo "sample $i done, lines: $(wc -l < /root/gw232_$i.txt)"
  sleep 10; el=$((el+10))
  f=$(grep -c GpuFlip /root/d232.log 2>/dev/null)
  s=$(grep -c "SubmitDcb #" /root/d232.log 2>/dev/null)
  echo "t=$el flips=$f submits=$s"
done
kill -9 $PID 2>/dev/null
echo "=== summary ==="
echo "flips=$(grep -c GpuFlip /root/d232.log)"
echo "submits=$(grep -c 'SubmitDcb #' /root/d232.log)"
echo "draw-pkts=$(grep -c 'kind=DrawIndex' /root/d232.log)"
echo "cx-indirect=$(grep -c 'DcbSetCxRegistersIndirect' /root/d232.log)"
echo "dispatches: $(grep -o 'dispatches total: [0-9]*' /root/d232.log | tail -1)"
echo "unimpl NIDs seen:"; grep -o 'unimplemented[^ ]* NID [^ ]*' /root/d232.log | sort | uniq -c | sort -rn | head -20
echo "=== last 15 log lines ==="; tail -15 /root/d232.log
echo RUN232-DONE
