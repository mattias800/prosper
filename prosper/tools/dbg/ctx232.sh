#!/bin/bash
# Show context around the fence-label build for the polled label, and check which buffer/submit it belongs to.
L=/root/d232f.log
n=$(grep -n "ReleaseMem action=0x14 dst=0x1 addr=0x1180f06760" "$L" | head -1 | cut -d: -f1)
echo "line=$n of $(wc -l < "$L")"
sed -n "$((n-60)),$((n+20))p" "$L" | grep -v "WAIT.empty\|delivered\|WaitEqueue eq"
echo "=== how many 0x1180f0 fence builds, and are any folded? ==="
grep -c "ReleaseMem action=0x14 dst=0x1 addr=0x1180f0" "$L"
grep -c "EOP write \[0x1180f0" "$L"
echo "=== last SubmitDcb line number vs fence-build line ==="
grep -n "SubmitDcb #" "$L" | tail -3
echo "=== WriteData execution to 0x1180f0? ==="
grep -c "WriteData \[0x1180f0" "$L"
