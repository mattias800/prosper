#!/bin/bash
# frame232.sh — dump one frame's packet-kind histogram + CS shader addrs from /root/s6.log
A=${1:-83516637}
B=${2:-83520912}
sed -n "${A},${B}p" /root/s6.log > /root/frame1.txt
echo "=== packet kinds in one frame ==="
grep -ao "kind=[A-Za-z]*" /root/frame1.txt | sort | uniq -c | sort -rn
echo "=== SetShRegDirect samples (CS shader addr regs) ==="
grep -a "SetShRegDirect" /root/frame1.txt | head -8
echo "=== dispatch sizes ==="
grep -a "kind=DispatchDirect" /root/frame1.txt | awk '{print $NF}' | sort | uniq -c | sort -rn | head
