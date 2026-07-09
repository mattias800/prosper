#!/bin/bash
# unk232.sh — opcode histogram of kind=Unknown packets in /root/frame1.txt
grep -a "kind=Unknown" /root/frame1.txt | awk '{print $3, $5}' | sort | uniq -c | sort -rn | head -25
echo "=== samples ==="
grep -a "kind=Unknown" /root/frame1.txt | head -10
