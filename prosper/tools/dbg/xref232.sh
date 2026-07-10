#!/bin/bash
# xref232.sh <hexaddr>... — find refs in /root/dis232.txt (objdump comment "# 0x...")
for a in "$@"; do
  echo "=== refs to 0x$a ==="
  grep -n "# 0x$a\b" /root/dis232.txt | head -12
done
