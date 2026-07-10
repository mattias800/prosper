#!/bin/bash
# call232.sh <hexaddr>... — find "call 0x..." refs in /root/dis232.txt
for a in "$@"; do
  echo "=== calls to 0x$a ==="
  grep -nE "call   0x$a\b" /root/dis232.txt | head -12
done
