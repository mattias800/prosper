#!/bin/bash
for f in /root/d232l_2.log /root/d232l_3.log /root/d232l_5.log /root/d232l_6.log; do
  echo "== $f lines=$(wc -l < "$f")"
  grep -a "read-submit id" "$f" | tail -2 | cut -c1-160
  echo "last read at line: $(grep -an 'read-submit id' "$f" | tail -1 | cut -d: -f1)"
done
