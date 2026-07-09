#!/bin/bash
for i in 1 2 3; do
  echo "=== SAMPLE $i Thread 1 section ==="
  awk '/^Thread 1 \(/{f=1} f{print; n++} n>70{exit}' /root/gw232i_$i.txt
done
