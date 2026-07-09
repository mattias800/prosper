#!/bin/bash
# Print the thread-1 (GameThread) section of each gw232i sample.
for i in 1 2 3; do
  echo "=== SAMPLE $i thread1 ==="
  sed -n '/^Thread 1 /,/^Thread 2 /p' /root/gw232i_$i.txt | head -60
done
