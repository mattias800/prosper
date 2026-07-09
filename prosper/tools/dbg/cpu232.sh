#!/bin/bash
# Per-thread CPU usage over 3s for the stalled DOLL (pid in /root/pid232.txt)
PID=$(cat /root/pid232.txt)
kill -0 "$PID" || { echo TARGET-DEAD; exit 1; }
declare -A A
for t in /proc/$PID/task/*; do
  tid=${t##*/}
  read -r -a f < "$t/stat" 2>/dev/null || continue
  A[$tid]=$(( ${f[13]} + ${f[14]} ))
done
sleep 3
for t in /proc/$PID/task/*; do
  tid=${t##*/}
  read -r -a f < "$t/stat" 2>/dev/null || continue
  now=$(( ${f[13]} + ${f[14]} ))
  d=$(( now - ${A[$tid]:-0} ))
  if [ "$d" -gt 0 ]; then
    name=$(cat "$t/comm" 2>/dev/null)
    echo "tid=$tid dticks=$d name=$name"
  fi
done | sort -t= -k3 -rn
echo CPU232-DONE
