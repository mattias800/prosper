#!/bin/bash
# Kill stray DOLL boot_trace processes (not Messenger, not ourselves)
for p in /proc/[0-9]*/cmdline; do
  d=${p%/cmdline}; pid=${d#/proc/}
  [ "$pid" = "$$" ] && continue
  if tr "\0" " " < "$p" 2>/dev/null | grep -q "boot_trace /root/PPSA17942"; then
    echo "killing $pid"; kill -9 "$pid" 2>/dev/null
  fi
done
echo kill_doll done
