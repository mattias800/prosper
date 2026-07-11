#!/bin/bash
# Inspect a set of waf validation logs. Usage: waf_inspect.sh <logfile> [logfile...]
for L in "$@"; do
  echo "===== $L ====="
  echo "-- fatal signature --"
  grep -iE "MallocBinned3 Corruption|unrecognized block|WORKER-THREAD FAULT|Canary was|LowLevelFatal|Assertion failed|Fatal error" "$L" | head -4
  echo "-- last progress --"
  grep "\[progress\]" "$L" | tail -1
  echo "-- WAF categories (count) --"
  grep -oE "cat=[A-Za-z0-9-]+" "$L" | sort | uniq -c
  echo "-- non-pointer/canary/stale suppressions (first 5) --"
  grep -E "REL-WAF-SUPPRESS.*cat=(A-canary|B-stale0)" "$L" | head -5
  echo "-- guest wedge/timeout markers --"
  grep -cE "timed out waiting|GameThread timed|watchdog abort|DEFER-TIMEOUT" "$L"
done
