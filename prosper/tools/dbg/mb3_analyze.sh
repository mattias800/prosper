#!/bin/bash
L=$1
echo "=== fatal signatures ==="
grep -cE "MallocBinned3 Corruption" "$L" | sed 's/^/  MallocBinned3 Corruption: /'
grep -cE "unrecognized block" "$L" | sed 's/^/  unrecognized block: /'
grep -cE "WORKER-THREAD FAULT" "$L" | sed 's/^/  WORKER-THREAD FAULT: /'
grep -c "1000000001" "$L" | sed 's/^/  1000000001 refs: /'
grep -c "20015f00" "$L" | sed 's/^/  20015f00 refs: /'
grep -c "REL1-FORGE" "$L" | sed 's/^/  REL1-FORGE suppressed: /'
echo "=== first fatal + context ==="
N=$(grep -nE "MallocBinned3 Corruption|unrecognized block|WORKER-THREAD FAULT" "$L" | head -1 | cut -d: -f1)
if [ -n "$N" ]; then A=$((N-4)); B=$((N+34)); sed -n "${A},${B}p" "$L"; fi
