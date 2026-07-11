#!/bin/bash
L=$1
N=$(grep -n "Canary was 0x3" "$L" | head -1 | cut -d: -f1)
echo "=== 40 lines BEFORE first Canary-0x3 (deep dump) ==="
A=$((N-42)); B=$((N-1)); sed -n "${A},${B}p" "$L"
