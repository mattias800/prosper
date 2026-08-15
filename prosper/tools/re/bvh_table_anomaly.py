#!/usr/bin/env python3
"""Name the single corrupted record in a cyclic capture, without needing a clean reference.

The LBVH model plus the measurement makes this possible: a clean table has 1030 sibling pairs and
ZERO unpaired side-bit records; a cyclic one has 1029 pairs and EXACTLY ONE unpaired record. So the
unpaired record identifies itself, and the cycle it creates can be walked out and printed.

This does not assume the unpaired record is the corrupted one -- it prints the cycle members
independently, so the two identifications either agree (and confirm each other) or do not.
"""
import glob
import struct
import sys

SIDE = 0x40000000
SHIFT, MASK = 3, (1 << 27) - 1


def load(path):
    d = open(path, 'rb').read()
    return list(struct.unpack('<%dI' % (len(d) // 4), d[:len(d) // 4 * 4]))


def unpaired_records(w):
    out = []
    for i, v in enumerate(w):
        if not (v & SIDE):
            continue
        if i and not (w[i - 1] & SIDE) and v == (w[i - 1] | SIDE):
            continue
        out.append(i)
    return out


def find_cycles(w):
    n = len(w)
    state = [0] * n          # 0 unvisited, 1 on current path, 2 done
    cycles = []
    for r in range(n):
        if state[r]:
            continue
        path = []
        idx = r
        while idx != 0 and idx < n and state[idx] == 0:
            state[idx] = 1
            path.append(idx)
            idx = (w[idx] >> SHIFT) & MASK
        if idx != 0 and idx < n and state[idx] == 1:
            cycles.append(path[path.index(idx):])
        for p in path:
            state[p] = 2
    return cycles


def parent(w, i):
    return (w[i] >> SHIFT) & MASK


for path in sorted(set(sum((glob.glob(p) for p in sys.argv[1:]), []))):
    w = load(path)
    if len(w) != 2063:
        continue
    unp = unpaired_records(w)
    cycles = find_cycles(w)
    print("== %s" % path.split('/')[-1])
    print("   unpaired side-bit records: %s" % unp)
    for c in cycles:
        print("   cycle len=%d members=%s" % (len(c), c))
        for m in c:
            mate = m - 1 if (w[m] & SIDE) else m + 1
            mate_ok = 0 <= mate < len(w)
            print("      slot %4d = 0x%08x  parent=%4d  side=%d   mate slot %4d = %s" %
                  (m, w[m], parent(w, m), 1 if w[m] & SIDE else 0, mate,
                   ("0x%08x" % w[mate]) if mate_ok else "-"))
    # Neighbourhood of each cycle member, so a hand check of the pairing is possible.
    for c in cycles:
        lo, hi = max(0, min(c) - 3), min(len(w), max(c) + 4)
        print("   neighbourhood [%d,%d):" % (lo, hi))
        for i in range(lo, hi):
            print("      %4d 0x%08x parent=%4d side=%d %s" %
                  (i, w[i], parent(w, i), 1 if w[i] & SIDE else 0,
                   "<-- cycle" if i in c else ""))
    print()
