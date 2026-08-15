#!/usr/bin/env python3
"""Test the LBVH model (#2542) against every captured 2063-dword parent table.

Codex's prediction: 2063 = 2*1032-1 is a full binary tree over 1032 leaves with 1031 internal
nodes, adjacent records are siblings sharing a parent, bit 30 identifies the child side, and the
root is the single unpaired node. If that is right, a well-formed table should show a pair count
at or very near 1031.

The point of running it over the CLEAN captures is that they are the ones the model must fit. A
model that only fits the corrupted ones explains nothing.
"""
import glob
import struct
import sys

SIDE = 0x40000000
SHIFT, MASK = 3, (1 << 27) - 1


def load(path):
    data = open(path, 'rb').read()
    return list(struct.unpack('<%dI' % (len(data) // 4), data[:len(data) // 4 * 4]))


def siblings(w):
    pairs = unpaired = side = 0
    for i, v in enumerate(w):
        if not (v & SIDE):
            continue
        side += 1
        if i and not (w[i - 1] & SIDE) and v == (w[i - 1] | SIDE):
            pairs += 1
        else:
            unpaired += 1
    return pairs, unpaired, side


def walk(w, roots):
    cyc = set()
    cyclic_roots = oob = 0
    maxd = 0
    n = len(w)
    for r in range(min(roots, n)):
        gen, seen, idx, d = r + 1, {}, r, 0
        while idx != 0:
            if idx >= n:
                oob += 1
                d += 1
                break
            if seen.get(idx) == gen:
                cyclic_roots += 1
                clen = d - seen_depth[idx]
                canon = member = idx
                for _ in range(clen):
                    canon = min(canon, member)
                    member = (w[member] >> SHIFT) & MASK
                cyc.add(canon)
                break
            seen[idx] = gen
            seen_depth[idx] = d
            idx = (w[idx] >> SHIFT) & MASK
            d += 1
        maxd = max(maxd, d)
    return len(cyc), cyclic_roots, oob, maxd


seen_depth = {}

paths = sorted(set(sum((glob.glob(p) for p in sys.argv[1:]), [])))
print("%-58s %6s %6s %6s %6s %7s %6s" %
      ("capture", "pairs", "unpair", "side", "cycles", "cycroot", "depth"))
clean_pairs, cyclic_pairs = [], []
for p in paths:
    w = load(p)
    if len(w) != 2063:
        continue
    seen_depth = {}
    pr, un, sd = siblings(w)
    cy, cr, ob, md = walk(w, 2063)
    print("%-58s %6d %6d %6d %6d %7d %6d" % (p.split('/')[-1][:58], pr, un, sd, cy, cr, md))
    (cyclic_pairs if cy else clean_pairs).append(pr)

def summary(name, xs):
    if not xs:
        print("%s: none" % name)
        return
    print("%s: n=%d pairs min=%d max=%d distinct=%s" %
          (name, len(xs), min(xs), max(xs), sorted(set(xs))[:8]))

print()
summary("CLEAN  (no cycles)", clean_pairs)
summary("CYCLIC (cycles>0) ", cyclic_pairs)
print()
print("LBVH prediction: a full binary tree over 1032 leaves has 1031 internal nodes,")
print("so a well-formed parent table should show pairs at or near 1031.")
