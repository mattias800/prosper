#!/usr/bin/env python3
"""Simulate 0x413dc3400's six-reference selection over a captured input, and diff it against what
the dispatch actually wrote.

Why six and not two. The builder does not read one pair of child references per node; it reads
three pairs, chaining through the first two (retained ISA, #2542):

    pc86   load the first pair from the compacted-node record        -> refs A, B
    pc161  follow (A >> 3) - 4 and load another pair                 -> refs C, D
    pc237  follow (B >> 3) - 4 and load another pair                 -> refs E, F
    six predicate/store blocks, each accepting tag in {2, 5}

A histogram over record dwords 0/1 therefore covers two of six candidates and CANNOT predict how
many parent slots a dispatch writes. That is why `pairs == 1030` is only an empirical control for one
route state and not a universal oracle: the active topology has to be derived per dispatch. What does
survive as a hard oracle is acyclicity — a cycle among reachable references cannot be explained by
legitimate non-materialised (tag != 2, 5) references.

Inputs are the tree-watch dumps: the node records (`aux-*.bin`, 64-byte stride) plus the parent
table's pre- and post-images from the same dispatch, so "what was written" is the changed set rather
than a guess.
"""
import glob
import os
import struct
import sys

RECORD_BYTES = 64
NODE_COUNT = 2063
MATERIALISED_TAGS = (2, 5)


def refs_of(records, index):
    """The two references a node record carries, or None when the index is out of range."""
    if not (0 <= index < len(records)):
        return None
    return records[index]


def decode_index(ref):
    """The builder's own decode: v_lshrrev_b32 3 then v_add_nc_u32 -4."""
    return (ref >> 3) - 4


def expected_destinations(records):
    """Every slot the six-reference walk says should receive a parent link.

    Returns (destinations, chased, dangling) where `chased` counts references followed to a second
    pair and `dangling` counts follows that left the array — the latter matters because a dangling
    follow is a reference the guest would also not be able to use.
    """
    destinations = []
    chased = dangling = 0
    for node in range(len(records)):
        a, b = records[node]
        for first in (a, b):
            if (first & 7) in MATERIALISED_TAGS:
                destinations.append(decode_index(first))
            # The chain is followed regardless of the first pair's tag: pc161/pc237 address from the
            # reference value, and the tag test gates the STORE, not the load.
            nxt = refs_of(records, decode_index(first))
            if nxt is None:
                dangling += 1
                continue
            chased += 1
            for second in nxt:
                if (second & 7) in MATERIALISED_TAGS:
                    destinations.append(decode_index(second))
    return destinations, chased, dangling


def reference_cycles(records):
    """Cycles among reachable materialised references. Iterative, so a deep chain cannot blow the
    Python stack and be mistaken for a structural result."""
    n = len(records)
    state = [0] * n
    cycles = []
    for start in range(n):
        if state[start]:
            continue
        stack = [(start, 0)]
        path = []
        while stack:
            node, edge = stack[-1]
            if edge == 0:
                if state[node] == 1:
                    cycles.append(tuple(path[path.index(node):]) if node in path else (node,))
                    stack.pop()
                    continue
                if state[node] == 2:
                    stack.pop()
                    continue
                state[node] = 1
                path.append(node)
            kids = [decode_index(r) for r in records[node]
                    if (r & 7) in MATERIALISED_TAGS and 0 <= decode_index(r) < n]
            if edge < len(kids):
                stack[-1] = (node, edge + 1)
                stack.append((kids[edge], 0))
            else:
                state[node] = 2
                if path and path[-1] == node:
                    path.pop()
                stack.pop()
    return cycles


def load_records(path):
    data = open(path, 'rb').read()
    count = min(NODE_COUNT, len(data) // RECORD_BYTES)
    return [struct.unpack_from('<II', data, i * RECORD_BYTES) for i in range(count)]


def load_words(path):
    data = open(path, 'rb').read()
    return list(struct.unpack('<%dI' % (len(data) // 4), data[:len(data) // 4 * 4]))


def main(directory):
    rows = []
    for post in sorted(glob.glob(os.path.join(directory, 'tree-post-*p413dc3400.bin'))):
        stem = os.path.basename(post)[len('tree-post-'):-len('-p413dc3400.bin')]
        pre = os.path.join(directory, 'tree-pre-%s-p413dc3400.bin' % stem)
        aux = glob.glob(os.path.join(directory, 'aux-*-%s.bin' % stem))
        if not os.path.exists(pre) or not aux:
            continue
        records = load_records(aux[0])
        if len(records) != NODE_COUNT:
            continue
        before, after = load_words(pre), load_words(post)
        written = {i for i in range(min(len(before), len(after))) if before[i] != after[i]}

        wanted, chased, dangling = expected_destinations(records)
        in_range = {d for d in wanted if 0 <= d < NODE_COUNT}
        cycles = reference_cycles(records)

        missing = sorted(in_range - written)
        unexpected = sorted(written - in_range)
        rows.append((stem, len(in_range), len(written), len(missing), len(unexpected),
                     len(cycles), dangling))
        print("== %s" % stem)
        print("   expected destinations (six-reference walk): %d distinct   chased=%d dangling=%d"
              % (len(in_range), chased, dangling))
        print("   actually written slots: %d" % len(written))
        print("   expected but NOT written: %d %s" % (len(missing), missing[:10]))
        print("   written but NOT expected: %d %s" % (len(unexpected), unexpected[:10]))
        print("   reference cycles: %d" % len(cycles))
        if cycles:
            print("   first cycles: %s" % [c for c in cycles[:4]])
        print()

    if rows:
        print("summary (stem, expected, written, missing, unexpected, cycles, dangling):")
        for r in rows:
            print("   %-28s %5d %5d %5d %5d %5d %6d" % r)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
