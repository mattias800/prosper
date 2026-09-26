#!/usr/bin/env python3
"""Show the brace-depth-1 statement shape of a line range in a file."""
import sys, re

path = sys.argv[1]
lo = int(sys.argv[2]); hi = int(sys.argv[3])
maxdepth = int(sys.argv[4]) if len(sys.argv) > 4 else 1

lines = open(path).read().split('\n')
depth = 0
# strip strings/chars/comments crudely
in_block_comment = False
start_of_stmt = None
for i in range(lo-1, hi):
    raw = lines[i]
    s = raw
    # remove line comments and strings
    out = []
    j = 0
    while j < len(s):
        if in_block_comment:
            k = s.find('*/', j)
            if k < 0:
                j = len(s); break
            in_block_comment = False; j = k+2; continue
        if s.startswith('//', j): break
        if s.startswith('/*', j):
            in_block_comment = True; j += 2; continue
        if s[j] == '"':
            j += 1
            while j < len(s):
                if s[j] == '\\': j += 2; continue
                if s[j] == '"': j += 1; break
                j += 1
            continue
        if s[j] == "'":
            j += 1
            while j < len(s):
                if s[j] == '\\': j += 2; continue
                if s[j] == "'": j += 1; break
                j += 1
            continue
        out.append(s[j]); j += 1
    code = ''.join(out)
    before = depth
    for c in code:
        if c in '{': depth += 1
        elif c in '}': depth -= 1
    if before <= maxdepth:
        txt = raw.strip()
        if txt:
            print(f"{i+1:6d} d{before} {txt[:150]}")
