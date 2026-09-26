#!/usr/bin/env python3
"""Report depth-1 blocks (and their sizes) inside a function body range."""
import sys

path = sys.argv[1]
lo = int(sys.argv[2]); hi = int(sys.argv[3])
minsize = int(sys.argv[4]) if len(sys.argv) > 4 else 20

lines = open(path).read().split('\n')
in_block_comment = False

def strip(s):
    global in_block_comment
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
    return ''.join(out)

depth = 0
stack = []   # (open_line, depth)
for i in range(lo-1, hi):
    code = strip(lines[i])
    for c in code:
        if c == '{':
            depth += 1
            stack.append((i+1, depth))
        elif c == '}':
            if stack:
                ol, d = stack.pop()
                if d == 2 and (i+1-ol) >= minsize:
                    head = lines[ol-1].strip()
                    # look back for a leading comment run
                    print(f"{ol:6d}-{i+1:<6d} ({i+1-ol+1:5d}) {head[:130]}")
            depth -= 1
