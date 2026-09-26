#!/usr/bin/env python3
"""Find the full line span of each `struct X {` ... `};` declared at a given indent
inside a line range, plus any leading comment run."""
import sys, re

path = sys.argv[1]
lo = int(sys.argv[2]); hi = int(sys.argv[3])
indent = sys.argv[4] if len(sys.argv) > 4 else '    '

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

# precompute stripped text for whole file (comment state must run from top)
in_block_comment = False
stripped = [strip(l) for l in lines]

pat = re.compile(r'^' + re.escape(indent) + r'(struct|class|union)\s+(\w+)')
i = lo - 1
while i < hi:
    m = pat.match(lines[i])
    if m and '{' in stripped[i]:
        name = m.group(2)
        depth = 0
        j = i
        while j < len(lines):
            for c in stripped[j]:
                if c == '{': depth += 1
                elif c == '}': depth -= 1
            if depth == 0 and j > i:
                break
            if depth == 0 and '{' in stripped[j] and '}' in stripped[j]:
                break
            j += 1
        # leading comment run
        s = i
        while s - 1 >= 0 and lines[s-1].strip().startswith('//'):
            s -= 1
        tail = lines[j].strip()
        print(f"{s+1}\t{j+1}\t{name}\t{tail[:80]}")
        i = j + 1
        continue
    i += 1
