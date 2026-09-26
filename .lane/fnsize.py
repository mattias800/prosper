#!/usr/bin/env python3
"""Report the line span of render_draw_pass_rgba in a given render_runner.h."""
import sys

path = sys.argv[1]
lines = open(path).read().split('\n')
if lines and lines[-1] == '':
    lines.pop()
head = next(i for i, l in enumerate(lines)
            if l.startswith('inline std::vector<uint8_t> render_draw_pass_rgba('))
# the body opens on the line ending the parameter list
depth = 0
bc = False
def strip(s):
    global bc
    out, j = [], 0
    while j < len(s):
        if bc:
            k = s.find('*/', j)
            if k < 0: break
            bc = False; j = k+2; continue
        if s.startswith('//', j): break
        if s.startswith('/*', j): bc = True; j += 2; continue
        if s[j] in '"\'':
            q = s[j]; j += 1
            while j < len(s):
                if s[j] == '\\': j += 2; continue
                if s[j] == q: j += 1; break
                j += 1
            continue
        out.append(s[j]); j += 1
    return ''.join(out)

for k in range(head):
    strip(lines[k])
j = head
seen = False
while j < len(lines):
    code = strip(lines[j])
    while '{}' in code:
        code = code.replace('{}', '')
    for c in code:
        if c == '{': depth += 1; seen = True
        elif c == '}': depth -= 1
    if seen and depth == 0:
        break
    j += 1
print(f"render_draw_pass_rgba: lines {head+1}-{j+1} = {j-head+1}; file = {len(lines)} "
      f"({100.0*(j-head+1)/len(lines):.1f}%)")
