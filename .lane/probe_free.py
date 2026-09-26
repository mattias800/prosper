#!/usr/bin/env python3
"""MEASUREMENT, not a refactor step. Cut a block out of render_draw_pass_rgba into a
parameterless function so the compiler names every free variable it uses. Run against a scratch
copy of the tree; it rewrites the header in place.

  probe_free.py '<exact head line>' <dedent> [occurrence]
"""
from __future__ import annotations
import sys, pathlib

FILE = pathlib.Path('prosper/tests/fixtures/render_runner.h')
ANCHOR = 'inline std::vector<uint8_t> render_draw_pass_rgba(std::span<const BackendDraw> draws,'


def strip_code(s, state):
    out, j = [], 0
    while j < len(s):
        if state['bc']:
            k = s.find('*/', j)
            if k < 0:
                break
            state['bc'] = False; j = k + 2; continue
        if s.startswith('//', j):
            break
        if s.startswith('/*', j):
            state['bc'] = True; j += 2; continue
        if s[j] in '"\'':
            q = s[j]; j += 1
            while j < len(s):
                if s[j] == '\\': j += 2; continue
                if s[j] == q: j += 1; break
                j += 1
            continue
        out.append(s[j]); j += 1
    return ''.join(out)


def block_end(lines, start):
    state = {'bc': False}
    for k in range(start):
        strip_code(lines[k], state)
    depth, j = 0, start
    while j < len(lines):
        for c in strip_code(lines[j], state):
            if c == '{': depth += 1
            elif c == '}': depth -= 1
        if j > start and depth == 0:
            return j
        j += 1
    sys.exit('unterminated')


def main():
    head_text = sys.argv[1]
    ded = int(sys.argv[2])
    occ = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    lines = FILE.read_text().split('\n')
    trailing = lines and lines[-1] == ''
    if trailing:
        lines.pop()
    hits = [i for i, l in enumerate(lines) if l == head_text]
    print(f"{len(hits)} occurrence(s); using #{occ}", file=sys.stderr)
    h = hits[occ]
    e = block_end(lines, h)
    whole = '--whole' in sys.argv
    span = lines[h:e + 1] if whole else lines[h + 1:e]
    body = [l[ded:] if l.strip() else l for l in span]
    a = next(i for i, l in enumerate(lines) if l == ANCHOR)
    n = sum(1 for l in lines if l.startswith('inline void probe_block'))
    fn = [f'inline void probe_block_{n}() {{'] + body + ['}', '']
    out = lines[:a] + fn + lines[a:h] + ['    // probe: block removed'] + lines[e + 1:]
    FILE.write_text('\n'.join(out) + ('\n' if trailing else ''))
    print(f"probe: cut lines {h + 1}-{e + 1} ({e - h + 1}) into probe_block()", file=sys.stderr)


if __name__ == '__main__':
    main()
