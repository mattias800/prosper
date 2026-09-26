#!/usr/bin/env python3
"""Turn capture-free function-local lambdas in render_draw_pass_rgba into namespace-scope inline
functions.

The BODY lines and any leading comment run move VERBATIM (the body dedented by exactly 4); only
the lambda's head line(s) and its `};` tail are replaced, by the new signature and a `}`. Two
checks, both able to fail: the dedent must be invertible line for line, and every line of the
result must be either a line of the input, a dedent of one, a line this script declares it wrote,
or an input line with `sonic_clear_probe` turned into a call.
"""
from __future__ import annotations
import sys, pathlib

FILE = pathlib.Path('prosper/tests/fixtures/render_runner.h')
ANCHOR = 'inline std::vector<uint8_t> render_draw_pass_rgba(std::span<const BackendDraw> draws,'


def strip_code(s: str, state: dict) -> str:
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


def find_block(lines: list[str], head_text: str) -> tuple[int, int]:
    hits = [i for i, l in enumerate(lines) if l == head_text]
    if len(hits) != 1:
        sys.exit(f"expected exactly one {head_text!r}, found {len(hits)}")
    i = hits[0]
    state = {'bc': False}
    for k in range(i):
        strip_code(lines[k], state)
    depth, j = 0, i
    while j < len(lines):
        for c in strip_code(lines[j], state):
            if c == '{': depth += 1
            elif c == '}': depth -= 1
        if j > i and depth == 0:
            return i, j
        j += 1
    sys.exit(f"unterminated block at {head_text!r}")


def dedent(body: list[str]) -> list[str]:
    out = []
    for l in body:
        if not l.strip():
            out.append(l); continue
        if not l.startswith('    '):
            sys.exit(f"body line is not indented 4: {l!r}")
        d = l[4:]
        if '    ' + d != l:
            sys.exit(f"dedent is not invertible for {l!r}")
        out.append(d)
    return out


def anchor_index(lines: list[str]) -> int:
    hits = [i for i, l in enumerate(lines) if l == ANCHOR]
    if len(hits) != 1:
        sys.exit(f"anchor not unique ({len(hits)})")
    return hits[0]


def convert(lines: list[str], head_text: str, head_lines: int, lead: int,
            signature: list[str], tail_text: str, tail_extra: list[str],
            dedent_body: bool = True) -> list[str]:
    head, tail = find_block(lines, head_text)
    if lines[tail] != tail_text:
        sys.exit(f"unexpected tail for {head_text!r}: {lines[tail]!r}")
    comments = lines[head - lead:head] if lead else []
    for c in comments:
        if not c.strip().startswith('//'):
            sys.exit(f"lead line is not a comment: {c!r}")
    raw = lines[head + head_lines:tail]
    body = dedent(raw) if dedent_body else raw
    a = anchor_index(lines)
    if not a < head - lead:
        sys.exit("anchor must precede the lambda and its comment run")
    fn = [c.strip() and c[4:] or c for c in comments] + signature + body + tail_extra + ['}', '']
    out = lines[:a] + fn + lines[a:head - lead] + lines[tail + 1:]
    start = a + len(comments) + len(signature)
    if out[start:start + len(body)] != body:
        sys.exit("verify: moved body is not where it was written")
    return out


PROBE_SIG = [
    'inline int sonic_clear_probe() {',
    '    // An accessor rather than a namespace-scope object on purpose: a function-local static of',
    '    // an inline function is ONE instance program-wide, initialized on the first call -- exactly',
    '    // where the static inside the render pass was initialized. A namespace-scope `static` in a',
    '    // header would be one copy per translation unit, each initialized before main, so a test',
    '    // that sets the variable and then renders would read a different answer than it does today.',
    '    static const int probe = [] {',
]

BANNER = ['// ---------------------------------------------------------------------------------------------', "// The pass's capture-free helpers.", '//', '// Each of these was a lambda inside `render_draw_pass_rgba` that captured nothing. A lambda that', '// captures nothing is a free function that has not been written down as one, and while it was', '// written that way it could only be called from inside the pass -- so every block that used one', '// was pinned in the body with it. They are ordinary functions now, and nothing else changed.', '// ---------------------------------------------------------------------------------------------', '']

JOBS = [
    # head text, head lines, lead comment lines, signature, tail text, extra lines before `}`
    ('    static const int sonic_clear_probe = [] {', 1, 9, PROBE_SIG, '    }();',
     ['    }();', '    return probe;'], False),
    ('    const auto effective_depth_clear = [](const prosper::gpu::ResolvedPipelineState* ps) {',
     1, 3,
     ['inline bool effective_depth_clear(const prosper::gpu::ResolvedPipelineState* ps) {'],
     '    };', [], True),
    ('    auto handle_bits = [](auto handle) {', 1, 0, [
        '// A Vulkan handle widened to 64 bits for use as part of a cache key. Generic because the',
        '// dispatchable and non-dispatchable handle types are not the same width on every build.',
        'inline uint64_t handle_bits(auto handle) {'], '    };', [], True),
    ('    auto float_bits = [](float value) {', 1, 0, [
        '// The IEEE bit pattern of a float, likewise for use as part of a cache key.',
        'inline uint32_t float_bits(float value) {'], '    };', [], True),
    ('    auto direct_range_eligible = [](const FrameBufferResource& r) {', 1, 0, [
        '// Whether a buffer resource can be uploaded as a direct guest range rather than through the',
        '// generic word copy: it must be a plain, large, contiguous guest buffer whose identity,',
        '// guest address and host pointer all agree.',
        'inline bool direct_range_eligible(const FrameBufferResource& r) {'], '    };', [], True),
    ('    auto setup_elapsed_ms = [](auto begin, auto end) {', 1, 0, [
        '// Elapsed milliseconds between two clock points, as a double.',
        'inline double setup_elapsed_ms(auto begin, auto end) {'], '    };', [], True),
    ('    auto record_stencil_dynamic_state = [](VkCommandBuffer command, const DV& v) {', 1, 0, [
        '// The stencil masks, reference and ops, set dynamically so one pipeline serves draws that',
        '// differ only in these.',
        'inline void record_stencil_dynamic_state(VkCommandBuffer command, const DV& v) {'],
     '    };', [], True),
    ('    auto record_pipeline_dynamic_state = [](VkCommandBuffer command, const DV& v) {', 1, 0, [
        '// The rasterizer and depth/stencil enables, for the same reason.',
        'inline void record_pipeline_dynamic_state(VkCommandBuffer command, const DV& v) {'],
     '    };', [], True),
    ('    auto record_draw_dynamic_state = [&](VkCommandBuffer command, const DV& v,', 2, 0, [
        '// Everything a draw sets dynamically. The only one of the three that consults the guest',
        '// pipeline state, because the clear-overwrite probe collapses the dynamic depth range onto',
        '// the clear value -- a diagnostic mode, never the default.',
        'inline void record_draw_dynamic_state(VkCommandBuffer command, const DV& v,',
        '                                      const prosper::gpu::ResolvedPipelineState* ps) {'],
     '    };', [], True),
]


def main() -> int:
    text = FILE.read_text()
    lines = text.split('\n')
    trailing = bool(lines) and lines[-1] == ''
    if trailing:
        lines.pop()
    before = list(lines)

    for head_text, head_lines, lead, signature, tail_text, tail_extra, ded in JOBS:
        lines = convert(lines, head_text, head_lines, lead, signature, tail_text, tail_extra, ded)
        print(f"hoisted {head_text.strip()[:64]}")

    # sonic_clear_probe is now a call. A bare use left behind would not compile, but a silent miss
    # is the failure mode worth a check rather than a build error to read.
    probe_at = next(i for i, l in enumerate(lines) if l == 'inline int sonic_clear_probe() {')
    for i in range(probe_at + len(PROBE_SIG), len(lines)):
        l = lines[i]
        if 'sonic_clear_probe' in l and 'return probe;' not in l:
            lines[i] = l.replace('sonic_clear_probe()', 'sonic_clear_probe') \
                        .replace('sonic_clear_probe', 'sonic_clear_probe()')
    for i in range(probe_at + len(PROBE_SIG), len(lines)):
        rest = lines[i].replace('sonic_clear_probe()', '')
        if 'sonic_clear_probe' in rest:
            sys.exit(f"a bare sonic_clear_probe survived on line {i + 1}: {lines[i]!r}")

    written = ({l for job in JOBS for l in job[3]} | {l for job in JOBS for l in job[5]}
               | set(BANNER) | {'}', ''})
    originals = set(before)
    dedented = {l[4:] for l in before if l.startswith('    ')}
    probed = {l.replace('sonic_clear_probe', 'sonic_clear_probe()') for l in before}
    probed |= {l[4:].replace('sonic_clear_probe', 'sonic_clear_probe()')
               for l in before if l.startswith('    ')}
    for l in lines:
        if l not in originals and l not in written and l not in dedented and l not in probed:
            sys.exit(f"unexpected new line in output: {l!r}")

    first = next(i for i, l in enumerate(lines)
                 if l == '// Investigation-only control for #2790. Mode 1 limits attachment clears'
                         ' to ALWAYS draws;')
    lines[first:first] = BANNER

    FILE.write_text('\n'.join(lines) + ('\n' if trailing else ''))
    print(f"{len(before)} -> {len(lines)} lines")
    return 0


if __name__ == '__main__':
    sys.exit(main())
