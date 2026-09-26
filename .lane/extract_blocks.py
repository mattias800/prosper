#!/usr/bin/env python3
"""Step 3/4: move whole statement runs out of render_draw_pass_rgba into namespace-scope inline
functions, replacing each with a call.

Every moved line is verbatim apart from a uniform dedent, and the dedent is checked to be
invertible line for line. The only lines this script writes are the signatures and the call sites
it declares below; the final pass rejects any other line in the output that is not an input line
or a dedent of one.
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


def locate(lines, text, marker=None):
    """Index of `text`, required unique -- or, with `marker`, the first occurrence after it."""
    if marker is not None:
        m = [i for i, l in enumerate(lines) if l == marker]
        if len(m) != 1:
            sys.exit(f"marker not unique ({len(m)}): {marker!r}")
        for i in range(m[0], len(lines)):
            if lines[i] == text:
                return i
        sys.exit(f"{text!r} not found after marker")
    hits = [i for i, l in enumerate(lines) if l == text]
    if len(hits) != 1:
        sys.exit(f"expected exactly one {text!r}, found {len(hits)}")
    return hits[0]


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
    sys.exit(f"unterminated block at line {start + 1}")


def dedent(body, n):
    pad = ' ' * n
    out = []
    for l in body:
        if not l.strip():
            out.append(l); continue
        if not l.startswith(pad):
            sys.exit(f"line is not indented {n}: {l!r}")
        d = l[n:]
        if pad + d != l:
            sys.exit(f"dedent is not invertible for {l!r}")
        out.append(d)
    return out


def extract(lines, job):
    """job: dict with first/last selection, dedent, signature, replacement."""
    first = locate(lines, job['first'], job.get('first_marker'))
    if 'last_block_head' in job:
        h = locate(lines, job['last_block_head'], job.get('last_marker'))
        last = block_end(lines, h)
        cut_first, cut_last = first, last
        body_first, body_last = first, last
    else:
        last = block_end(lines, first)
        cut_first, cut_last = first, last
        body_first, body_last = first + 1, last - 1   # the block's interior
        if 'inner_head' in job:
            # The body is a nested block's interior; the CUT is still the outer block, whose
            # condition the replacement folds into the call site.
            ih = next(i for i in range(first, last + 1) if lines[i] == job['inner_head'])
            ie = block_end(lines, ih)
            body_first, body_last = ih + 1, ie - 1

    body = dedent(lines[body_first:body_last + 1], job['dedent'])
    a = locate(lines, ANCHOR)
    if not a < cut_first:
        sys.exit("anchor must precede the block")
    fn = job['signature'] + body + ['}', '']
    out = lines[:a] + fn + lines[a:cut_first] + job['replacement'] + lines[cut_last + 1:]
    start = a + len(job['signature'])
    if out[start:start + len(body)] != body:
        sys.exit("verify: moved body is not where it was written")
    return out


def read_frag(name):
    p = pathlib.Path('.lane/frag') / name
    ls = p.read_text().split('\n')
    if ls and ls[-1] == '':
        ls.pop()
    return ls


BANNER = [
    '// ---------------------------------------------------------------------------------------------',
    "// The pass's reporting, lifted out of it.",
    '//',
    '// Each of these was a block inside `render_draw_pass_rgba` gated on an environment variable, or',
    '// on the timing mode. None of them affects what is rendered: they read what the pass already',
    '// decided or what the device already produced, and write to stderr. They sat in the body only',
    '// because the locals they read were there, and they were a fifth of it.',
    '// ---------------------------------------------------------------------------------------------',
    '',
]

JOBS = [
    {   # PROSPER_DEPTH_CLEAR_WHY
        'first': '    if (PROSPER_ENV_ON("PROSPER_DEPTH_CLEAR_WHY") && use_depth) {',
        'dedent': 4,
        'signature': read_frag('depthwhy_hdr.txt'),
        'replacement': [
            '    if (PROSPER_ENV_ON("PROSPER_DEPTH_CLEAR_WHY") && use_depth)',
            '        report_depth_clear_derivation(logical_draws, color_target, depth_clear);',
        ],
    },
    {   # PROSPER_DSLOG
        'first': '    if (PROSPER_ENV_ON("PROSPER_DSLOG")) {',
        'dedent': 4,
        'signature': read_frag('dslog_hdr.txt'),
        'replacement': [
            '    if (PROSPER_ENV_ON("PROSPER_DSLOG"))',
            '        report_backend_ds_state(W, H, draws, use_depth, use_stencil, persistent_ds,',
            '                                ds_layout_initialized, depth_was_valid, stencil_was_valid,',
            '                                ds_key, depth_clear, stencil_clear);',
        ],
    },
    {   # PROSPER_BUFVERIFY
        'first': '    if (buffer_verify_enabled) {',
        'dedent': 4,
        'signature': read_frag('bufverify_hdr.txt'),
        'replacement': [
            '    if (buffer_verify_enabled)',
            '        report_backend_buffer_verify(buffer_verify_records, shared_buffers);',
        ],
    },
    {   # PROSPER_DS_SLICE_CENSUS
        'first': '        if (PROSPER_ENV_ON("PROSPER_DS_SLICE_CENSUS")) {',
        'dedent': 8,
        'signature': read_frag('dsslice_hdr.txt'),
        'replacement': [
            '        if (PROSPER_ENV_ON("PROSPER_DS_SLICE_CENSUS"))',
            '            report_ds_slice_census(ds_key, use_depth, depth_used_meaningfully);',
        ],
    },
    {   # PROSPER_DRAW_STATS
        'first': '    if (ds_active) {',
        'dedent': 4,
        'signature': read_frag('drawstats_hdr.txt'),
        'replacement': [
            '    if (ds_active)',
            '        report_draw_stats_funnel(dev, ds_stats_pool, ds_occ_pool, dv, draws,',
            '                                 batch_completed);',
        ],
    },
    {   # the geometry-probe readback
        'first': '    if (geom_active) {',
        'first_marker': ('    // Geometry-probe readback: report where the probed draw\'s '
                         'post-transform clip-space vertices landed.'),
        'inner_head': '        if (batch_completed) {',
        'dedent': 8,
        'signature': read_frag('geom_hdr.txt'),
        'replacement': [
            '    if (geom_active && batch_completed)',
            '        report_geometry_probe_readback(dev, geom_counter_mem, geom_mem, geom_cap,',
            '                                       geom_target_label);',
        ],
    },
    {   # the backend timing aggregates and windows
        'first': '        struct TimingTotals {',
        'last_block_head': '        if (print_backend_timing_windows && totals.calls % 25 == 0) {',
        'dedent': 4,
        'signature': read_frag('timing_hdr.txt'),
        'replacement': [
            '        accumulate_backend_render_timing(call_timing, draws, texture_stats,',
            '                                         resource_reuse_stats, pipeline_stats);',
        ],
    },
]


def main() -> int:
    text = FILE.read_text()
    lines = text.split('\n')
    trailing = bool(lines) and lines[-1] == ''
    if trailing:
        lines.pop()
    before = list(lines)

    for job in JOBS:
        lines = extract(lines, job)
        print(f"extracted {job['first'].strip()[:66]}")

    first = locate(lines, JOBS[0]['signature'][0])
    lines[first:first] = BANNER

    written = set(BANNER) | {'}', ''}
    for job in JOBS:
        written |= set(job['signature']) | set(job['replacement'])
    originals = set(before)
    dedents = set()
    for n in {job['dedent'] for job in JOBS}:
        dedents |= {l[n:] for l in before if l.startswith(' ' * n)}
    for l in lines:
        if l not in originals and l not in written and l not in dedents:
            sys.exit(f"unexpected new line in output: {l!r}")

    FILE.write_text('\n'.join(lines) + ('\n' if trailing else ''))
    print(f"{len(before)} -> {len(lines)} lines")
    return 0


if __name__ == '__main__':
    sys.exit(main())
