#!/usr/bin/env python3
"""Step 4: move the two texture-upload loops out of render_draw_pass_rgba.

Same machinery and the same checks as .lane/extract_blocks.py -- every moved line is verbatim
apart from a uniform, invertible dedent, and the only new lines are the signatures and calls
declared here.
"""
from __future__ import annotations
import sys, pathlib, importlib.util

spec = importlib.util.spec_from_file_location('eb', pathlib.Path('.lane/extract_blocks.py'))
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)

JOBS = [
    {   # the recording loop
        'first': '    for (const auto& upload : texture_uploads) {',
        'first_marker': ('    // Upload each distinct texture once. Draw descriptors may use '
                         'separate views/samplers over the'),
        'whole': True,
        'dedent': 0,
        'signature': eb.read_frag('texup_hdr.txt'),
        'replacement': [
            '    record_texture_uploads(texture_uploads, cmd, graphics_shader_stages);',
        ],
    },
    {   # the post-submit writeback
        'first': '    if (storage_writeback_requested && batch_completed) {',
        'dedent': 4,
        'signature': eb.read_frag('writeback_hdr.txt'),
        'replacement': [
            '    if (storage_writeback_requested && batch_completed)',
            '        writeback_storage_textures(texture_uploads, dev);',
        ],
    },
]


def extract(lines, job):
    first = eb.locate(lines, job['first'], job.get('first_marker'))
    last = eb.block_end(lines, first)
    if job.get('whole'):
        body_first, body_last = first, last
    else:
        body_first, body_last = first + 1, last - 1
    body = eb.dedent(lines[body_first:body_last + 1], job['dedent'])
    a = eb.locate(lines, eb.ANCHOR)
    if not a < first:
        sys.exit('anchor must precede the block')
    fn = job['signature'] + body + ['}', '']
    out = lines[:a] + fn + lines[a:first] + job['replacement'] + lines[last + 1:]
    start = a + len(job['signature'])
    if out[start:start + len(body)] != body:
        sys.exit('verify: moved body is not where it was written')
    return out


def main() -> int:
    text = eb.FILE.read_text()
    lines = text.split('\n')
    trailing = bool(lines) and lines[-1] == ''
    if trailing:
        lines.pop()
    before = list(lines)

    for job in JOBS:
        lines = extract(lines, job)
        print(f"extracted {job['first'].strip()[:66]}")

    written = {'}', ''}
    for job in JOBS:
        written |= set(job['signature']) | set(job['replacement'])
    originals = set(before)
    dedents = set()
    for n in {job['dedent'] for job in JOBS}:
        dedents |= {l[n:] for l in before if l.startswith(' ' * n)}
    for l in lines:
        if l not in originals and l not in written and l not in dedents:
            sys.exit(f"unexpected new line in output: {l!r}")

    eb.FILE.write_text('\n'.join(lines) + ('\n' if trailing else ''))
    print(f"{len(before)} -> {len(lines)} lines")
    return 0


if __name__ == '__main__':
    sys.exit(main())
