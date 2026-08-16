#!/usr/bin/env python3
"""Reconstruct one frame's render-pass graph from a `PROSPER_RTTLOG=1` run log.

The question this answers is "where does the picture stop being there". A title can render a complete
world into an off-screen target and still present black, and neither a draw census nor a target census
can see that: both are counts of *work*, and the defect is in the *dataflow*. What separates the two is
the pair (what a pass sampled, what its output then contained), which `PROSPER_RTTLOG` already prints
and which nothing assembled.

The log interleaves two line kinds, and the ordering is the whole trick:

    [rtt] sample tex addr=0x… WxH fmt=N -> HIT|miss (cache_size=…)     one per sampled binding
    [rtt] pass target=0x… extent=WxH … (N draws) px_nonzero=… rgb_nonblack=…    one per pass group

Every `sample` line belongs to a draw of the *next* `pass` line, so a pass's inputs are exactly the
samples since the previous pass. Frames are delimited by the `SCANOUT` pass.

`rgb_nonblack` is the number of pixels with a non-zero colour, so a pass whose inputs are populated and
whose output is not is where content is lost. Note `px_nonzero` counts BYTES and includes alpha: a
buffer reading `px_nonzero=8294400 rgb_nonblack=3527` at 3840x2160 is alpha-only, not "nearly full".
That distinction is why both are printed here rather than a single "is it black" verdict.

Usage:
    rtt_pass_graph.py <run.log> [--frames N] [--min-draws N] [--only 0xADDR]
"""
import argparse
import re
import sys

PASS_RE = re.compile(
    r'\[rtt\] pass (?:target|c\d+)=(0x[0-9a-f]+) extent=(\d+)x(\d+).*?\((\d+) draws\) '
    r'(?:src=(\d+)B )?px_nonzero=(\d+)(?: rgb_nonblack=(\d+))?')
SAMPLE_RE = re.compile(
    r'\[rtt\] sample tex addr=(0x[0-9a-f]+) (\d+)x(\d+) fmt=(\d+) -> (HIT|HIT-GPU|HIT-CPU|HIT-UNIFORM|miss|DS-DEPTH|DS-STENCIL)')

# prosper::gpu::DataFormat, for reading a pass's role off its format rather than guessing from extent.
FORMATS = {0: 'unknown', 1: 'f32', 2: 'u32', 3: 'i32', 4: 'f16', 5: 'un16', 6: 'sn16', 7: 'u16',
           8: 'i16', 9: 'un8', 10: 'sn8', 11: 'u8', 12: 'i8', 13: 'bc1', 14: 'bc2', 15: 'bc3',
           16: 'bc4', 17: 'bc5', 18: 'bc6', 19: 'bc7', 20: 'f11f11f10', 21: 'un2_10_10_10'}


def fmt_name(n):
    return FORMATS.get(n, 'fmt%d' % n)


def parse(path):
    """Yield frames, each a list of passes; a pass is (target, w, h, draws, nz, rgb, inputs, scanout)."""
    frames, passes, pending = [], [], []
    with open(path, 'r', errors='replace') as handle:
        for line in handle:
            if '[rtt] ' not in line:
                continue
            m = SAMPLE_RE.search(line)
            if m:
                pending.append((m.group(1), int(m.group(2)), int(m.group(3)),
                                int(m.group(4)), m.group(5)))
                continue
            m = PASS_RE.search(line)
            if not m:
                continue
            scanout = 'SCANOUT' in line
            # src is the readback byte count. Zero counters with src=0 mean the readback was
            # deferred -- "we never looked" -- not that the target is black. Logs written before
            # the field existed have src=None, and their zeros are simply ambiguous.
            src = int(m.group(5)) if m.group(5) is not None else None
            passes.append((m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)),
                           int(m.group(6)), int(m.group(7)) if m.group(7) else None,
                           pending, scanout, src))
            pending = []
            if scanout:
                frames.append(passes)
                passes = []
    if passes:
        frames.append(passes)
    return frames


def summarize_inputs(inputs):
    """Distinct sampled sources, in first-use order, with their hit state and sample count."""
    order, seen = [], {}
    for addr, w, h, fmt, state in inputs:
        key = (addr, w, h, fmt)
        if key not in seen:
            seen[key] = [0, state]
            order.append(key)
        seen[key][0] += 1
        if state == 'miss':
            seen[key][1] = 'miss'
    return [(k, seen[k][0], seen[k][1]) for k in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--frames', type=int, default=1, help='how many trailing frames to print')
    ap.add_argument('--min-draws', type=int, default=0, help='hide passes below this draw count')
    ap.add_argument('--only', help='print only passes writing or sampling this address')
    args = ap.parse_args()

    frames = parse(args.log)
    if not frames:
        print('no [rtt] pass lines found -- was the run made with PROSPER_RTTLOG=1?', file=sys.stderr)
        return 1
    print('%d frame(s) delimited by SCANOUT; showing the last %d\n' %
          (len(frames), min(args.frames, len(frames))))

    for frame in frames[-args.frames:]:
        for target, w, h, draws, nz, rgb, inputs, scanout, src in frame:
            if draws < args.min_draws:
                continue
            summarized = summarize_inputs(inputs)
            if args.only and args.only not in (
                    [target] + [k[0] for k, _, _ in summarized]):
                continue
            pixels = w * h
            share = ('%5.1f%%' % (100.0 * rgb / pixels)) if rgb is not None and pixels else '    ?'
            # A zero that came from an empty readback is not a measurement of the target.
            unread = ' NOT READ BACK (counters are void)' if src == 0 else ''
            print('%s %-14s %9s draws=%-5d rgb_nonblack=%-9s (%s of %d px)  px_nonzero=%d%s' %
                  ('SCANOUT' if scanout else '       ', target, '%dx%d' % (w, h), draws,
                   rgb if rgb is not None else '?', share, pixels, nz, unread))
            for (addr, iw, ih, ifmt), count, state in summarized:
                print('             <- %-14s %9s %-12s x%-4d %s' %
                      (addr, '%dx%d' % (iw, ih), fmt_name(ifmt), count,
                       {'HIT': '', 'HIT-GPU': '', 'HIT-CPU': 'cpu',
                       'HIT-UNIFORM': 'UNIFORM COLOUR -- no texture detail',
                       'DS-DEPTH': 'depth-bridge', 'DS-STENCIL': 'stencil-bridge'}
                      .get(state, 'MISS -- guest bytes')))
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
