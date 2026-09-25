#!/usr/bin/env python3
"""Classify native CPU-clock samples against complete F8 renderer-submit spans.

This is sampled CPU activity in wall-time intervals, not elapsed-time attribution.
F8 timestamps are relative to its trigger; perf script times use the monotonic clock.
Export leaf IPs even when DWARF unwinding fails with:
  perf script -G --no-inline -F comm,pid,tid,time,period,event,ip,sym,dso

See PERF_F8_GAP.md for recording and validity checks outside this parser.
"""
import argparse
import bisect
import collections
import json
import re
from pathlib import Path

HEADER = re.compile(r"^(.+?)\s+(\d+)/(\d+)\s+(\d+)\.(\d{1,9}):\s+(\d+)\s+cpu-clock:u:\s*$")
FRAME = re.compile(r"^\s+([0-9a-fA-F]+)\s+(.+)\s+\((.+)\)\s*$")
FLAT = re.compile(r"^(.+?)\s+(\d+)/(\d+)\s+(\d+)\.(\d{1,9}):\s+(\d+)\s+cpu-clock:u:\s+([0-9a-fA-F]+)\s+(.+)\s+\((.+)\)\s*$")


def load_spans(path):
    with Path(path).open() as source:
        records = [json.loads(line) for line in source]
    headers = [r for r in records if r.get('type') == 'header']
    footers = [r for r in records if r.get('type') == 'footer']
    if len(headers) != 1 or len(footers) != 1:
        raise ValueError('F8 capture needs exactly one header and one footer')
    header, footer = headers[0], footers[0]
    if not footer.get('complete') or footer.get('renderer_dropped') or footer.get('compute_dropped'):
        raise ValueError('F8 capture is incomplete or dropped detail records')
    trigger = header['trigger_monotonic_ns']
    if type(trigger) is not int or trigger < 0:
        raise ValueError('F8 trigger_monotonic_ns must be a nonnegative integer')
    renderer = [r for r in records if r.get('type') == 'renderer']
    if footer.get('renderer_records') != len(renderer):
        raise ValueError('F8 renderer record count does not match footer')
    for r in renderer:
        if (type(r.get('span_start_t_ns')) is not int or type(r.get('t_ns')) is not int or
                r['span_start_t_ns'] < 0 or r['t_ns'] < 0):
            raise ValueError('F8 renderer span lacks two nonnegative integer endpoints')
    spans = sorted((trigger + r['span_start_t_ns'], trigger + r['t_ns'])
                   for r in renderer)
    if len(spans) < 2 or any(a >= b for a, b in spans) or any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError('need at least two complete, nonoverlapping renderer spans')
    return trigger, spans


def iter_samples(path):
    current = None
    with Path(path).open() as source:
        for line_no, line in enumerate(source, 1):
            line = line.rstrip('\n')
            match = FLAT.fullmatch(line)
            if match:
                if current is not None:
                    raise ValueError(f'line {line_no}: mixed flat and call-graph exports')
                comm, pid, tid, seconds, fraction, period, ip, symbol, dso = match.groups()
                yield {'comm': comm.strip(), 'pid': int(pid), 'tid': int(tid),
                       'time_ns': int(seconds) * 10**9 + int(fraction.ljust(9, '0')),
                       'period': int(period),
                       'frames': [{'ip': int(ip, 16), 'symbol': symbol, 'dso': dso}]}
                continue
            match = HEADER.fullmatch(line)
            if match:
                if current is not None:
                    yield current
                comm, pid, tid, seconds, fraction, period = match.groups()
                current = {'comm': comm, 'pid': int(pid), 'tid': int(tid),
                           'time_ns': int(seconds) * 10**9 + int(fraction.ljust(9, '0')),
                           'period': int(period), 'frames': []}
                continue
            match = FRAME.fullmatch(line)
            if match:
                if current is None:
                    raise ValueError(f'line {line_no}: frame without sample')
                ip, symbol, dso = match.groups()
                current['frames'].append({'ip': int(ip, 16), 'symbol': symbol, 'dso': dso})
            elif line.strip():
                raise ValueError(f'line {line_no}: unsupported perf script line: {line[:100]}')
    if current is not None:
        yield current


def zone(time_ns, spans, starts):
    index = bisect.bisect_right(starts, time_ns) - 1
    if index < 0 or time_ns >= spans[-1][1]:
        return 'outside'
    if time_ns < spans[index][1]:
        return 'renderer-span'
    return 'between-spans'


def require_window_coverage(first_perf, last_perf, spans, edge_slack_ns=500_000_000):
    if first_perf is None or last_perf is None:
        raise ValueError('perf script contains no samples')
    if first_perf > spans[0][0] + edge_slack_ns or last_perf < spans[-1][1] - edge_slack_ns:
        raise ValueError('perf samples do not cover the F8 detail interval: '
                         f'perf=[{first_perf},{last_perf}] F8=[{spans[0][0]},{spans[-1][1]}]')


def summarize(f8, perf_script):
    trigger, spans = load_spans(f8)
    starts = [start for start, _ in spans]
    rows = {name: {'samples': 0, 'period_ns': 0,
                   'tids': collections.Counter(), 'leaves': collections.Counter(),
                   'app_frames': collections.Counter(), 'no_stack': 0,
                   'no_named_app_frame': 0}
            for name in ('renderer-span', 'between-spans', 'outside')}
    first_perf = last_perf = None
    pids = set()
    for sample in iter_samples(perf_script):
        pids.add(sample['pid'])
        if len(pids) != 1:
            raise ValueError('perf export contains multiple processes; record one game process')
        time_ns = sample['time_ns']
        first_perf = time_ns if first_perf is None else min(first_perf, time_ns)
        last_perf = time_ns if last_perf is None else max(last_perf, time_ns)
        row = rows[zone(time_ns, spans, starts)]
        row['samples'] += 1
        row['period_ns'] += sample['period']
        row['tids'][str(sample['tid'])] += 1
        if sample['frames']:
            leaf = sample['frames'][0]
            row['leaves'][f"{leaf['symbol']} ({leaf['dso']})"] += 1
        else:
            row['no_stack'] += 1
        app_frame = next((frame for frame in sample['frames']
                          if Path(frame['dso']).name == 'prosper-app' and
                          not frame['symbol'].startswith('[unknown]')), None)
        if app_frame:
            row['app_frames'][app_frame['symbol']] += 1
        else:
            row['no_named_app_frame'] += 1
    require_window_coverage(first_perf, last_perf, spans)
    if not rows['renderer-span']['samples'] or not rows['between-spans']['samples']:
        raise ValueError('perf/F8 overlap lacks samples in one of the two comparison zones')
    return {
        'pid': next(iter(pids)),
        'trigger_monotonic_ns': trigger,
        'f8_spans': len(spans),
        'f8_first_span_ns': spans[0][0], 'f8_last_span_ns': spans[-1][1],
        'perf_first_sample_ns': first_perf, 'perf_last_sample_ns': last_perf,
        'zones': {name: {'samples': row['samples'], 'sampled_cpu_clock_ms': row['period_ns']/1e6,
                         'no_stack_samples': row['no_stack'],
                         'no_named_app_frame_samples': row['no_named_app_frame'],
                         'top_tids': row['tids'].most_common(12),
                         'top_leaves': row['leaves'].most_common(20),
                         'top_app_frames': row['app_frames'].most_common(20)}
                  for name, row in rows.items()}}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('f8', type=Path)
    ap.add_argument('perf_script', type=Path)
    args = ap.parse_args()
    print(json.dumps(summarize(args.f8, args.perf_script), indent=2))
