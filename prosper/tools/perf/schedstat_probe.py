#!/usr/bin/env python3
"""Sample the primary-name threads' Linux scheduler counters for an F8 interval.

The output is private diagnostic evidence, not a timing benchmark. Each row gives
the monotonic read bracket so an offline join can discard intervals crossing F8
span boundaries. A thread is identified by (TID, /proc stat starttime), not comm.
"""
import argparse
import json
import os
from pathlib import Path
import time


def stat_fields(raw):
    close = raw.rfind(')')
    if close < 0:
        raise ValueError('malformed /proc stat')
    fields = raw[close + 2:].split()
    if len(fields) <= 19:
        raise ValueError('short /proc stat')
    return fields[0], int(fields[19])


def read_thread(root, tid, wanted_name):
    task = root / 'task' / str(tid)
    comm = (task / 'comm').read_text().strip()
    if comm != wanted_name:
        return None
    state, starttime = stat_fields((task / 'stat').read_text())
    values = (task / 'schedstat').read_text().split()
    if len(values) != 3:
        raise ValueError('short /proc schedstat')
    runtime_ns, runnable_ns, switches = map(int, values)
    wchan = (task / 'wchan').read_text().strip() if state != 'R' else ''
    return {'tid': tid, 'comm': comm, 'starttime': starttime, 'state': state,
            'runtime_ns': runtime_ns, 'runnable_ns': runnable_ns,
            'switches': switches, 'wchan': wchan}


def select_threads(root, wanted_name):
    selected = []
    for text_tid in os.listdir(root / 'task'):
        try:
            if (root / 'task' / text_tid / 'comm').read_text().strip() == wanted_name:
                selected.append(int(text_tid))
        except OSError:
            continue  # A thread exited during the one-time selection.
    return sorted(selected)


# `proc_root` exists so the pacing, identity and failure paths can be driven against a synthetic
# tree in `test_schedstat_probe.py`. Nothing but the test passes it; production is always /proc.
def collect(pid, name, seconds, hz, proc_root=Path('/proc')):
    root = Path(proc_root) / str(pid)
    _, process_starttime = stat_fields((root / 'stat').read_text())
    # Select once. The target has been running for minutes by the measurement window;
    # scanning every task's comm on every tick would itself perturb a small gap.
    candidate_tids = select_threads(root, name)
    start = time.monotonic_ns()
    stop = start + round(seconds * 1e9)
    step = round(1e9 / hz)
    if step <= 0:
        raise ValueError('sampling interval rounded to zero')
    rows = []
    failures = 0
    overruns = 0
    tick = start
    terminal = 'completed'
    while True:
        now = time.monotonic_ns()
        if now >= stop:
            break
        if now < tick:
            time.sleep((tick - now) / 1e9)
        before = time.monotonic_ns()
        try:
            _, current_starttime = stat_fields((root / 'stat').read_text())
            if current_starttime != process_starttime:
                terminal = 'process identity changed'
                break
            threads = []
            for tid in candidate_tids:
                try:
                    thread = read_thread(root, tid, name)
                    if thread is not None:
                        threads.append(thread)
                except (OSError, ValueError):
                    failures += 1  # Thread exit or an unreadable proc entry.
        except (OSError, ValueError):
            terminal = 'process exited or became unreadable'
            break
        after = time.monotonic_ns()
        rows.append({'before_ns': before, 'after_ns': after, 'threads': threads})
        tick += step
        if after > tick:
            overruns += 1
            tick = after
    return {'pid': pid, 'process_starttime': process_starttime, 'name': name,
            'candidate_tids': candidate_tids,
            'requested_seconds': seconds, 'requested_hz': hz, 'start_ns': start,
            'stop_ns': time.monotonic_ns(), 'terminal': terminal,
            'sample_count': len(rows),
            'matched_thread_rows': sum(len(row['threads']) for row in rows),
            'read_failures': failures,
            'overruns': overruns, 'samples': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--name', default='prosper-app')
    parser.add_argument('--seconds', type=float, default=20.0)
    parser.add_argument('--hz', type=float, default=200.0)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.pid <= 0 or args.seconds <= 0 or args.hz <= 0 or args.out.exists():
        parser.error('positive pid/seconds/hz and a fresh output path are required')
    report = collect(args.pid, args.name, args.seconds, args.hz)
    args.out.write_text(json.dumps(report, separators=(',', ':')) + '\n')
    print(json.dumps({key: report[key] for key in
                      ('terminal', 'sample_count', 'matched_thread_rows',
                       'read_failures', 'overruns')}))
    return 0 if (report['terminal'] == 'completed' and report['sample_count']
                 and report['matched_thread_rows']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
