#!/usr/bin/env python3
"""Summarize complete opt-in compute buffer cache snapshots from a runtime log.

Use PROSPER_COMPUTE_BUFFER_TIMING and PROSPER_COMPUTE_BUFFER_CACHE_CENSUS with the program/F8
selectors. Requested residency is not physical memory usage. Access ages are cache-clock
increments, not elapsed time; unchanged last-use only proves inactivity in observed intervals.
Incomplete snapshots are refused instead of implying the omitted owners are reclaimable.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

IDENTITY = ("submit", "dispatch", "order", "code", "hash")
KEY = ("addr", "host-key", "bytes", "semantic", "logical-bytes", "binding-bytes")


def number(row, key):
    value = int(row[key], 0)
    if value < 0:
        raise ValueError(f"negative {key}")
    return value


def summarize(lines):
    snapshots = {}
    owners = defaultdict(list)
    for line in lines:
        prefix = line.split(" ", 1)[0]
        if prefix not in ("[compute-buffer-cache]", "[compute-buffer-cache-owner]"):
            continue
        pairs = [word.split("=", 1) for word in line.split()[1:]]
        row = dict(pairs)
        if len(row) != len(pairs):
            raise ValueError("duplicate record fields")
        identity = tuple(number(row, key) for key in IDENTITY)
        if prefix == "[compute-buffer-cache]":
            if identity in snapshots:
                raise ValueError("duplicate snapshot")
            snapshots[identity] = row
        else:
            owners[identity].append(row)
    if not snapshots or set(owners) - set(snapshots):
        raise ValueError("empty census or orphan owner rows")
    by_key = defaultdict(list)
    clocks = []
    last_uses = {}
    for identity, summary in snapshots.items():
        rows = owners[identity]
        if any(number(summary, k) not in (0, 1) for k in ("ok", "completion-unproven")):
            raise ValueError("invalid summary boolean")
        if summary["phase"] != "before-cleanup":
            raise ValueError("unknown census phase")
        if (number(summary, "complete") != 1 or
                number(summary, "entries") != number(summary, "emitted") or
                number(summary, "emitted") != len(rows)):
            raise ValueError("incomplete snapshot")
        if number(summary, "bytes") != sum(number(r, "primary-allocation-bytes") +
                                           number(r, "result-allocation-bytes") for r in rows):
            raise ValueError("owner charges do not sum to requested residency")
        if number(summary, "bytes") > number(summary, "limit"):
            raise ValueError("requested residency exceeds cache limit")
        if len({tuple(number(r, k) for k in KEY) for r in rows}) != len(rows):
            raise ValueError("duplicate owner key")
        clock = number(summary, "clock")
        if clocks and clock < clocks[-1]:
            raise ValueError("cache clock reset; analyze each context separately")
        clocks.append(clock)
        for row in rows:
            if number(row, "content-valid") not in (0, 1):
                raise ValueError("invalid owner boolean")
            if number(row, "last-use") > clock:
                raise ValueError("owner access lies after snapshot")
            key = tuple(number(row, k) for k in KEY)
            use = number(row, "last-use")
            if key in last_uses and use < last_uses[key]:
                raise ValueError("owner last-use moved backwards")
            last_uses[key] = use
            by_key[key].append((summary, row))
    result = []
    for key, observations in by_key.items():
        uses = [number(r, "last-use") for s, r in observations]
        ages = [number(s, "clock") - number(r, "last-use") for s, r in observations]
        result.append(dict(zip(KEY, key), snapshots=len(observations),
            pinned_snapshots=sum(number(r, "pins") > 0 for s, r in observations),
            observed_last_use_values=len(set(uses)), first_last_use=uses[0], last_last_use=uses[-1],
            min_access_age=min(ages), max_access_age=max(ages),
            primary_bytes=sorted({number(r, "primary-allocation-bytes") for s, r in observations}),
            baseline_bytes=sorted({number(r, "result-allocation-bytes") for s, r in observations})))
    return {"snapshots": len(snapshots), "clock_range": [clocks[0], clocks[-1]],
            "failed_snapshots": sum(number(s, "ok") == 0 for s in snapshots.values()),
            "completion_unproven_snapshots": sum(number(s, "completion-unproven") != 0
                                                 for s in snapshots.values()),
            "entry_counts": sorted({number(s, "entries") for s in snapshots.values()}),
            "cache_limits": sorted({number(s, "limit") for s in snapshots.values()}),
            "total_bytes": sorted({number(s, "bytes") for s in snapshots.values()}),
            "owners": sorted(result, key=lambda r: (-r["primary_bytes"][0], tuple(r[k] for k in KEY)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    try:
        with args.log.open(errors="replace") as stream:
            result = summarize(stream)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
