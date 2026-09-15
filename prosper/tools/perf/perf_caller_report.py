#!/usr/bin/env python3
"""Report recorded libc caller recovery before attributing copy/allocation costs.

Export with: perf script --no-inline -i cpu.data
  -F comm,pid,tid,time,period,event,ip,sym,dso > stacks.txt
Optional loss input: perf script -D -i cpu.data > records.txt
Only cpu-cycles:u/k exports are accepted. This reads files; it never records a game.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

HEADER = re.compile(r"^\s*(\S(?:.*?\S)?)\s+(\d+)/(\d+)\s+(\d+)\.(\d{1,9}):\s+(\d+)\s+(cpu-cycles:[uk]):\s*$")
FRAME = re.compile(r"^\s+([0-9a-fA-F]+)\s+(.+) \((.+)\)\s*$")
UNKNOWN = re.compile(r"^(?:\[unknown\]|\?+|0x[0-9a-fA-F]+)(?:\+0x[0-9a-fA-F]+)?$")
FAMILIES = ("copy", "compare", "fill", "allocation", "free")


def samples(path):
    current = None
    count = 0
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            line = line.rstrip("\r\n")
            match = HEADER.fullmatch(line)
            if match:
                if current is not None:
                    yield current
                comm, pid, tid, seconds, fraction, period, event = match.groups()
                current = dict(comm=comm, pid=int(pid), tid=int(tid), event=event,
                               period=int(period), frames=[],
                               time_ns=int(seconds) * 10**9 + int(fraction.ljust(9, "0")))
                count += 1
                continue
            match = FRAME.fullmatch(line)
            if match:
                if current is None:
                    raise ValueError(f"line {number}: frame without sample")
                ip, symbol, dso = match.groups()
                current["frames"].append(dict(ip=int(ip, 16), symbol=symbol, dso=dso))
            elif line.strip():
                raise ValueError(f"line {number}: unsupported export line")
    if not count:
        raise ValueError("no samples; empty input is not a clean profile")
    yield current


def family(frame, libc_dsos):
    if frame["dso"] not in libc_dsos:
        return None
    symbol = re.sub(r"\+0x[0-9a-fA-F]+$", "", frame["symbol"]).removeprefix("__GI_")
    if re.match(r"_*(?:memcpy|memmove)(?:_|$)", symbol):
        return "copy"
    if re.match(r"_*(?:memcmp|bcmp)(?:_|$)", symbol):
        return "compare"
    if re.match(r"_*memset(?:_|$)", symbol):
        return "fill"
    if re.match(r"(?:__libc_|_int_)?(?:malloc|calloc|realloc|memalign)$|__libc_malloc2$|(?:tcache_get|sysmalloc|_mid_memalign)(?:_|$)", symbol):
        return "allocation"
    if re.match(r"(?:__libc_|_int_)?free(?:_|$)|(?:tcache_put|munmap_chunk)(?:_|$)", symbol):
        return "free"
    return None


def counter():
    return dict(samples=0, period=0)


def add(row, sample):
    row["samples"] += 1
    row["period"] += sample["period"]


def summary():
    return dict(total=counter(), empty=counter(), families={name: dict(
        total=counter(), leaf_only=counter(), no_named_app_ancestor=counter(),
        named_app_ancestor=counter(), unresolved_before_app=counter(),
        suspicious_low_ip=counter(), callers={}) for name in FAMILIES})


def consume(result, sample, app_dsos, libc_dsos):
    add(result["total"], sample)
    frames = sample["frames"]
    if not frames:
        add(result["empty"], sample)
        return
    name = family(frames[0], libc_dsos)
    if name is None:
        return
    row = result["families"][name]
    add(row["total"], sample)
    if len(frames) == 1:
        add(row["leaf_only"], sample)
    if any(f["ip"] < 4096 for f in frames):
        add(row["suspicious_low_ip"], sample)
    caller_index = next((i for i, f in enumerate(frames[1:], 1)
                         if f["dso"] in app_dsos and not UNKNOWN.fullmatch(f["symbol"])
                         and f["ip"] >= 4096), None)
    if caller_index is None:
        add(row["no_named_app_ancestor"], sample)
        return
    add(row["named_app_ancestor"], sample)
    if any(UNKNOWN.fullmatch(f["symbol"]) or f["ip"] < 4096
           for f in frames[1:caller_index]):
        add(row["unresolved_before_app"], sample)
    caller = frames[caller_index]
    key = (caller["symbol"], caller["dso"], caller["ip"])
    add(row["callers"].setdefault(key, counter()), sample)


def finalize(result):
    total = result["total"]["period"]
    for row in result["families"].values():
        denom = row["total"]["period"]
        for name, value in row.items():
            if name != "callers":
                value["percent_family_periods"] = 100 * value["period"] / denom if denom else None
                value["percent_event_periods"] = 100 * value["period"] / total if total else None
        row["callers"] = [dict(symbol=symbol, dso=dso, ip=hex(ip), **value)
                          for (symbol, dso, ip), value in sorted(row["callers"].items(),
                          key=lambda item: (-item[1]["period"], item[0]))]
    return result


def report(path, app_dsos, libc_dsos, records=None):
    if not app_dsos or not libc_dsos or app_dsos & libc_dsos:
        raise ValueError("application and libc selectors must be nonempty and disjoint")
    selector_matches = {dso: dict(frames=0, named_frames=0) for dso in app_dsos | libc_dsos}
    events = {}
    count = 0
    for sample in samples(path):
        count += 1
        for frame in sample["frames"]:
            if frame["dso"] in selector_matches:
                selector_matches[frame["dso"]]["frames"] += 1
                selector_matches[frame["dso"]]["named_frames"] += not bool(UNKNOWN.fullmatch(frame["symbol"]))
        event = events.setdefault(sample["event"], dict(aggregate=summary(), threads={}))
        key = (sample["pid"], sample["tid"])
        thread = event["threads"].setdefault(key, dict(comms=set(), **summary()))
        thread["comms"].add(sample["comm"])
        consume(event["aggregate"], sample, app_dsos, libc_dsos)
        consume(thread, sample, app_dsos, libc_dsos)
    for event in events.values():
        if not event["aggregate"]["total"]["period"]:
            raise ValueError("event has zero aggregate period; no weighted evidence")
        finalize(event["aggregate"])
        event["threads"] = [dict(pid=pid, tid=tid, **finalize(row))
                            for (pid, tid), row in sorted(event["threads"].items())]
        for row in event["threads"]:
            row["comms"] = sorted(row["comms"])
    loss = None
    if records is not None:
        sample_markers = lost_markers = 0
        with Path(records).open(encoding="utf-8") as stream:
            for line in stream:
                sample_markers += bool(re.search(r"\bPERF_RECORD_SAMPLE\(", line))
                lost_markers += bool(re.search(r"\bPERF_RECORD_LOST(?:_SAMPLES)?\b", line))
        if sample_markers != count:
            raise ValueError(f"record/sample count mismatch: {sample_markers} != {count}")
        loss = dict(sample_record_markers=sample_markers, lost_record_markers=lost_markers)
    def sha(p):
        with Path(p).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(schema=1, app_dsos=sorted(app_dsos), libc_dsos=sorted(libc_dsos),
                selector_matches=selector_matches,
                unobserved_selectors=sorted(dso for dso, value in selector_matches.items() if not value["frames"]),
                stacks_sha256=sha(path), records_sha256=sha(records) if records else None,
                observed_record_loss=loss, events=events, limits=[
                    "Nearest named matching recorded app ancestor is not a direct caller; omitted frames cannot be recovered here.",
                    "Unresolved-before-app and suspicious-low-IP counts overlap other categories; heuristics do not validate stacks.",
                    "A named ancestor can still be wrong. Low-IP flags cover addresses below 4096 only, not every invalid frame.",
                    "An unobserved selector may be a wrong path or absent frames; do not treat it as evidence that the selected code did no work.",
                    "Only explicitly selected libc DSOs and listed function families are classified. Other allocator/copy work is excluded.",
                    "Event/thread period denominators stay separate; sampled periods are not elapsed time, calls, bytes or FPS.",
                    "Missing record export means loss unobserved. Marker counts are not counts of lost samples or proof of a complete recording.",
                    "Count agreement does not prove exports share an origin; verify capture identity, build IDs and exporter exit status separately.",
                    "PID/TID grouping is per recording, not identity across launches. This report does not measure recording overhead or worker coverage."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stacks", type=Path)
    parser.add_argument("--app-dso", required=True, action="append")
    parser.add_argument("--libc-dso", required=True, action="append")
    parser.add_argument("--records", type=Path)
    args = parser.parse_args()
    try:
        result = report(args.stacks, set(args.app_dso), set(args.libc_dso), args.records)
    except (OSError, ValueError) as error:
        parser.exit(2, f"perf_caller_report: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
