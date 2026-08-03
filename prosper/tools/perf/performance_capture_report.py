#!/usr/bin/env python3
"""Inspect a bounded prosper F8 `.prperf` performance capture.

The capture is evidence, not an oracle. This report separates cheap process/frame-pacing counters
from post-trigger renderer/compute timings and says "inconclusive" when the required population is
missing. It never interprets a dropped/capped record count as the real event count.
"""

import argparse
import json
import math
import sys


class CaptureError(ValueError):
    pass


def load_capture(path):
    records = []
    try:
        with open(path, "r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise CaptureError(f"line {line_number}: invalid JSON: {exc}") from exc
    except OSError as exc:
        raise CaptureError(str(exc)) from exc
    validate_capture(records)
    return records


def validate_capture(records):
    if not records:
        raise CaptureError("empty capture")
    header = records[0]
    if (header.get("type"), header.get("format"), header.get("version")) != (
            "header", "prosper-performance-capture", 1):
        raise CaptureError("not a supported prosper performance capture")
    footers = [record for record in records if record.get("type") == "footer"]
    if len(footers) != 1 or not footers[0].get("complete"):
        raise CaptureError("capture is incomplete (missing complete footer)")
    footer = footers[0]
    actual = {
        "pre_samples": sum(r.get("type") == "sample" and r.get("phase") == "pre" for r in records),
        "post_samples": sum(r.get("type") == "sample" and r.get("phase") == "post" for r in records),
        "renderer_records": sum(r.get("type") == "renderer" for r in records),
        "compute_records": sum(r.get("type") == "compute" for r in records),
    }
    for key, value in actual.items():
        if footer.get(key) != value:
            raise CaptureError(f"footer {key}={footer.get(key)!r}, but file contains {value}")
    for key in ("renderer_dropped", "compute_dropped"):
        if not isinstance(footer.get(key), int) or footer[key] < 0:
            raise CaptureError(f"footer has invalid {key}")


def _counter_rate(samples, field, seconds):
    if len(samples) < 2 or seconds <= 0:
        return None
    first, last = samples[0].get(field), samples[-1].get(field)
    if first is None or last is None or last < first:
        return None
    return (last - first) / seconds


def _total(records, field):
    return sum(float(record.get(field, 0.0)) for record in records
               if math.isfinite(float(record.get(field, 0.0))))


def summarize(records):
    header = records[0]
    footer = next(record for record in records if record.get("type") == "footer")
    post = sorted((record for record in records
                   if record.get("type") == "sample" and record.get("phase") == "post"),
                  key=lambda record: record.get("t_ns", 0))
    renderer = [record for record in records if record.get("type") == "renderer"]
    compute = [record for record in records if record.get("type") == "compute"]

    seconds = None
    if len(post) >= 2 and post[-1].get("t_ns", 0) > post[0].get("t_ns", 0):
        seconds = (post[-1]["t_ns"] - post[0]["t_ns"]) / 1e9
    cpu_cores = _counter_rate(post, "process_cpu_ns", seconds or 0) if seconds else None
    if cpu_cores is not None:
        cpu_cores /= 1e9

    rates = {
        "guest_fps": _counter_rate(post, "guest_presents", seconds or 0) if seconds else None,
        "rendered_fps": _counter_rate(post, "rendered_frames", seconds or 0) if seconds else None,
        "host_fps": _counter_rate(post, "host_presented_frames", seconds or 0) if seconds else None,
    }
    rss = [record.get("rss_bytes") for record in post if record.get("rss_bytes") is not None]

    graphics_total = _total(renderer, "total_ms")
    compute_total = _total(compute, "total_ms")
    measured_total = graphics_total + compute_total
    gpu_wait_total = _total(renderer, "gpu_wait_ms")
    gpu_timestamp_samples = sum(int(record.get("gpu_timestamp_samples", 0))
                                for record in renderer)
    # A zero device duration has meaning only when Vulkan actually returned timestamps. If any
    # record paid a GPU wait without timestamp samples, keep the whole wait unsplit rather than
    # manufacturing "overhead = wait - 0" from unavailable device evidence.
    gpu_timestamps_available = gpu_timestamp_samples > 0 and all(
        float(record.get("gpu_wait_ms", 0.0)) <= 0 or
        int(record.get("gpu_timestamp_samples", 0)) > 0
        for record in renderer)
    gpu_device_total = _total(renderer, "gpu_device_ms") if gpu_timestamps_available else None
    gpu_wait_overhead = (max(0.0, gpu_wait_total - gpu_device_total)
                         if gpu_timestamps_available else None)
    components = {
        # setup_resources is nested in backend, while build_resources is the frontend materializer.
        "renderer-resource": _total(renderer, "build_resources_ms") +
                             _total(renderer, "setup_resources_ms"),
        "gpu-wait": gpu_wait_total,
        "gpu-device": gpu_device_total,
        "gpu-wait-overhead": gpu_wait_overhead,
        "readback": _total(renderer, "readback_ms"),
        "compute": compute_total,
    }
    classification_components = {
        "renderer-resource": components["renderer-resource"],
        "readback": components["readback"],
        "compute": components["compute"],
    }
    if gpu_timestamps_available:
        classification_components["gpu-device"] = gpu_device_total
        classification_components["gpu-wait-overhead"] = gpu_wait_overhead
    else:
        classification_components["gpu-wait"] = gpu_wait_total

    classification = "inconclusive"
    reason = "the capture does not contain enough post-trigger process and timing data"
    wall_ms = (seconds or 0) * 1000.0
    if measured_total > 0:
        largest, cost = max(classification_components.items(), key=lambda item: item[1])
        share = cost / measured_total
        if share >= 0.40:
            classification = largest
            reason = f"{largest} is the largest measured component ({cost:.1f} ms, {share:.0%})"
        elif cpu_cores is not None and cpu_cores >= 0.80 and wall_ms and measured_total < wall_ms * 0.40:
            classification = "cpu-outside-renderer"
            reason = (f"the process used {cpu_cores:.2f} CPU cores while measured renderer/compute "
                      f"work covered only {measured_total / wall_ms:.0%} of the sampled wall window")
        else:
            reason = "measured work is split across components; no component reaches the 40% evidence threshold"
    elif cpu_cores is not None and cpu_cores >= 0.80:
        classification = "cpu-outside-renderer"
        reason = (f"the process used {cpu_cores:.2f} CPU cores with no retained renderer/compute "
                  "timing records")

    pacing_note = None
    if rates["guest_fps"] is not None and rates["rendered_fps"] is not None:
        if rates["guest_fps"] > max(1.0, rates["rendered_fps"] * 1.5):
            pacing_note = ("guest flips materially outpace rendered frames; the capture proves a "
                           "production/presentation gap but does not assign its cause")
    if footer["renderer_dropped"] or footer["compute_dropped"]:
        truncation = (f"detail truncated: renderer dropped {footer['renderer_dropped']}, "
                      f"compute dropped {footer['compute_dropped']}")
    else:
        truncation = "detail not truncated"

    return {
        "title_id": header.get("title_id", ""),
        "title": header.get("title", ""),
        "revision": header.get("revision", "unknown"),
        "seconds": seconds,
        "cpu_cores": cpu_cores,
        "rss_min": min(rss) if rss else None,
        "rss_max": max(rss) if rss else None,
        "rates": rates,
        "graphics_total_ms": graphics_total,
        "compute_total_ms": compute_total,
        "gpu_timestamp_samples": gpu_timestamp_samples,
        "gpu_timestamps_available": gpu_timestamps_available,
        "components": components,
        "classification": classification,
        "reason": reason,
        "pacing_note": pacing_note,
        "truncation": truncation,
        "counts": {
            "pre": footer["pre_samples"],
            "post": footer["post_samples"],
            "renderer": footer["renderer_records"],
            "compute": footer["compute_records"],
        },
    }


def _fmt_rate(value):
    return "unavailable" if value is None else f"{value:.2f}/s"


def print_summary(summary):
    label = summary["title"] or summary["title_id"] or "untitled process"
    print(f"prosper F8 performance capture: {label}")
    print(f"revision: {summary['revision']}")
    counts = summary["counts"]
    print(f"records: pre={counts['pre']} post={counts['post']} "
          f"renderer={counts['renderer']} compute={counts['compute']} ({summary['truncation']})")
    if summary["seconds"] is None:
        print("post sample window: unavailable (fewer than two ordered samples)")
    else:
        print(f"post sample window: {summary['seconds']:.2f} s")
    cpu = summary["cpu_cores"]
    print("process CPU: " + ("unavailable" if cpu is None else f"{cpu:.2f} cores"))
    if summary["rss_min"] is None:
        print("RSS: unavailable on this platform/run")
    else:
        print(f"RSS: {summary['rss_min'] / 2**20:.1f}..{summary['rss_max'] / 2**20:.1f} MiB")
    rates = summary["rates"]
    print(f"rates: guest flips={_fmt_rate(rates['guest_fps'])} "
          f"rendered={_fmt_rate(rates['rendered_fps'])} host-presented={_fmt_rate(rates['host_fps'])}")
    print(f"measured totals: graphics={summary['graphics_total_ms']:.1f} ms "
          f"compute={summary['compute_total_ms']:.1f} ms")
    print(f"GPU timestamps: {summary['gpu_timestamp_samples']} samples " +
          ("(device/wait split available)" if summary["gpu_timestamps_available"] else
           "(device/wait split unavailable)"))
    print("components: " + " ".join(
        f"{key}=unavailable" if value is None else f"{key}={value:.1f}ms"
        for key, value in summary["components"].items()))
    print(f"primary evidence: {summary['classification']} — {summary['reason']}")
    if summary["pacing_note"]:
        print(f"pacing: {summary['pacing_note']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", help="completed .prperf file")
    parser.add_argument("--json", action="store_true", help="emit the derived summary as JSON")
    args = parser.parse_args(argv)
    try:
        summary = summarize(load_capture(args.capture))
    except CaptureError as exc:
        print(f"performance_capture_report: {exc}", file=sys.stderr)
        return 2
    if args.json:
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
