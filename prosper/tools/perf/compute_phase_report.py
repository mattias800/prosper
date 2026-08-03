#!/usr/bin/env python3
"""Aggregate `[compute-phase]` lines into a compute-side frame decomposition.

WHY THIS EXISTS
---------------
`PROSPER_COMPUTE_PHASE_TIMING=1` already emits a complete 17-timer decomposition of every compute
dispatch -- but it emits it *per dispatch*, one line each. A routed title produces tens of thousands
of those lines, so the instrument that knows where compute time goes has, in practice, been unreadable
at run scale. `PROSPER_RENDER_TIMING` is readable precisely because it rolls its phases up; this tool
gives the compute side the same shape, offline, without touching the emulator or perturbing a run.

Reads a run log (stdin or a path), parses `[compute-phase]` records, and prints:

  * the phase table -- ms, share of total, and mean ms/dispatch -- nested so each child sits under
    the parent whose interval contains it;
  * "unattributed" rows, i.e. parent time its own children do not explain (this is where an
    unmeasured cost hides, so it is printed rather than silently dropped);
  * real image bindings ranked by stable shader hash + binding + class, with persistence/upload-skip
    state and a bounded list of observed guest addresses;
  * the programs that cost the most, each with its dominant leaf phase.

`setup_ms` deliberately spans the image-binding loop as well as descriptor validation and buffer
binding, and that loop has no sub-timer on the `[compute-phase]` line -- so on an image-heavy title
most of `setup` lands in its "unattributed" row. Add `PROSPER_COMPUTE_IMAGE_TIMING=1` to the run and
this tool will also roll up the resulting `[compute-image]` records, which decompose exactly that
interval. Sampled cache lookup is a sibling of upload preparation, while storage cache validation is
nested inside `prepare_ms`; the report preserves that hierarchy and warns when a record violates it.
Run both switches together whenever setup is the dominant phase.

WHAT THIS TOOL DOES NOT SEE
---------------------------
`[compute-phase]` is emitted from `execute_item`, so it covers **backend-executed dispatches only**.
Dispatches that match the CPU fast path return before `execute_item` and emit no record at all -- on
Astro Bot that is 32,667 of 139,151, nearly a quarter. Any "N % of dispatches" statement derived from
this tool is therefore a share of backend-executed dispatches, not of the guest's dispatches. Read the
run log's `[render-timing] compute_cpu_fast fills=N` line and add it to the denominator before quoting
a rate.

Usage:
    compute_phase_report.py [LOG ...] [--top N] [--since-submit N] [--program 0xADDR] [--csv]
"""

import argparse
import re
import sys
from collections import defaultdict

RECORD = re.compile(r"\[compute-phase\]\s+(.*)")
IMAGE_RECORD = re.compile(r"\[compute-image\]\s+(.*)")
FIELD = re.compile(r"([A-Za-z_][\w-]*)=([^\s]+)")
HEX_VALUE = re.compile(r"0x[0-9a-fA-F]+$")
DECIMAL_VALUE = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)$")
MAX_BINDING_ADDRESSES = 4

# `[compute-image]` decomposes one image binding inside the setup_ms interval. Several of its field
# names (cache_ms, prepare_ms) collide with `[compute-phase]` fields that mean something else
# entirely, so the two record types are parsed separately and never merged into one dictionary.
IMAGE_TOP_LEVEL_PHASES = [
    ("query (driver caps)", "query_ms"),
    ("import (renderer RTT)", "import_ms"),
    ("staging alloc", "staging_ms"),
    ("image allocation", "allocation_ms"),
    ("view", "view_ms"),
    ("sampler", "sampler_ms"),
]
IMAGE_TIMER_KEYS = [key for _, key in IMAGE_TOP_LEVEL_PHASES] + ["cache_ms", "prepare_ms"]

# (label, key, parent) -- parent is the phase whose measured interval contains this one.
# `total_ms` is the root; the top-level five partition it by construction in execute_item().
PHASES = [
    ("setup",              "setup_ms",              "total_ms"),
    ("  validate",         "setup_validate_ms",     "setup_ms"),
    ("  buffers",          "setup_buffers_ms",      "setup_ms"),
    ("pipeline",           "pipeline_ms",           "total_ms"),
    ("dispatch (GPU)",     "dispatch_ms",           "total_ms"),
    ("writeback",          "writeback_ms",          "total_ms"),
    ("  prepare",          "writeback_prepare_ms",  "writeback_ms"),
    ("  buffers",          "writeback_buffers_ms",  "writeback_ms"),
    ("  images",           "writeback_images_ms",   "writeback_ms"),
    ("    map",            "map_ms",                "writeback_images_ms"),
    ("    prepare",        "prepare_ms",            "writeback_images_ms"),
    ("    watch",          "watch_ms",              "writeback_images_ms"),
    ("    pack",           "pack_ms",               "writeback_images_ms"),
    ("    layout(retile)",  "layout_ms",            "writeback_images_ms"),
    ("    notify",         "notify_ms",             "writeback_images_ms"),
    ("    cache",          "cache_ms",              "writeback_images_ms"),
    ("  publish",          "writeback_publish_ms",  "writeback_ms"),
    ("cleanup",            "cleanup_ms",            "total_ms"),
]
# Leaf phases are the ones an optimisation can actually target; a parent is just their sum.
LEAVES = [k for (_, k, _) in PHASES if k not in {p for (_, _, p) in PHASES}]


def _fields(text):
    out = {}
    for key, value in FIELD.findall(text):
        key = key.replace("-", "_")
        if HEX_VALUE.fullmatch(value):
            out[key] = int(value, 16)
        elif DECIMAL_VALUE.fullmatch(value):
            out[key] = float(value)
        else:
            # `[compute-image]` uses text fields for the binding class and extent. Retain unknown
            # future tokens too: silently dropping a discriminator is the unsafe compatibility mode.
            out[key] = value
    return out


def _storage_image(image):
    return image.get("class") == "storage"


def _image_root_time(image):
    """Sum only siblings inside one image's `ms` interval.

    Sampled-image cache lookup finishes before prepare starts. Storage-image cache lookup starts after
    prepare starts, so adding both `cache_ms` and `prepare_ms` double-counts storage bindings.
    """
    total = sum(image.get(key, 0.0) for _, key in IMAGE_TOP_LEVEL_PHASES)
    total += image.get("prepare_ms", 0.0)
    if not _storage_image(image):
        total += image.get("cache_ms", 0.0)
    return total


def _known_tally(yes, known, total):
    if not known:
        return "-"
    if known == total:
        return str(yes)
    return f"{yes}+?{total - known}"


def _known_gib(byte_total, known, total):
    if not known:
        return "-"
    suffix = "" if known == total else "+?"
    return f"{byte_total / (1 << 30):.2f}{suffix}"


def parse(streams, since_submit, program):
    records, images = [], []
    for stream in streams:
        for line in stream:
            match = RECORD.search(line)
            if match:
                fields = _fields(match.group(1))
                if "total_ms" not in fields:
                    continue
                if since_submit is not None and fields.get("submit", 0) < since_submit:
                    continue
                if program is not None and fields.get("code") != program:
                    continue
                records.append(fields)
                continue
            match = IMAGE_RECORD.search(line)
            if match:
                fields = _fields(match.group(1))
                # `ms` is this binding's whole interval; without it the record cannot be attributed.
                if "ms" not in fields:
                    continue
                if program is not None and fields.get("code") != program:
                    continue
                # NOTE: `[compute-image]` carries no `submit=` field, so --since-submit CANNOT be
                # applied here. main() suppresses the image section rather than mixing a filtered
                # dispatch denominator with an unfiltered image numerator.
                images.append(fields)
    return records, images


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", help="run logs (default: stdin)")
    ap.add_argument("--top", type=int, default=12, help="programs to list (default 12)")
    ap.add_argument("--since-submit", type=int, default=None,
                    help="ignore dispatches before this submit number (skips boot/warmup)")
    ap.add_argument("--program", type=lambda s: int(s, 0), default=None,
                    help="restrict to one program code address")
    ap.add_argument("--csv", action="store_true", help="emit the phase table as CSV")
    args = ap.parse_args()

    streams = []
    if args.logs:
        try:
            streams = [open(p, "r", errors="replace") for p in args.logs]
        except OSError as error:
            print(f"cannot read log: {error}", file=sys.stderr)
            return 2
    else:
        streams = [sys.stdin]
    records, images = parse(streams, args.since_submit, args.program)
    # `[compute-image]` has no submit ordinal, so it cannot honour --since-submit. Keeping it would
    # divide a whole-log image numerator by a filtered dispatch denominator: with a filter that
    # skips 10 % of a route every image row inflates ~11 % and still looks entirely plausible.
    images_suppressed_by_filter = bool(images) and args.since_submit is not None
    if images_suppressed_by_filter:
        images = []

    if not records:
        # Name the filters when they are set. "No records" reads as "the switch was off", and
        # sending someone to re-run a 10-minute route because --since-submit was too high is exactly
        # the kind of misdirection this tool exists to avoid.
        filters = []
        if args.since_submit is not None:
            filters.append(f"--since-submit {args.since_submit}")
        if args.program is not None:
            filters.append(f"--program 0x{args.program:x}")
        if filters:
            print(f"no [compute-phase] records matched {' and '.join(filters)} "
                  f"(records may exist outside that filter)", file=sys.stderr)
        else:
            print("no [compute-phase] records found "
                  "(run with PROSPER_COMPUTE_PHASE_TIMING=1)", file=sys.stderr)
        return 1

    # A dispatch that fails leaves execute_item() through an early break, so phase_dispatch and
    # phase_writeback are never advanced past phase_start. Its `dispatch_ms` is then computed
    # backwards (and prints NEGATIVE) while `cleanup_ms` swallows the whole dispatch. Those records
    # carry a valid `total_ms` and nothing else, so they are reported separately and never summed
    # into the phase table -- mixing them in silently inverts it. See #1732.
    # A record with no `ok` field at all is a truncated or interleaved line. Default it to FAILED,
    # not succeeded: the whole point of this split is that mixing a broken record into the phase
    # table inverts it, so the unsafe default must be the one that keeps it out.
    failed = [r for r in records if r.get("ok", 1) == 0 or "ok" not in r]
    records = [r for r in records if r.get("ok", 1) != 0 and "ok" in r]
    if not records:
        print(f"all {len(failed)} dispatches in this log failed; no phase decomposition is "
              f"available (their sub-timers are not meaningful)", file=sys.stderr)
        return 1

    totals = defaultdict(float)
    for rec in records:
        for _, key, _ in PHASES:
            totals[key] += rec.get(key, 0.0)
        totals["total_ms"] += rec["total_ms"]

    n = len(records)
    grand = totals["total_ms"]
    submits = {int(r["submit"]) for r in records if "submit" in r}
    programs = {int(r["code"]) for r in records if "code" in r}

    # Self-checks. This tool encodes a model of execute_item()'s phase structure, and that model can
    # silently stop matching the emitter -- a new phase, a moved boundary, a renamed field. Both
    # checks below hold by construction today (verified across 18,933 real records), so a violation
    # means the model is stale and the table is fiction. Warn rather than print it silently: a
    # decomposition with an impossible row is worse than no decomposition.
    #
    # 1. Every phase of a successful dispatch is a forward interval.
    # Collected, not just printed: these also become a banner INSIDE the table. stderr alone is not
    # enough -- `report.py run.log > table.txt`, or pasting stdout into a PR body, discards exactly
    # the warning while keeping a table that still looks authoritative, and the durable artifact is
    # where the damage lands (trap 45's shape).
    model_warnings = []
    negative = [k for (_, k, _) in PHASES if totals[k] < -1e-6]
    if negative:
        model_warnings.append(
            f"negative time in {', '.join(negative)} on ok=1 records; the phase boundaries are "
            f"not all being set")
    # 2. The top-level phases partition total_ms, and no child exceeds its parent. Checked per
    #    record against the printed 2-decimal precision, then reported as a count.
    tolerance = 0.06
    top = [k for (_, k, parent) in PHASES if parent == "total_ms"]
    broken_sum = sum(
        1 for r in records
        if abs(sum(r.get(k, 0.0) for k in top) - r["total_ms"]) > tolerance)
    broken_nest = 0
    for parent in {p for (_, _, p) in PHASES} - {"total_ms"}:
        kids = [k for (_, k, p) in PHASES if p == parent]
        broken_nest += sum(
            1 for r in records
            if sum(r.get(k, 0.0) for k in kids) > r.get(parent, 0.0) + tolerance)
    if broken_sum or broken_nest:
        model_warnings.append(
            f"this tool's phase model does not match these records ({broken_sum} where the "
            f"top-level phases do not sum to total_ms, {broken_nest} where children exceed their "
            f"parent); the emitter in live_compute.cpp has probably changed")

    # `[compute-image]` prints at three-decimal precision. Check its hierarchy per record before
    # aggregating: a negative residual in the final table must be visibly identified as an invalid
    # model, not presented as a meaningful performance result. Aliases have no sub-timers and are
    # intentionally outside this model.
    image_tolerance = 0.006
    model_images = [image for image in images if not image.get("alias", 0.0)]
    broken_storage_nest = sum(
        1 for image in model_images
        if _storage_image(image) and
        image.get("cache_ms", 0.0) > image.get("prepare_ms", 0.0) + image_tolerance)
    broken_image_root = sum(
        1 for image in model_images
        if _image_root_time(image) > image["ms"] + image_tolerance)
    negative_image_timers = sum(
        1 for image in model_images
        if any(image.get(key, 0.0) < -image_tolerance for key in IMAGE_TIMER_KEYS))
    if broken_storage_nest or broken_image_root or negative_image_timers:
        model_warnings.append(
            f"this tool's image model does not match these records ({broken_storage_nest} storage "
            f"cache intervals exceed prepare_ms, {broken_image_root} where top-level image children "
            f"exceed ms, {negative_image_timers} with a negative image timer); storage cache must "
            f"remain nested inside prepare upload")
    for warning in model_warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    if args.csv:
        # CSV is the output MOST likely to be redirected, so the model warning must ride along here
        # too -- a spreadsheet built from a table whose model is stale carries no trace of it.
        for warning in model_warnings:
            print(f"# NOT TRUSTWORTHY,{warning}")
        print("phase,key,ms,pct,ms_per_dispatch")
        for label, key, _ in PHASES:
            print(f"{label.strip()},{key},{totals[key]:.3f},"
                  f"{100.0 * totals[key] / grand if grand else 0:.2f},{totals[key] / n:.4f}")
        return 0

    print(f"compute decomposition: {n} succeeded dispatches, {len(submits)} submits, "
          f"{len(programs)} programs, {grand:.0f} ms total")
    print(f"  mean {grand / n:.3f} ms/dispatch"
          + (f", {n / len(submits):.1f} dispatches/submit" if submits else ""))
    if failed:
        failed_ms = sum(r["total_ms"] for r in failed)
        share = 100.0 * len(failed) / (len(failed) + n)
        print(f"  EXCLUDED: {len(failed)} FAILED dispatches ({share:.0f}% of the "
              f"{len(failed) + n} that reached execute_item), {failed_ms:.0f} ms wall. Their phase "
              f"split is not recoverable -- a failed dispatch leaves execute_item early, so the "
              f"phase spanning the break prints NEGATIVE and cleanup_ms absorbs the whole record.")
        print("  That share is NOT the fraction of all guest dispatches: CPU-fast-path fills return "
              "before execute_item and emit no record here. Take their count from the run log's "
              "'[render-timing] compute_cpu_fast fills=N' line and add it to the denominator.")
        by_program = defaultdict(int)
        for r in failed:
            by_program[int(r.get("code", 0))] += 1
        worst = sorted(by_program.items(), key=lambda kv: -kv[1])[:5]
        print("  worst failing programs: "
              + ", ".join(f"0x{c:x} ({k})" for c, k in worst))
    print()
    print(f"  {'phase':<20}{'ms':>12}{'% total':>10}{'ms/disp':>12}")
    print("  " + "-" * 54)

    # Walk the tree so each parent is followed by its children and then by the part of its interval
    # the children do not explain. An unattributed remainder is a real measurement, not noise: it is
    # time inside the parent interval that no sub-timer covers, and it is exactly where an unmeasured
    # cost would sit -- so it is printed rather than silently dropped.
    children = defaultdict(list)
    label_of = {}
    for label, key, parent in PHASES:
        children[parent].append(key)
        label_of[key] = label

    def row(label, ms, depth):
        pct = 100.0 * ms / grand if grand else 0.0
        print(f"  {'  ' * depth + label:<20}{ms:>12.1f}{pct:>9.1f}%{ms / n:>12.3f}")

    def walk(key, depth):
        for child in children.get(key, []):
            row(label_of[child].strip(), totals[child], depth)
            walk(child, depth + 1)
        kids = children.get(key)
        if kids:
            rest = totals[key] - sum(totals[k] for k in kids)
            # Relative to the PARENT, not to the grand total: a small parent (writeback_images_ms)
            # would otherwise lose its remainder silently, which is the failure this row exists for.
            parent_ms = totals[key]
            if parent_ms and abs(rest) / parent_ms > 0.01:
                row("unattributed", rest, depth)

    walk("total_ms", 0)

    print("  " + "-" * 54)
    print(f"  {'total':<20}{grand:>12.1f}{100.0:>9.1f}%{grand / n:>12.3f}")
    for warning in model_warnings:
        print(f"  !! NOT TRUSTWORTHY: {warning}. Update the report model before using this table.")

    # The image-binding loop sits inside setup_ms and has no `[compute-phase]` sub-timer, so on an
    # image-heavy title it IS setup's unattributed row. Break it out when the run carried
    # PROSPER_COMPUTE_IMAGE_TIMING; say so explicitly when it did not, because a missing section
    # here reads far too easily as "there was nothing there".
    print()
    # An aliased binding is emitted with only `ms` -- no sub-timers and no byte counts -- because it
    # folded into an earlier binding and did no work. Counting those in `len(images)` divides every
    # ms/binding by the wrong denominator (they are 64 % of Astro Bot's records) and dumps their whole
    # `ms` into `unattributed`. Report them, do not average over them. Note this section must never
    # `return`: the top-programs table below is independent of it, and an all-alias log is ordinary.
    aliases = [i for i in images if i.get("alias", 0.0)]
    images = [i for i in images if not i.get("alias", 0.0)]
    if not images and aliases:
        print(f"  setup image bindings: all {len(aliases)} records are aliased folds "
              f"(no work, no sub-timers); nothing to decompose.")
    elif images:
        image_total = sum(i["ms"] for i in images)
        # `[compute-image]` carries no ok flag, so these records span FAILED dispatches as well --
        # a dispatch that fails has usually already bound its images. Denominating them against the
        # succeeded-only total would therefore overstate their share (and can exceed 100%). Use the
        # wall time of every dispatch, and label it, so the two sections are not silently on
        # different bases.
        base = grand + sum(r["total_ms"] for r in failed)
        pct = lambda ms: (100.0 * ms / base) if base else 0.0
        print(f"  setup image bindings: {len(images)} real bindings, {image_total:.0f} ms "
              f"({pct(image_total):.1f}% of ALL dispatch wall, succeeded + failed)"
              + (f"; {len(aliases)} further records were aliased folds "
                 f"({sum(i['ms'] for i in aliases):.0f} ms), excluded" if aliases else ""))
        print(f"  {'sub-phase':<26}{'ms':>12}{'% all':>10}{'ms/binding':>13}")
        print("  " + "-" * 61)
        def image_row(label, ms, depth=0):
            print(f"  {'  ' * depth + label:<26}{ms:>12.1f}{pct(ms):>9.1f}%"
                  f"{ms / len(images):>13.3f}")

        image_totals = {
            key: sum(image.get(key, 0.0) for image in images)
            for key in IMAGE_TIMER_KEYS
        }
        storage_cache = sum(
            image.get("cache_ms", 0.0) for image in images if _storage_image(image))
        sampled_cache = image_totals["cache_ms"] - storage_cache

        image_row("query (driver caps)", image_totals["query_ms"])
        image_row("import (renderer RTT)", image_totals["import_ms"])
        image_row("cache lookup (sampled)", sampled_cache)
        image_row("staging alloc", image_totals["staging_ms"])
        image_row("prepare upload (inclusive)", image_totals["prepare_ms"])
        image_row("storage cache (included)", storage_cache, 1)
        image_row("prepare exclusive", image_totals["prepare_ms"] - storage_cache, 1)
        image_row("image allocation", image_totals["allocation_ms"])
        image_row("view", image_totals["view_ms"])
        image_row("sampler", image_totals["sampler_ms"])

        # Only top-level siblings are attributed against the binding interval. Storage cache is a
        # child of prepare and is printed for attribution without being added a second time.
        attributed = sum(_image_root_time(image) for image in images)
        rest = image_total - attributed
        image_row("unattributed", rest)
        print("  " + "-" * 61)
        guest_bytes = sum(i.get("guest", 0.0) for i in images)
        staging_bytes = sum(i.get("staging", 0.0) for i in images)
        imported = sum(1 for i in images if i.get("imported", 0.0))
        print(f"  guest bytes bound {guest_bytes / (1 << 30):.2f} GiB, "
              f"staged {staging_bytes / (1 << 30):.2f} GiB, "
              f"{imported} bindings imported from the renderer")
    elif images_suppressed_by_filter:
        print("  setup image bindings: SUPPRESSED because --since-submit is set.")
        print("  [compute-image] records carry no submit ordinal, so they cannot be filtered to")
        print("  match; showing them would divide a whole-log numerator by a filtered denominator.")
        print("  Re-run without --since-submit for the image decomposition.")
    else:
        print("  setup image bindings: no [compute-image] records in this log.")
        print("  Re-run with PROSPER_COMPUTE_IMAGE_TIMING=1 to decompose setup's unattributed row;")
        print("  its absence here does NOT mean image binding was free.")

    # Aggregate real bindings by the cross-run shader identity when available. A run-local code
    # address remains the compatibility fallback for old logs. Addresses are evidence about variants,
    # not part of the group key: Astro's storage binding alternates backing addresses while remaining
    # one shader/binding cost centre. Keep only a bounded set so a hostile or corrupt log cannot make
    # this diagnostic allocate without limit.
    if images:
        binding_groups = {}
        for image in images:
            if "hash" in image:
                identity = ("hash", int(image["hash"]))
            else:
                identity = ("code", int(image.get("code", 0)))
            binding = int(image["binding"]) if "binding" in image else None
            image_class = str(image.get("class", "unknown"))
            key = (identity, binding, image_class)
            group = binding_groups.setdefault(key, {
                "binding": binding, "class": image_class,
                "n": 0, "ms": 0.0,
                "persistent_yes": 0, "persistent_known": 0,
                "upload_yes": 0, "upload_known": 0,
                "guest": 0.0, "guest_known": 0,
                "staging": 0.0, "staging_known": 0,
                "addresses": set(), "addresses_known": 0, "addresses_overflow": False,
            })
            group["n"] += 1
            group["ms"] += image["ms"]
            if "persistent" in image:
                group["persistent_known"] += 1
                group["persistent_yes"] += bool(image["persistent"])
            if "upload_skipped" in image:
                group["upload_known"] += 1
                group["upload_yes"] += bool(image["upload_skipped"])
            if "guest" in image:
                group["guest_known"] += 1
                group["guest"] += image["guest"]
            if "staging" in image:
                group["staging_known"] += 1
                group["staging"] += image["staging"]
            if "addr" in image:
                group["addresses_known"] += 1
                address = int(image["addr"])
                if address not in group["addresses"]:
                    if len(group["addresses"]) < MAX_BINDING_ADDRESSES:
                        group["addresses"].add(address)
                    else:
                        group["addresses_overflow"] = True

        ranked_bindings = sorted(
            binding_groups.items(), key=lambda item: -item[1]["ms"])[:args.top]
        print()
        print(f"  top {len(ranked_bindings)} real image binding groups by setup cost")
        print(f"  {'shader':<23}{'bind':>6} {'class':<8}{'n':>7}{'ms':>11}{'mean':>9}"
              f"{'persist':>11}{'skip':>9}{'guestGiB':>11}{'stageGiB':>11}  addresses")
        print("  " + "-" * 126)
        for (identity, _, _), group in ranked_bindings:
            identity_kind, identity_value = identity
            shader = (f"hash:0x{identity_value:016x}" if identity_kind == "hash"
                      else f"code:0x{identity_value:x}")
            address_text = ",".join(f"0x{address:x}" for address in sorted(group["addresses"]))
            if group["addresses_overflow"]:
                address_text += ("," if address_text else "") + "+more"
            if group["addresses_known"] and group["addresses_known"] != group["n"]:
                address_text += ("," if address_text else "") + "+?"
            if not address_text:
                address_text = "-"
            binding_text = str(group["binding"]) if group["binding"] is not None else "-"
            print(f"  {shader:<23}{binding_text:>6} {group['class']:<8}{group['n']:>7}"
                  f"{group['ms']:>11.1f}{group['ms'] / group['n']:>9.3f}"
                  f"{_known_tally(group['persistent_yes'], group['persistent_known'], group['n']):>11}"
                  f"{_known_tally(group['upload_yes'], group['upload_known'], group['n']):>9}"
                  f"{_known_gib(group['guest'], group['guest_known'], group['n']):>11}"
                  f"{_known_gib(group['staging'], group['staging_known'], group['n']):>11}  "
                  f"{address_text}")

    # Which program to attack, and which leaf inside it. Ranking programs by *total* cost (not mean)
    # is deliberate: a cheap kernel dispatched thousands of times outranks an expensive rare one.
    per_program = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    for rec in records:
        code = int(rec.get("code", 0))
        counts[code] += 1
        for _, key, _ in PHASES:
            per_program[code][key] += rec.get(key, 0.0)
        per_program[code]["total_ms"] += rec["total_ms"]

    ranked = sorted(per_program.items(), key=lambda kv: -kv[1]["total_ms"])[:args.top]
    print()
    print(f"  top {len(ranked)} programs by total cost")
    print(f"  {'code':<16}{'n':>8}{'ms':>12}{'% total':>9}{'mean_ms':>10}  dominant leaf")
    print("  " + "-" * 76)
    for code, phase in ranked:
        dominant = max(LEAVES, key=lambda k: phase[k])
        # Use the timer key, not the display label: "buffers" appears under both setup and
        # writeback, and naming the wrong one sends the reader to the wrong code path.
        label = dominant[:-3]
        print(f"  0x{code:<14x}{counts[code]:>8}{phase['total_ms']:>12.1f}"
              f"{100.0 * phase['total_ms'] / grand if grand else 0:>8.1f}%"
              f"{phase['total_ms'] / counts[code]:>10.3f}  {label} "
              f"({100.0 * phase[dominant] / phase['total_ms'] if phase['total_ms'] else 0:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
