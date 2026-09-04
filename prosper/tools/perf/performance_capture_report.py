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


def _gpu_present_adopted(post):
    """Was GPU present adopted? True / False / None when the capture cannot say.

    `rendered_frame_counter` returns nullopt when GPU present is adopted rather than substituting the
    app's host-present count, so the PRESENCE OF THE FIELD is the signal -- exactly the rule
    `_resource_breakdown` states below ("Absent and zero are the same number and opposite facts").

    Deliberately NOT derived from `rendered_fps`: that rate is also None for a sample population with
    fewer than two entries, a non-increasing `t_ns`, or a zero window, so an OFFSCREEN capture (whose
    samples carry a real counter) would read as adopted and get told its readback is real work --
    inverting the advice in the one direction this whole note exists to prevent. That inversion was
    live at 92c27046 and is what this function replaces.
    """
    if not post:
        return None
    present = [s.get("rendered_frames") is not None for s in post]
    if all(present):
        return False        # a real counter on every sample => GPU present was NOT adopted
    if not any(present):
        return True         # absent/null throughout => adopted
    return None             # adopted then lost, or never coherent: say so rather than guess


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


def _resource_breakdown(renderer):
    """Decompose renderer-resource, or say the capture cannot support it.

    `renderer-resource` is the largest component in every capture taken so far, and it is TWO LAYERS
    added together: `build_resources_ms` (the frontend materializer) and `setup_resources_ms` (the
    backend binder). Their sub-buckets belong to one layer each and are not parts of one another.
    Subtracting a frontend bucket from the backend total yields a large, plausible, meaningless
    residue -- a mistake made and published once (#2215), which is why this function exists.

    A capture written before the backend sub-buckets were recorded must report them **unavailable**,
    never 0. Absent and zero are the same number and opposite facts: a 0 here reads as "the backend
    did no descriptor work", and the remainder then reads as unattributed work -- exactly the wrong
    conclusion, now printed with authority by a tool. So the presence of the field is what decides,
    not its value.
    """
    if not renderer:
        return None
    have_backend = any("res_texture_ms" in record for record in renderer)
    # The frontend pair was renamed when the layers were separated; accept the old spelling so an
    # older capture still reports the half it does carry.
    def frontend(new, old):
        return _total(renderer, new) if any(new in r for r in renderer) else _total(renderer, old)
    breakdown = {
        "build_resources": _total(renderer, "build_resources_ms"),
        "frontend_texture": frontend("frontend_texture_ms", "texture_ms"),
        "frontend_buffer": frontend("frontend_buffer_ms", "buffer_ms"),
        "setup_resources": _total(renderer, "setup_resources_ms"),
        "backend_available": have_backend,
    }
    # The frontend texture leaf's OWN classes. The renderer has recorded these since #2250 and this
    # report never printed them, so every capture taken since has carried the answer to "which cache
    # outcome is the texture time in?" and no reader could see it. On Stray's title screen the leaf
    # is 1110.1 ms and 930.8 of it -- 84% -- is in a single unnamed residual; printing the classes is
    # what located it. Same rule as the backend leaves: absent must report UNAVAILABLE, never 0.
    breakdown["tex_classes_available"] = all("frontend_tex_rtt_ms" in r for r in renderer)
    if breakdown["tex_classes_available"]:
        for key in ("rtt", "compute", "local", "persist_hit", "persist_reuse", "persist_miss"):
            breakdown["tex_" + key] = _total(renderer, f"frontend_tex_{key}_ms")
        # Split out later than the six above; an older capture folds it into the residual, which is
        # where it was before it had a name.
        breakdown["tex_invalid_available"] = all(
            "frontend_tex_persist_invalid_ms" in r for r in renderer)
        breakdown["tex_persist_invalid"] = (
            _total(renderer, "frontend_tex_persist_invalid_ms")
            if breakdown["tex_invalid_available"] else 0.0)
        # The COUNT of references nothing claimed, paired with the signed millisecond residual
        # below. `performance_capture.hpp` says these two disagree only if the classification is
        # wrong -- which is a cross-check nobody could run while the report printed one of them.
        breakdown["tex_other_n_available"] = all("frontend_tex_other_n" in r for r in renderer)
        breakdown["tex_other_n"] = (
            sum(r.get("frontend_tex_other_n", 0) for r in renderer)
            if breakdown["tex_other_n_available"] else None)
        breakdown["tex_other"] = breakdown["frontend_texture"] - sum(
            breakdown["tex_" + key] for key in
            ("rtt", "compute", "local", "persist_hit", "persist_reuse", "persist_miss",
             "persist_invalid"))
        # The slowest single reference that reached none of the named classes, with the identity
        # needed to explain it. One witness per record; report the worst across the window.
        witness = max(renderer, key=lambda r: r.get("frontend_tex_other_slowest_ms", 0.0), default=None)
        if witness and witness.get("frontend_tex_other_slowest_ms", 0.0) > 0.0:
            breakdown["tex_other_witness"] = {
                "ms": witness["frontend_tex_other_slowest_ms"],
                "addr": witness.get("frontend_tex_other_addr", 0),
                "source_bytes": witness.get("frontend_tex_other_source_bytes", 0),
                "width": witness.get("frontend_tex_other_width", 0),
                "height": witness.get("frontend_tex_other_height", 0),
                "depth": witness.get("frontend_tex_other_depth", 0),
                "format": witness.get("frontend_tex_other_format", 0),
                "components": witness.get("frontend_tex_other_components", 0),
                "tile_mode": witness.get("frontend_tex_other_tile_mode", 0),
                "compute_candidate": witness.get("frontend_tex_other_compute_candidate", False),
                "persistent_candidate": witness.get("frontend_tex_other_persistent_candidate", False),
                "compressed": witness.get("frontend_tex_other_compressed", False),
            }
    if have_backend:
        breakdown.update({
            "res_texture": _total(renderer, "res_texture_ms"),
            "res_buffer": _total(renderer, "res_buffer_ms"),
            "res_buffer_copy": _total(renderer, "res_buffer_copy_ms"),
            "res_descriptor": _total(renderer, "res_descriptor_ms"),
        })
        # Same rule as have_backend, one level down and for the same reason. A capture predating the
        # buffer leaves must report them UNAVAILABLE, never 0 -- with 0 the remainder below would
        # absorb every leaf it is missing and print a large residual, which is precisely the false
        # "unattributed work" reading this partition was added to stop.
        #
        # all(), not any(). Only one site emits these fields today and it sets them together, so the
        # two are equivalent right now -- but they differ exactly when that stops being true, and
        # they fail in opposite directions. any() would report a breakdown for a partially-populated
        # capture, in which _total's .get(field, 0.0) silently contributes 0 for every record missing
        # the field: under-counted, and indistinguishable from a real measurement. all() reports
        # UNAVAILABLE instead. Fail closed, which is this whole function's thesis.
        breakdown["buffer_leaves_available"] = all("res_buffer_create_ms" in r for r in renderer)
        # The same signed-remainder argument, one level down. res_buffer_ms had NO exhaustive
        # partition until now -- only `copy` and (in the stderr window) `acquire`, two candidate
        # mechanisms out of an unknown number. An unmeasured region does not read as unmeasured; it
        # reads as the subject's cost. A diagnostic added inside it billed 305 ms/submit to the
        # renderer and produced a confident, entirely false finding before a single-variable A/B
        # killed it. This remainder is what makes that visible on the first run instead of the third.
        # `acquire` is a stderr-only term, so it is folded into the remainder here rather than
        # invented -- which is why a healthy value is a few ms, not ~0.
        if breakdown["buffer_leaves_available"]:
            breakdown["res_buffer_create"] = _total(renderer, "res_buffer_create_ms")
            breakdown["res_buffer_index_find"] = _total(renderer, "res_buffer_index_find_ms")
            breakdown["res_buffer_index_insert"] = _total(renderer, "res_buffer_index_insert_ms")
            breakdown["res_buffer_hash"] = _total(renderer, "res_buffer_hash_ms")
            breakdown["res_buffer_other"] = (
                breakdown["res_buffer"] - breakdown["res_buffer_copy"]
                - breakdown["res_buffer_create"] - breakdown["res_buffer_index_find"]
                - breakdown["res_buffer_index_insert"]
                - breakdown["res_buffer_hash"])
        # The remainder is reported as TWO numbers, and this is not fussiness -- it is this file's own
        # argument applied to sign instead of presence.
        #
        # A negative remainder means the sub-buckets over-count their parent: a real defect, in the
        # instrument rather than in the renderer. Clamping it to 0.0 does not "surface" that, it emits
        # the single most reassuring line the tool can produce -- `other=0.0` reads as "every
        # millisecond of setup_resources is attributed", which is the BEST possible state. So a broken
        # instrument and a perfect one would print identically, which is exactly the collapse this
        # module exists to prevent one level up ("absent and zero are the same number and opposite
        # facts"). An earlier revision of this function did clamp, with a comment claiming it
        # surfaced the defect.
        raw_other = (breakdown["setup_resources"] - breakdown["res_texture"]
                     - breakdown["res_buffer"] - breakdown["res_descriptor"])
        breakdown["res_other"] = max(0.0, raw_other)
        breakdown["res_over_attributed"] = max(0.0, -raw_other)   # 0 normally; non-zero is a defect
    return breakdown


def _hex64(value):
    return f"0x{value:016x}"


def _compute_program_groups(records, limit=10, address_limit=8):
    groups = {}
    unknown_records = 0
    unknown_dispatches = 0
    unknown_total_ms = 0.0
    for record in records:
        address = record.get("program_addr")
        program_hash = record.get("program_hash")
        total_ms = float(record.get("total_ms", 0.0))
        dispatches = int(record.get("dispatches", 0))
        if not isinstance(address, int) or not isinstance(program_hash, int):
            unknown_records += 1
            unknown_dispatches += dispatches
            unknown_total_ms += total_ms
            continue
        group = groups.setdefault(program_hash, {
            "records": 0, "dispatches": 0, "total_ms": 0.0, "max_ms": 0.0,
            "addresses": set(),
        })
        group["records"] += 1
        group["dispatches"] += dispatches
        group["total_ms"] += total_ms
        group["max_ms"] = max(group["max_ms"], total_ms)
        group["addresses"].add(address)

    ranked = []
    for program_hash, group in groups.items():
        addresses = sorted(group.pop("addresses"))
        group["program_hash"] = _hex64(program_hash)
        group["mean_ms"] = group["total_ms"] / group["records"]
        group["address_count"] = len(addresses)
        group["addresses"] = [_hex64(address) for address in addresses[:address_limit]]
        ranked.append(group)
    ranked.sort(key=lambda group: (-group["total_ms"], group["program_hash"]))
    return {
        "group_count": len(ranked),
        "groups_omitted": max(0, len(ranked) - limit),
        "groups": ranked[:limit],
        "unknown_records": unknown_records,
        "unknown_dispatches": unknown_dispatches,
        "unknown_total_ms": unknown_total_ms,
    }


# The share a component must reach to WIN the classification. Named because two things depend on
# it and it was previously a bare literal in one of them, so neither the dependency nor the value
# was pinned by anything -- moving it passed the whole suite.
CLASSIFICATION_EVIDENCE_SHARE = 0.40

# Readback must reach this share of measured work before the harness-readback note is emitted.
#
# It sits BELOW the evidence bar by construction: the note exists to cover the gap under that bar,
# a readback large enough to mislead but too small to win the verdict. That inequality is the real
# invariant and is what the tests assert.
#
# One quarter of the bar is a CHOSEN fraction, not a derived one -- it came out of review as a
# rationalisation for 0.10 rather than from anything about the data, and pinning it as an equality
# would give a reviewer's off-hand ratio the authority of an invariant. Recorded as the reason the
# value is what it is; the tests deliberately do not enforce the ratio.
#
# WHAT THE TESTS ACTUALLY PIN, measured per-arm rather than by exit code, because the two are not
# the same question and reading only the exit code gets this wrong:
#
#   threshold     failing arm
#   <= 0.020      test_non_readback_verdict_carries_no_readback_note   (2 ms / 2%, PRE-EXISTING)
#   0.021-0.200   none
#   >  0.200      test_readback_note_threshold_is_a_share_not_a_ranking (its 20% assertion)
#
# So the pinned band is (0.020, 0.200] and 0.10 is NOT itself pinned -- it can move anywhere in a
# 10x range with no test objecting. Two corrections worth keeping, because the obvious reading of
# the arms is wrong in both directions:
#   * the LOWER bound is held by the pre-existing 2% test, NOT by the 1% arm added alongside this
#     threshold. That arm passes at 0.011 and at 0.020; it never binds.
#   * the UPPER bound is held by the 20% assertion, NOT by the 31% motivating arm. That arm's share
#     is 0.3120 and it does not fail anywhere in the swept range.
# The 1% and 31% arms still earn their place -- they hold verdict and GPU-present state constant so
# the only variable is share, and the 31% one reproduces the capture that motivated the change --
# but neither is what fixes the band, and a comment claiming otherwise sends the next reader to the
# wrong test when they want to tighten it.
# Deriving this FROM the bar means a bar mutation is a COMPOUND mutation: the threshold moves with
# it, so at a bar of 0.80 or more the threshold leaves the (0.020, 0.200] band its own arms pin and
# readback tests start failing for reasons that have nothing to do with the bar. Harmless for any
# plausible bar value, and worth knowing before reading a wide sweep's failures.
READBACK_NOTE_MIN_SHARE = CLASSIFICATION_EVIDENCE_SHARE / 4


# The note's closing advice depends on whether readback DECIDED the verdict. "before acting on this
# verdict" is right when the verdict IS readback; on a capture whose verdict is a decisive compute or
# renderer-resource share it is wrong advice attached to a correct conclusion, which is worse than no
# advice -- the reader is told to discard a finding the readback had no part in.
# Verdicts that CANNOT flip when the readback is removed. Removing readback scales every other
# component's share up by the same factor, so a component that already won still wins. A verdict
# that exists precisely BECAUSE nothing won ("inconclusive"), or because measured work was small
# against the wall window ("cpu-outside-renderer"), can and does flip -- measured on this change's
# own motivating capture, where dropping the harness readback turns "inconclusive" into "compute
# (1134.0 ms, 53%)". This module already records the same effect from hardware: #3152 saw a verdict
# become renderer-resource with the graphics total halved once the capture ran through a real window.
_VERDICTS_READBACK_CANNOT_FLIP = frozenset({
    "compute", "renderer-resource", "gpu-device", "gpu-wait", "gpu-wait-overhead",
})


def _readback_note_tail(classification):
    if classification == "readback":
        # A full sentence, like the other four. It was a trailing subordinate clause until the
        # callers grew their own terminating period, at which point it rendered as
        # "...through a real window. before acting on this verdict." -- and no assertion could see
        # it, because every arm used `assertIn` on a substring the break left intact.
        return " Readback IS this capture's verdict, so there is nothing else here to act on."
    if classification in _VERDICTS_READBACK_CANNOT_FLIP:
        # Safe to reassure: this verdict survives the readback being removed.
        return (f" The {classification} verdict above does not depend on it: removing the readback "
                "raises every other share, so a component that already won still wins.")
    # Not safe. Saying "the verdict is unaffected" here would invite trust in a verdict the harness
    # may have produced, which is the opposite of what this note exists to prevent.
    return (f" The {classification} verdict above may itself be an artefact of that readback -- it "
            "is not a verdict some component won, so removing the readback can change it.")


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
    compute_programs = _compute_program_groups(compute)
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
        if share >= CLASSIFICATION_EVIDENCE_SHARE:
            classification = largest
            reason = f"{largest} is the largest measured component ({cost:.1f} ms, {share:.0%})"
        elif cpu_cores is not None and cpu_cores >= 0.80 and wall_ms and measured_total < wall_ms * 0.40:
            classification = "cpu-outside-renderer"
            reason = (f"the process used {cpu_cores:.2f} CPU cores while measured renderer/compute "
                      f"work covered only {measured_total / wall_ms:.0%} of the sampled wall window")
        else:
            reason = ("measured work is split across components; no component reaches the "
                      f"{CLASSIFICATION_EVIDENCE_SHARE:.0%} evidence threshold")
    elif cpu_cores is not None and cpu_cores >= 0.80:
        classification = "cpu-outside-renderer"
        reason = (f"the process used {cpu_cores:.2f} CPU cores with no retained renderer/compute "
                  "timing records")

    # A READBACK verdict has two very different meanings, and the capture can tell them apart
    # rather than guess. `rendered_frame_counter` returns nullopt exactly when GPU present was
    # adopted (frontends/prosper-app/present_policy.hpp), so a null rendered-frame population IS the
    # GPU-present signal, serialized on every sample.
    #
    #   * GPU present NOT adopted -- prosper-app fell back to copying every scanout frame to the CPU,
    #     which happens under SDL_VIDEODRIVER=offscreen where the surface needs
    #     VK_EXT_headless_surface. Then readback is the harness, and optimising it is wasted work.
    #     Measured on The Forgotten City (#3152): same title, phase and trigger gave readback
    #     396.9 ms / 51% and "primary evidence: readback" offscreen, against 0.0 ms through a real
    #     window, where the verdict became renderer-resource and the graphics total halved.
    #   * GPU present ADOPTED -- the scanout readback is skipped, so a large `readback_ms` is real
    #     work: non-deferred colour-target readback, storage writeback, or a copy forced by
    #     `authoritative_readback`, which every ordered DMA copy sets. That is a finding to chase,
    #     NOT to dismiss, and saying so is the point of splitting the two cases.
    #
    # Deliberately not blamed on `tools/screenshot`: it cannot produce a capture at all, since only
    # prosper-app arms one. The `PROSPER_TILECENSUS` readback trap (instrument trap 237) is a
    # different instrument that happens to share the mechanism.
    readback_note = None
    # Warn whenever readback is a LEADING component, not only when it wins the classification.
    # A capture where readback is large but sits under the 40% evidence threshold is classified
    # "inconclusive" -- and then prints a big number with no warning attached, which is the exact
    # shape that sends a reader chasing the harness instead of the title. Measured on a Dragon
    # Quest VII capture: readback=977.7ms against compute=1134.0ms, classified "inconclusive",
    # note silent.
    #
    # Ranking is the wrong trigger and was the first attempt: a "is readback the max component"
    # test stays silent at 977.7 vs 1134.0, which is precisely the case that misleads. So the
    # trigger is readback's SHARE of measured work, with a threshold deliberately well below the
    # 40% classification bar -- the whole point is to cover the gap under that bar.
    #
    # See READBACK_NOTE_MIN_SHARE at module scope for the threshold and what pins it.
    readback_ms = classification_components.get("readback")
    material_readback = (
        isinstance(readback_ms, (int, float)) and measured_total > 0 and
        readback_ms / measured_total >= READBACK_NOTE_MIN_SHARE)
    if classification == "readback" or material_readback:
        adopted = _gpu_present_adopted(post)
        if adopted is None:
            readback_note = (
                "this capture cannot say whether GPU present was adopted, so it cannot say whether "
                "the readback is real or the harness copying every scanout frame to the CPU. Check "
                "the run log for a GPU-present surface failure." + _readback_note_tail(classification))
        elif adopted is False:
            readback_note = (
                "GPU present was NOT adopted for this capture, so the frontend copied every scanout "
                "frame to the CPU -- most often prosper-app under SDL_VIDEODRIVER=offscreen, which "
                "needs VK_EXT_headless_surface. This readback is the harness, not the title. "
                "Re-measure through a real window." + _readback_note_tail(classification))
        else:
            readback_note = (
                "GPU present WAS adopted, so scanout readback is skipped and this readback is real "
                "work -- non-deferred colour-target readback, storage writeback, or a copy forced by "
                "authoritative_readback (every ordered DMA copy sets it). Chase it rather than "
                "dismissing it as a harness artifact.")

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
        "compute_programs": compute_programs,
        "gpu_timestamp_samples": gpu_timestamp_samples,
        "gpu_timestamps_available": gpu_timestamps_available,
        "components": components,
        "resource_breakdown": _resource_breakdown(renderer),
        "classification": classification,
        "reason": reason,
        "pacing_note": pacing_note,
        "readback_note": readback_note,
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
    programs = summary["compute_programs"]
    print(f"compute identities: groups={programs['group_count']} "
          f"unknown={programs['unknown_records']} records/{programs['unknown_total_ms']:.1f} ms")
    for group in programs["groups"]:
        addresses = ",".join(group["addresses"])
        if group["address_count"] > len(group["addresses"]):
            addresses += f",+{group['address_count'] - len(group['addresses'])}"
        print(f"  {group['program_hash']} records={group['records']} "
              f"dispatches={group['dispatches']} total={group['total_ms']:.1f} ms "
              f"mean={group['mean_ms']:.2f} ms max={group['max_ms']:.2f} ms "
              f"addresses={addresses}")
    if programs["groups_omitted"]:
        print(f"  ... {programs['groups_omitted']} lower-cost groups omitted")
    print(f"GPU timestamps: {summary['gpu_timestamp_samples']} samples " +
          ("(device/wait split available)" if summary["gpu_timestamps_available"] else
           "(device/wait split unavailable)"))
    print("components: " + " ".join(
        f"{key}=unavailable" if value is None else f"{key}={value:.1f}ms"
        for key, value in summary["components"].items()))
    breakdown = summary.get("resource_breakdown")
    if breakdown:
        # renderer-resource is the largest component in every capture taken so far, and it is two
        # layers added together. Print the decomposition rather than leaving a reader to subtract:
        # the frontend and backend buckets are NOT parts of one another, and subtracting across them
        # yields a large plausible residue that looks like unattributed work. That mistake has been
        # made once already and published (#2215).
        print("  build_resources (frontend materializer): "
              f"{breakdown['build_resources']:.1f}ms"
              f"  [texture={breakdown['frontend_texture']:.1f} buffer={breakdown['frontend_buffer']:.1f}]")
        if breakdown["tex_classes_available"]:
            invalid = (f"persist_invalid={breakdown['tex_persist_invalid']:.1f}"
                       if breakdown["tex_invalid_available"] else "persist_invalid=UNAVAILABLE")
            print("    texture by cache outcome: "
                  f"rtt={breakdown['tex_rtt']:.1f}"
                  f" compute={breakdown['tex_compute']:.1f}"
                  f" local={breakdown['tex_local']:.1f}"
                  f" persist_hit={breakdown['tex_persist_hit']:.1f}"
                  f" persist_reuse={breakdown['tex_persist_reuse']:.1f}"
                  f" persist_miss={breakdown['tex_persist_miss']:.1f}"
                  f" {invalid}"
                  f" other={breakdown['tex_other']:+.1f}"
                  + ("" if breakdown["tex_other_n"] is None
                     else f"/{breakdown['tex_other_n']}refs"))
            # Loud, and only when the two residuals disagree. A signed millisecond remainder with
            # NO unclassified references means the classification is losing time a named class
            # should hold -- a defect in this instrument, not in the renderer, and the breakdown
            # above is not trustworthy while it holds.
            if (breakdown["tex_other_n"] == 0 and abs(breakdown["tex_other"]) > 1.0):
                print(f"    *** {breakdown['tex_other']:+.1f}ms is unattributed with ZERO "
                      "unclassified references — the texture classification is losing time, "
                      "which is an instrument defect rather than a renderer one")
            witness = breakdown.get("tex_other_witness")
            if witness:
                # The identity of the slowest unclassified reference. Without it `other` is a number
                # with nothing to act on; with it the surface can be looked up directly.
                print(f"    slowest unclassified reference: {witness['ms']:.1f}ms"
                      f" addr=0x{witness['addr']:x}"
                      f" {witness['width']}x{witness['height']}x{witness['depth']}"
                      f" fmt={witness['format']}/{witness['components']}c"
                      f" tile={witness['tile_mode']}"
                      f" src={witness['source_bytes'] / (1024 * 1024):.1f}MiB"
                      f" compute_cand={int(witness['compute_candidate'])}"
                      f" persist_cand={int(witness['persistent_candidate'])}"
                      f" dcc={int(witness['compressed'])}")
        if breakdown["backend_available"]:
            print("  setup_resources (backend binding):       "
                  f"{breakdown['setup_resources']:.1f}ms"
                  f"  [texture={breakdown['res_texture']:.1f}"
                  f" buffer={breakdown['res_buffer']:.1f} ("
                  + (f"copy={breakdown['res_buffer_copy']:.1f}"
                     f" create={breakdown['res_buffer_create']:.1f}"
                     f" index_find={breakdown['res_buffer_index_find']:.1f}"
                     f" index_insert={breakdown['res_buffer_index_insert']:.1f}"
                     f" hash={breakdown['res_buffer_hash']:.1f}"
                     f" other={breakdown['res_buffer_other']:+.1f}"
                     if breakdown["buffer_leaves_available"]
                     else f"copy={breakdown['res_buffer_copy']:.1f};"
                          " create/index_find/index_insert/hash UNAVAILABLE") + ")"
                  f" descriptor={breakdown['res_descriptor']:.1f}"
                  f" other={breakdown['res_other']:.1f}]")
            # Loud, and only when it is genuinely non-zero. A sub-bucket total exceeding its parent
            # is an instrument defect, and the breakdown above is untrustworthy while it holds.
            if breakdown["res_over_attributed"] > 0.05:
                print("  *** the sub-buckets EXCEED setup_resources by "
                      f"{breakdown['res_over_attributed']:.1f}ms — the breakdown above is not "
                      "trustworthy; this is a defect in the instrument, not in the renderer")
        else:
            print("  setup_resources (backend binding):       "
                  f"{breakdown['setup_resources']:.1f}ms"
                  "  [breakdown UNAVAILABLE — this capture predates the backend sub-buckets;"
                  " do NOT subtract the frontend figures above from it, they are a different layer]")
    print(f"primary evidence: {summary['classification']} — {summary['reason']}")
    if summary.get("readback_note"):
        print(f"readback: {summary['readback_note']}")
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
