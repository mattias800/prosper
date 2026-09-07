#!/usr/bin/env python3
"""Check actual backend decisions and selectors, with full compute logging disabled.

The executable verifies shader output, external-mutation repair and publication independently.
This wrapper requires the corresponding per-owner evidence, so empty output cannot pass a
positive arm. Each process starts with a fresh buffer cache and lazy diagnostic selectors.
"""

import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


PREFIX = "[compute-buffer-timing] "
FIXTURE = "[buffer-timing-fixture] "
BYTES = 2 << 20
TIMERS = (
    "setup_ms validation_ms upload_compare_ms upload_copy_ms upload_map_ms "
    "upload_watch_ms writeback_ms result_compare_ms guest_copy_ms guest_layout_ms "
    "result_map_ms result_watch_ms baseline_ms notify_ms source_validation_ms provenance_ms"
).split()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def fields(line, prefix):
    pairs = [part.split("=", 1) for part in line[len(prefix):].split()]
    require(all(len(pair) == 2 for pair in pairs), f"malformed record: {line}")
    result = dict(pairs)
    require(len(result) == len(pairs), f"duplicate fields: {line}")
    return result


def number(row, key):
    return int(row[key], 0)


def expect(row, **wanted):
    for key, value in wanted.items():
        key = key.replace("_", "-")
        actual = number(row, key) if isinstance(value, int) else row[key]
        require(actual == value,
                f"dispatch {row['dispatch']} owner {row['owner']}: "
                f"{key}={actual!r}, expected {value!r}")


def check_selected(rows, witness):
    require(len(rows) == 8, f"expected eight unique owner records, got {len(rows)}")
    grouped = {}
    identities = set()
    for row in rows:
        identity = (number(row, "submit"), number(row, "dispatch"), number(row, "owner"))
        require(identity not in identities, f"duplicate owner record {identity}")
        identities.add(identity)
        dispatch = identity[1]
        grouped.setdefault(dispatch, []).append(row)
        expect(row, submit=3407, order=dispatch * 10, ok=1,
               code=number(witness, "code"), hash=number(witness, "hash"),
               owner_resolved=1, host_backed=0, host_key=0, semantic=0,
               logical_bytes=BYTES, bytes=BYTES, guest_bytes=BYTES,
               writable=1, atomic=0, persistent=1)
        for timer in TIMERS:
            value = float(row[timer])
            require(math.isfinite(value) and value >= 0, f"invalid {timer}: {row[timer]}")
    require(set(grouped) == set(range(1, 8)), f"wrong dispatch census: {sorted(grouped)}")
    # A disabled clock can preserve every decision and byte counter. Require a positive measured
    # interval over the whole workload, while allowing short individual leaf timers to round to 0.
    for timer in ("setup_ms", "writeback_ms"):
        require(sum(float(row[timer]) for row in rows) > 0,
                f"positive arm collected no {timer}")
    for dispatch in range(1, 7):
        require(len(grouped[dispatch]) == 1, f"alias duplicated at dispatch {dispatch}")
        expect(grouped[dispatch][0], owner=0, owner_index=0, aliases=1, bindings="0,1")
    initial_address = number(grouped[1][0], "addr")
    require(all(number(grouped[n][0], "addr") == initial_address for n in range(1, 7)),
            "same-address cache sequence lost its identity")
    cold = grouped[1][0]
    expect(cold, cache="miss", validation="pooled-full", upload_skipped=0,
           compared_bytes=BYTES, baseline="created",
           gpu_compare="ineligible", writeback="changed",
           result_compared_bytes=BYTES, guest_copied_bytes=BYTES)
    # A fresh Vulkan allocation's initial contents are unspecified. The cold path always
    # compares its full extent, but an already-equal allocation legitimately needs no copy.
    require(number(cold, "uploaded-bytes") in (0, BYTES), "invalid cold upload byte count")
    journal = grouped[2][0]
    expect(journal, cache="hit", validation="journal", upload_skipped=1,
           compared_bytes=0, uploaded_bytes=0, gpu_compare="unchanged",
           writeback="gpu-unchanged", result_compared_bytes=0, guest_copied_bytes=0)
    exact = grouped[3][0]
    expect(exact, cache="hit", validation="full", upload_skipped=1,
           compared_bytes=BYTES, uploaded_bytes=0, gpu_compare="unchanged",
           writeback="gpu-unchanged", result_compared_bytes=0, guest_copied_bytes=0)
    watched = number(witness, "watch") == 1
    for dispatch in (4, 6):
        row = grouped[dispatch][0]
        expect(row, cache="hit", validation="watch" if watched else "full",
               upload_skipped=1, compared_bytes=0 if watched else BYTES,
               uploaded_bytes=0, gpu_compare="unchanged", writeback="gpu-unchanged",
               result_compared_bytes=0, guest_copied_bytes=0)
        if watched:
            expect(row, dirty_watch_chunks=0, total_watch_chunks=2)
    repaired = grouped[5][0]
    expect(repaired, cache="hit", validation="dirty-chunks" if watched else "full",
           upload_skipped=0, compared_bytes=(1 << 20) if watched else BYTES,
           uploaded_bytes=(1 << 20) if watched else BYTES,
           gpu_compare="ineligible", writeback="changed",
           result_compared_bytes=BYTES, guest_copied_bytes=BYTES)
    if watched:
        expect(repaired, dirty_watch_chunks=1, total_watch_chunks=2)
    separated = grouped[7]
    require(len(separated) == 2, "distinct storage buffers must report two owners")
    owners = {number(row, "owner"): row for row in separated}
    require(set(owners) == {0, 1}, "distinct buffer bindings lost an owner")
    for owner, row in owners.items():
        expect(row, owner_index=owner, aliases=0, bindings=str(owner))
    require(number(owners[0], "addr") == initial_address and
            number(owners[1], "addr") != initial_address,
            "distinct-address arm must retain the old identity and add a new one")
    expect(owners[0], cache="hit", upload_skipped=1, writeback="gpu-unchanged")
    expect(owners[1], cache="miss", validation="pooled-full", compared_bytes=BYTES,
           baseline="created", writeback="changed", guest_copied_bytes=BYTES)


def check_promotion(rows, witness):
    require(len(rows) == 9, f"expected nine promotion owner records, got {len(rows)}")
    indexed = {number(row, "dispatch"): row for row in rows}
    require(set(indexed) == set(range(1, 10)), "duplicate/missing promotion dispatch")
    cpu_result = witness["mode"] == "promotion-cpu"
    watched = number(witness, "watch") == 1
    address = number(indexed[1], "addr")
    for dispatch, row in indexed.items():
        expect(row, submit=3407 + dispatch, order=dispatch * 10, ok=1,
               code=number(witness, "code"), hash=number(witness, "hash"),
               addr=address, owner=0, owner_index=0, owner_resolved=1,
               aliases=1, bindings="0,1", host_backed=0, host_key=0, semantic=0,
               logical_bytes=BYTES, bytes=BYTES, guest_bytes=BYTES,
               writable=1, atomic=0, persistent=1)
        for timer in TIMERS:
            require(math.isfinite(float(row[timer])) and float(row[timer]) >= 0,
                    f"invalid {timer}: {row[timer]}")
        changed = dispatch in (1, 3, 8)
        expect(row, cache="miss" if dispatch == 1 else "hit",
               upload_skipped=0 if changed else 1,
               writeback="changed" if changed else "unchanged" if cpu_result else "gpu-unchanged",
               gpu_compare="ineligible" if changed or cpu_result else "unchanged",
               result_compared_bytes=BYTES if changed or cpu_result else 0,
               guest_copied_bytes=BYTES if changed else 0)
        if cpu_result:
            expect(row, baseline="disabled")
        if dispatch == 1:
            expect(row, validation="pooled-full", compared_bytes=BYTES)
            if not cpu_result:
                expect(row, baseline="created")
            require(number(row, "uploaded-bytes") in (0, BYTES), "invalid cold upload bytes")
        elif dispatch == 8 and watched:
            expect(row, validation="dirty-chunks", dirty_watch_chunks=1, total_watch_chunks=2,
                   compared_bytes=1 << 20, uploaded_bytes=1 << 20)
        elif dispatch in (7, 9) and watched:
            expect(row, validation="watch", dirty_watch_chunks=0, total_watch_chunks=2,
                   compared_bytes=0, uploaded_bytes=0)
        else:
            expect(row, validation="full", compared_bytes=BYTES,
                   uploaded_bytes=BYTES if changed else 0)
            if watched:
                # 2: first unchanged validation; 3: mutation resets the ladder. 4 and 5
                # accumulate two new validations. 6 arms before its third full comparison;
                # the setup census still saw zero preexisting watches on that acquisition.
                expect(row, dirty_watch_chunks=0, total_watch_chunks=0)
    for timer in ("setup_ms", "writeback_ms"):
        require(sum(float(row[timer]) for row in rows) > 0,
                f"promotion arm collected no {timer}")


def run_fixture(binary, mode, directory):
    # These tests intentionally control diagnostic/cache policy. Preserve driver/validation
    # settings so running the ctest under the validation-layer wrapper still validates Vulkan.
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("PROSPER_COMPUTE") and
                   key not in {"PROSPER_NO_PERSISTENT_COMPUTE_BUFFERS",
                               "PROSPER_NO_PERSISTENT_COMPUTE_BUFFER_RESULTS"}}
    result = subprocess.run([str(binary), mode, str(directory)], env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=120, check=False)
    # Keep all layer messages visible to the outer suite's validation scanner, including success.
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    require(result.returncode == 0, f"{mode}: fixture failed with {result.returncode}")
    lines = result.stderr.splitlines()
    require(f"{FIXTURE}success mode={mode}" in lines, f"{mode}: no successful execution witness")
    witnesses = [fields(line, FIXTURE) for line in lines
                 if line.startswith(FIXTURE + "mode=")]
    require(len(witnesses) == 1, f"{mode}: expected one fixture identity")
    require(witnesses[0]["mode"] == mode, "fixture mode mismatch")
    rows = [fields(line, PREFIX) for line in lines if line.startswith(PREFIX)]
    if mode in ("selected", "capture-armed"):
        check_selected(rows, witnesses[0])
    elif mode in ("promotion", "promotion-cpu"):
        check_promotion(rows, witnesses[0])
    else:
        require(not rows, f"{mode}: rejected selector/capture gate emitted buffer records")
    return witnesses[0]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_compute_buffer_timing_runtime.py FIXTURE_EXECUTABLE")
    binary = Path(sys.argv[1]).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="compute-buffer-timing-") as scratch:
        witnesses = []
        errors = []
        for mode in ("selected", "wrong-code", "wrong-hash", "disabled",
                     "capture-idle", "capture-armed", "promotion", "promotion-cpu"):
            directory = Path(scratch) / mode
            directory.mkdir()
            try:
                witnesses.append(run_fixture(binary, mode, directory))
            except AssertionError as error:
                # Run both promotion implementations even when the first exposes a defect.
                errors.append(f"{mode}: {error}")
        require(not errors, "\n".join(errors))
        require(len({row["hash"] for row in witnesses}) == 1,
                "selector controls did not execute the same compiled guest shader")
    print("compute buffer runtime: owner decisions, six selectors and two promotion arms passed")


if __name__ == "__main__":
    main()
