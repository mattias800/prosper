#!/usr/bin/env python3
"""Exercise primary residency versus optional result allocations on the actual backend.

Charges are VkMemoryRequirements sizes retained by the cache, not physical allocator pool sizes.
A fresh probe process discovers those sizes before the two pressure processes set their cap.
"""

import os
from pathlib import Path
import subprocess
import sys


PREFIX = "[compute-buffer-timing] "
FIXTURE = "[buffer-residency-fixture] "
MIB = 1 << 20
BYTES = 2 * MIB


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def parse(line, prefix):
    pairs = [part.split("=", 1) for part in line[len(prefix):].split()]
    require(all(len(pair) == 2 for pair in pairs), f"malformed record: {line}")
    result = dict(pairs)
    require(len(result) == len(pairs), f"duplicate field: {line}")
    return result


def number(row, key):
    return int(row[key], 0)


def expect(row, **wanted):
    for key, value in wanted.items():
        key = key.replace("_", "-")
        actual = number(row, key) if isinstance(value, int) else row[key]
        require(actual == value, f"dispatch {row['dispatch']} owner {row['owner']}: "
                f"{key}={actual!r}, expected {value!r}")


def run(binary, mode, cap_mib, large_bytes=None):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("PROSPER_COMPUTE") and
                   key not in {"PROSPER_NO_PERSISTENT_COMPUTE_BUFFERS",
                               "PROSPER_NO_PERSISTENT_COMPUTE_BUFFER_RESULTS"}}
    command = [str(binary), mode, str(cap_mib)]
    if large_bytes is not None:
        command.append(str(large_bytes))
    result = subprocess.run(command, env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, timeout=120, check=False)
    # Preserve validation-layer messages for the outer suite's scanner on success as well.
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    require(result.returncode == 0, f"{mode}: fixture exited {result.returncode}")
    lines = result.stderr.splitlines()
    require(FIXTURE + "success mode=" + mode in lines, f"{mode}: no successful guest-output witness")
    witnesses = [parse(line, FIXTURE) for line in lines if line.startswith(FIXTURE + "mode=")]
    require(len(witnesses) == 1 and witnesses[0]["mode"] == mode, "missing/invalid identity")
    witness = witnesses[0]
    rows = [parse(line, PREFIX) for line in lines if line.startswith(PREFIX)]
    require(rows, f"{mode}: empty cache-decision census")
    identities = set()
    for row in rows:
        dispatch = number(row, "dispatch")
        identity = (dispatch, number(row, "owner"))
        require(identity not in identities, f"duplicate unique owner {identity}")
        identities.add(identity)
        large_owner = mode == "requirements-large" or (mode == "partial" and identity == (3, 2))
        owner_bytes = large_bytes if large_owner else BYTES
        expect(row, submit=4400 + dispatch, order=10 * dispatch, ok=1,
               code=number(witness, "code"), hash=number(witness, "hash"),
               owner_resolved=1, host_backed=0, host_key=0, semantic=0,
               bytes=owner_bytes, logical_bytes=owner_bytes, guest_bytes=owner_bytes,
               writable=1, atomic=0, cache_limit=cap_mib * MIB)
        charge = number(row, "primary-allocation-bytes") + number(row, "result-allocation-bytes")
        require(0 <= charge <= number(row, "cache-bytes") <= cap_mib * MIB,
                f"invalid cache accounting or exceeded cap: {row}")
    return rows, witness


def probe(binary):
    rows, witness = run(binary, "requirements", 64)
    require(len(rows) == 1, "requirements probe duplicated alias ownership")
    row = rows[0]
    expect(row, owner=0, aliases=2, bindings="0,1,2", cache="miss", persistent=1,
           baseline="created", writeback="changed")
    primary = number(row, "primary-allocation-bytes")
    baseline = number(row, "result-allocation-bytes")
    require(primary >= BYTES and baseline >= BYTES, "probe did not retain both allocation types")
    expect(row, cache_bytes=primary + baseline)
    # Fit all three primaries and one baseline; also fit the two primaries+baselines used by
    # the pinned arm. Rounding is in MiB because that is the production cache-cap interface.
    cap_mib = (max(3 * primary + baseline, 2 * (primary + baseline)) + MIB - 1) // MIB
    require(cap_mib <= 1024 and cap_mib * MIB < 3 * primary + 2 * baseline,
            "unsupported driver allocation shape: cannot establish bounded baseline pressure")
    return primary, baseline, cap_mib, witness


def pressure(binary, mode, primary, baseline, cap_mib):
    rows, witness = run(binary, mode, cap_mib)
    pinned = mode == "pinned"
    require(len(rows) == (12 if pinned else 9), "wrong pressure owner census")
    grouped = {}
    for row in rows:
        grouped.setdefault(number(row, "dispatch"), []).append(row)
    last = 10 if pinned else 9
    require(set(grouped) == set(range(1, last + 1)), "missing pressure dispatch")
    for dispatch, owners in grouped.items():
        if pinned and dispatch == 3:
            continue
        require(len(owners) == 1, "an exact alias produced multiple owners")
        expect(owners[0], owner=0, owner_index=0, aliases=2, bindings="0,1,2",
               persistent=1, primary_allocation_bytes=primary)
    first = grouped[1][0]
    second = grouped[2][0]
    expect(first, cache="miss", result_allocation_bytes=baseline)
    expect(second, cache="miss", result_allocation_bytes=baseline)
    if pinned:
        by_binding = {number(row, "owner"): row for row in grouped[3]}
        require(set(by_binding) == {0, 1, 2}, "pinned arm lost a distinct owner")
        for binding, row in by_binding.items():
            expect(row, owner_index=binding, aliases=0, bindings=str(binding))
        for binding in (0, 1):
            expect(by_binding[binding], cache="hit", persistent=1,
                   primary_allocation_bytes=primary, result_allocation_bytes=baseline,
                   gpu_compare="unchanged", writeback="gpu-unchanged")
        # The two existing result owners are pinned during C's attempted admission. Refusal
        # keeps C transient for this dispatch; it must not destroy resources already in use.
        expect(by_binding[2], cache="miss", persistent=0, primary_allocation_bytes=0,
               result_allocation_bytes=0, writeback="changed")
        require(len({number(row, "cache-bytes") for row in by_binding.values()}) == 1,
                "end-of-dispatch owner rows disagree on total residency")
        expect(by_binding[0], cache_bytes=2 * (primary + baseline))
    cold_c = 4 if pinned else 3
    warm = cold_c + 1
    expect(grouped[cold_c][0], cache="miss")
    addresses = [number(first, "addr"), number(second, "addr"),
                 number(grouped[cold_c][0], "addr")]
    require(len(set(addresses)) == 3, "working set does not contain three distinct buffers")
    for offset in range(3):
        row = grouped[warm + offset][0]
        expect(row, addr=addresses[offset], cache="hit", upload_skipped=1, uploaded_bytes=0)
        require(row["writeback"] in ("unchanged", "gpu-unchanged"),
                "warm identical guest result was not recognized")
    changed = warm + 3
    expect(grouped[changed][0], addr=addresses[0], cache="hit", upload_skipped=1,
           uploaded_bytes=0, writeback="changed", guest_copied_bytes=BYTES)
    repeated = grouped[changed + 1][0]
    expect(repeated, addr=addresses[0], cache="hit", upload_skipped=1, uploaded_bytes=0,
           guest_copied_bytes=0)
    require(repeated["writeback"] in ("unchanged", "gpu-unchanged"),
            "changed result did not become the new unchanged result")
    repaired = grouped[changed + 2][0]
    expect(repaired, addr=addresses[1], cache="hit", upload_skipped=0,
           uploaded_bytes=BYTES, writeback="changed", guest_copied_bytes=BYTES,
           gpu_compare="ineligible")
    return witness


def partial_reclaim(binary, primary, baseline, cap_mib):
    combined = primary + baseline
    limit = cap_mib * MIB
    large_bytes = ((limit - combined) // MIB + 1) * MIB
    require(BYTES <= large_bytes <= limit, "unsupported partial-admission source extent")
    # Measure C in a fresh roomy cache. Its actual primary must fit the pressure cap alone;
    # otherwise the ordinary oversized-allocation guard could hide a broken reclaim preflight.
    discovery, discovery_identity = run(binary, "requirements-large", max(64, cap_mib), large_bytes)
    require(len(discovery) == 1, "large probe duplicated alias ownership")
    measured = discovery[0]
    expect(measured, owner=0, aliases=2, bindings="0,1,2", cache="miss", persistent=1,
           baseline="created", writeback="changed", guest_copied_bytes=large_bytes)
    large_primary = number(measured, "primary-allocation-bytes")
    require(large_primary >= large_bytes and limit - combined < large_primary <= limit,
            "unsupported allocation shape: cannot isolate insufficient partial reclaim")
    # B's full combined charge is idle and reclaimable, but less than the required reclamation.
    require(0 < combined < 2 * combined + large_primary - limit,
            "partial-reclaim fixture lacks nonzero, insufficient reclaimable bytes")
    rows, witness = run(binary, "partial", cap_mib, large_bytes)
    require(witness["hash"] == discovery_identity["hash"], "large probe changed compiled shader")
    require(len(rows) == 5, "wrong partial-reclaim owner census")
    indexed = {(number(row, "dispatch"), number(row, "owner")): row for row in rows}
    require(set(indexed) == {(1, 0), (2, 0), (3, 0), (3, 2), (4, 0)},
            "missing/duplicate partial-reclaim dispatch owner")
    for index in (1, 2):
        expect(indexed[index, 0], cache="miss", persistent=1, aliases=2, bindings="0,1,2",
               primary_allocation_bytes=primary, result_allocation_bytes=baseline)
    for key in ((3, 0), (4, 0)):
        expect(indexed[key], cache="hit", persistent=1, upload_skipped=1,
               primary_allocation_bytes=primary, result_allocation_bytes=baseline,
               cache_bytes=2 * combined, writeback="gpu-unchanged", gpu_compare="unchanged",
               guest_copied_bytes=0)
    expect(indexed[3, 0], owner_index=0, aliases=1, bindings="0,1",
           addr=number(indexed[1, 0], "addr"))
    expect(indexed[3, 2], owner_index=2, aliases=0, bindings="2", cache="miss", persistent=0,
           primary_allocation_bytes=0, result_allocation_bytes=0, cache_bytes=2 * combined,
           writeback="changed", guest_copied_bytes=large_bytes)
    expect(indexed[4, 0], owner_index=0, aliases=2, bindings="0,1,2",
           addr=number(indexed[2, 0], "addr"))
    require(len({number(indexed[key], "addr") for key in ((1, 0), (2, 0), (3, 2))}) == 3,
            "partial-reclaim owners are not three distinct allocations")
    return witness


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_compute_buffer_residency_runtime.py FIXTURE_EXECUTABLE")
    binary = Path(sys.argv[1]).resolve(strict=True)
    primary, baseline, cap_mib, witness = probe(binary)
    print(f"cache residency charges: primary={primary} baseline={baseline} cap={cap_mib} MiB")
    errors = []
    for mode in ("cycle", "pinned"):
        try:
            other = pressure(binary, mode, primary, baseline, cap_mib)
            require(other["hash"] == witness["hash"], "pressure arm changed its compiled shader")
        except AssertionError as error:
            errors.append(f"{mode}: {error}")
    try:
        other = partial_reclaim(binary, primary, baseline, cap_mib)
        require(other["hash"] == witness["hash"], "partial-reclaim arm changed compiled shader")
    except AssertionError as error:
        errors.append(f"partial: {error}")
    require(not errors, "\n".join(errors))
    print("compute buffer residency: primary reuse, bounded charges and failed admission passed")


if __name__ == "__main__":
    main()
