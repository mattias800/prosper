#!/usr/bin/env python3
"""The baseline comparison must FAIL when it cannot compare, not pass.

#3464 W6. `skip_survey.py --baseline` locks in today's refused-shader counts per title and reddens
when one goes up. The subtlety is entirely in the non-obvious verdicts, and both directions of getting
them wrong are silent:

**A run that could not be judged must FAIL, not pass.** This is the opposite stance from the survey's
own reporting, and deliberately so. A survey reports "I could not tell"; a GUARD that cannot tell has
not established the thing it exists to establish, and passing would mean a title that stopped booting
reads as a title with no dropped shaders. Trap 273 is this family exactly.

**A different GPU must refuse to compare at all.** The counts are host-dependent in the extreme: on an
AMD host wave64 is native, nothing is refused, and every count is legitimately 0. Comparing a
Windows/NVIDIA baseline against a Linux/AMD run would report "31 shaders fixed" and go green -- the
most encouraging possible output for a comparison that measured nothing.

Run: python test_skip_baseline.py
"""
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

import skip_survey  # noqa: E402

failures = []


def check(name, ok, detail=""):
    if ok:
        print("ok   - %s" % name)
    else:
        failures.append(name)
        print("FAIL - %s %s" % (name, detail))


NVIDIA = "NVIDIA GeForce RTX 4090 (discrete GPU)"
AMD = "AMD Radeon Graphics (RADV) (integrated GPU)"


def row(title, refused, frames=3000, device=NVIDIA, exited_early=False, admitted=0):
    return {"title": title, "refused_shaders": refused, "admitted_shaders": admitted,
            "frames": frames, "device": device, "exited_early": exited_early,
            "elapsed": 150.0, "returncode": None, "reasons": {"0x2 wave-any": refused},
            "widths": [64] if refused else [], "log": ""}


def main() -> int:
    base = {"device": NVIDIA, "titles": {"PPSA13579": 8, "PPSA02664": 4, "PPSA24651": 0}}

    # ---- the ordinary verdicts ------------------------------------------------------------------
    ok, why = skip_survey.compare_to_baseline(
        [row("PPSA13579", 8), row("PPSA02664", 4), row("PPSA24651", 0)], base)
    check("an unchanged corpus passes", ok, why)

    # A complete row set: only the count under test differs, so the verdict cannot come from
    # a title being absent. An earlier draft passed one row and went green for that reason.
    ok, why = skip_survey.compare_to_baseline([row("PPSA13579", 9), row("PPSA02664", 4), row("PPSA24651", 0)], base)
    check("a title that refuses MORE fails", not ok, why)
    check("...and the message names the title and both counts",
          "PPSA13579" in why and "8" in why and "9" in why, why)

    ok, why = skip_survey.compare_to_baseline([row("PPSA13579", 3), row("PPSA02664", 4), row("PPSA24651", 0)], base)
    check("a title that refuses FEWER passes", ok, why)
    check("...and says so, so the baseline gets tightened rather than drifting",
          "3" in why and "8" in why, why)

    # A title going from clean to non-clean is the regression this exists for.
    ok, why = skip_survey.compare_to_baseline([row("PPSA13579", 8), row("PPSA02664", 4), row("PPSA24651", 1)], base)
    check("a title regressing from 0 to 1 fails", not ok, why)

    # ---- the verdicts that must not silently pass -------------------------------------------------
    # A run that never got going has established nothing. Passing it would let a title that stopped
    # booting read as a title with no dropped shaders.
    ok, why = skip_survey.compare_to_baseline(
        [row("PPSA13579", 0, frames=0), row("PPSA02664", 4), row("PPSA24651", 0)], base)
    check("a run that presented no frames FAILS rather than reading as an improvement", not ok, why)
    check("...and says it could not be judged, not that it improved",
          "judge" in why.lower() or "uncomparable" in why.lower(), why)

    ok, why = skip_survey.compare_to_baseline(
        [row("PPSA13579", 0, exited_early=True), row("PPSA02664", 4),
         row("PPSA24651", 0)], base)
    check("a run that exited early FAILS rather than reading as an improvement", not ok, why)

    # A title in the baseline that was not surveyed at all is not evidence of anything.
    ok, why = skip_survey.compare_to_baseline([row("PPSA13579", 8)], base)
    check("a baseline title missing from the run FAILS", not ok, why)
    check("...and names the missing titles", "PPSA02664" in why, why)

    # ---- the host guard, which is the one that would go GREEN while measuring nothing --------------
    ok, why = skip_survey.compare_to_baseline(
        [row("PPSA13579", 0, device=AMD), row("PPSA02664", 0, device=AMD),
         row("PPSA24651", 0, device=AMD)], base)
    check("a different GPU REFUSES to compare rather than reporting 12 shaders fixed", not ok, why)
    check("...and names both devices so the reason is obvious",
          "NVIDIA" in why and "AMD" in why, why)

    # An unknown device is not a matching device.
    ok, why = skip_survey.compare_to_baseline(
        [row("PPSA13579", 8, device=""), row("PPSA02664", 4, device=""),
         row("PPSA24651", 0, device="")], base)
    check("an unrecorded device refuses to compare", not ok, why)

    # ---- review findings: passes that establish nothing -----------------------------------------
    #
    # A guard's only interesting failure is passing while measuring nothing, and each of these did.

    # F1. An empty baseline matched everything it knew about, which was nothing.
    ok, why = skip_survey.compare_to_baseline([row("PPSA13579", 99)], {"device": NVIDIA,
                                                                       "titles": {}})
    check("an empty baseline FAILS instead of vacuously passing", not ok, why)
    ok, why = skip_survey.compare_to_baseline([row("PPSA13579", 99)], {"device": NVIDIA})
    check("a baseline with no titles key FAILS", not ok, why)

    # F2. The "baseline records no device" branch had NO coverage: the arm that appeared to test it
    # landed on the GPU-mismatch branch instead, and deleting the branch left a SILENT PASS, since
    # {""} != {""} is False. Both sides unrecorded is the case that reaches it.
    ok, why = skip_survey.compare_to_baseline(
        [row("PPSA13579", 8, device="")], {"device": "", "titles": {"PPSA13579": 8}})
    check("a baseline with no recorded device FAILS even when the run also has none", not ok, why)
    # Not `"baseline" in why` -- that is true of 7 of the 8 verdict messages, so the arm
    # would survive nearly any mutation. Assert on the phrase only this branch produces.
    check("...and says the BASELINE is the unusable side",
          "lists no titles" in why or "records no device" in why, why)

    # F5. A surveyed title absent from the baseline was dropped silently, so a title refusing 99
    # could read as "unchanged".
    ok, why = skip_survey.compare_to_baseline(
        [row("PPSA13579", 8), row("PPSA99999", 99)], {"device": NVIDIA,
                                                      "titles": {"PPSA13579": 8}})
    check("a surveyed title missing from the baseline is REPORTED, not dropped",
          "PPSA99999" in why, why)
    check("...and its refusal count is named so 99 cannot hide behind 'unchanged'",
          "99" in why, why)

    # ---- the RECORD side, which the first mutation sweep never reached ---------------------------
    #
    # The sweep covered compare_to_baseline and stopped at the function boundary, leaving F3 and F4 --
    # the previous round's own fixes -- uncovered. Truth is defined on this side: a wrong baseline
    # makes every later comparison wrong, however well guarded the comparison is.

    # F4's rule, on the record side.
    data, why = skip_survey.baseline_from([row("PPSA13579", 0, frames=0)])
    check("a baseline REFUSES a run that presented no frames", data is None, why)
    check("...and names the title and the reason",
          "PPSA13579" in why and "no frames" in why, why)

    # The drift: this row is judgeable by frames but has no device, and was recorded as 0 while the
    # comparison refused it -- so --write-baseline X --baseline X wrote a file then rejected it.
    mixed = [row("PPSA13579", 8), row("PPSA02664", 0, device="")]
    data, why = skip_survey.baseline_from(mixed)
    check("a baseline REFUSES a row whose renderer announced no device", data is None, why)
    check("...so it can never write a file it would then refuse to compare against",
          "PPSA02664" in why, why)

    # Two devices in one run cannot produce one attributable baseline.
    data, why = skip_survey.baseline_from([row("PPSA13579", 8), row("PPSA02664", 4, device=AMD)])
    check("a baseline REFUSES a run spanning two GPUs", data is None, why)

    # ---- F3: an ORDERING, which a condition-mutation sweep cannot express -------------------------
    #
    # Every guard in this file is pinned by neutralising its condition. F3 is not a condition: it is
    # that the write happens AFTER the refusal check. Hoisting `write_text` above the guard leaves
    # every condition untouched and the suite green, while restoring the bug where a refused write
    # clobbers the committed baseline and prints "REFUSING to write". This arm asserts the effect on
    # disk rather than the control flow, which is the only way to see it.
    with tempfile.TemporaryDirectory() as tmp:
        existing = Path(tmp) / "committed.json"
        existing.write_text('{"device": "PRECIOUS", "titles": {"KEEP": 1}}')
        before = existing.read_bytes()
        data, why = skip_survey.write_baseline([row("PPSA13579", 0, frames=0)], existing)
        check("a refused write returns no data", data is None, why)
        check("a refused write leaves an existing baseline BYTE-IDENTICAL",
              existing.read_bytes() == before, existing.read_text())

    # ---- round-trip ------------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "baseline.json"
        rows = [row("PPSA13579", 8), row("PPSA02664", 4)]
        data, why = skip_survey.write_baseline(rows, path)
        check("a good run writes a baseline", data is not None, why)
        loaded = json.loads(path.read_text())
        check("a written baseline records the device", loaded.get("device") == NVIDIA, str(loaded))
        check("a written baseline records each title's count",
              loaded.get("titles") == {"PPSA13579": 8, "PPSA02664": 4}, str(loaded))
        ok, why = skip_survey.compare_to_baseline(rows, loaded)
        check("a freshly written baseline compares clean against its own run", ok, why)

    print("\n%d checks failed" % len(failures) if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
