#!/usr/bin/env python3
"""Self-test for compute_phase_report.py.

Every case here is one the report has already got wrong once. A report that prints a *plausible* wrong
table is worse than one that crashes, so most of these assert on what must NOT appear.

Run: python3 tools/perf/test_compute_phase_report.py
"""

import re
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).resolve().parent / "compute_phase_report.py"

PHASE_FIELDS = [
    "setup_ms", "setup_validate_ms", "setup_buffers_ms", "pipeline_ms", "dispatch_ms",
    "writeback_ms", "writeback_prepare_ms", "writeback_buffers_ms", "writeback_images_ms",
    "writeback_publish_ms", "map_ms", "prepare_ms", "watch_ms", "pack_ms", "layout_ms",
    "notify_ms", "cache_ms", "cleanup_ms",
]


def phase(submit=1, code=0x1, ok=1, **over):
    """One `[compute-phase]` line; unnamed fields default to 0 and total_ms is made consistent."""
    values = {k: 0.0 for k in PHASE_FIELDS}
    values.update(over)
    total = over.pop("total_ms", None)
    if total is None:
        total = (values["setup_ms"] + values["pipeline_ms"] + values["dispatch_ms"]
                 + values["writeback_ms"] + values["cleanup_ms"])
    body = " ".join(f"{k}={values[k]:.2f}" for k in PHASE_FIELDS)
    return (f"[compute-phase] submit={submit} code=0x{code:x} ok={ok} {body} "
            f"total_ms={total:.2f} subgroup=0\n")


def image(code=0x1, alias=False, ms=1.0, **over):
    if alias:
        return (f"[compute-image] code=0x{code:x} binding=1 class=sampled alias=1 "
                f"extent=8x8x1 ms={ms:.3f}\n")
    fields = {"query_ms": 0, "import_ms": 0, "cache_ms": 0, "staging_ms": 0, "prepare_ms": 0,
              "allocation_ms": 0, "view_ms": 0, "sampler_ms": 0}
    fields.update(over)
    body = " ".join(f"{k}={v:.3f}" for k, v in fields.items())
    return (f"[compute-image] code=0x{code:x} binding=1 class=sampled imported=0 extent=8x8x1 "
            f"guest=1024 staging=1024 normalized=0 texel=0 sampled-float=0 rgba8-reuse=0 "
            f"{body} ms={ms:.3f}\n")


def run(log_text, *args):
    proc = subprocess.run([sys.executable, str(TOOL), *args], input=log_text,
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def image_row(out, label):
    """(ms, ms_per_binding) for one row of the IMAGE table only.

    Scoped deliberately: `unattributed` also appears in the phase table above, so a whole-output scan
    reads the wrong row and then fails or passes for reasons unrelated to the image arithmetic.
    Returns parsed floats -- a substring test on the formatted number is true of its own prefixes.
    """
    section = out.split("setup image bindings:")
    if len(section) < 2:
        return (float("nan"), float("nan"))
    for line in section[1].splitlines():
        if line.strip().startswith(label):
            numbers = re.findall(r"-?\d+\.\d+", line)
            if len(numbers) >= 2:
                return (float(numbers[0]), float(numbers[-1]))
    return (float("nan"), float("nan"))


FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok  {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILURES.append(name)


def main():
    print("compute_phase_report self-test")

    # A failed dispatch's sub-timers are garbage (dispatch_ms negative, cleanup_ms == total_ms).
    # Summing them inverted the real Astro Bot table, so they must never reach the phase table.
    # The failed record's negative dispatch_ms must OUTWEIGH the succeeded one, or the two cancel and
    # the aggregate stays positive -- which is how this fixture first passed while proving nothing.
    # Per-check mutation coverage caught that: deleting the ok=0 exclusion left this check green.
    log = (phase(ok=1, setup_ms=1, dispatch_ms=8, writeback_ms=1)
           + phase(ok=0, setup_ms=50, dispatch_ms=-50, cleanup_ms=51, total_ms=51))
    code, out, err = run(log)
    check("failed dispatch excluded from the table", "1 succeeded dispatches" in out, out)
    check("failed dispatch still reported", "1 FAILED" in out, out)
    # Assert on the numbers in the table body, not on the presence of a hyphen anywhere in the
    # output -- the prose and the rules are full of them, and a substring test here would pass or
    # fail for reasons unrelated to the arithmetic.
    negatives = [line for line in out.splitlines()
                 if re.match(r"^\s{2}\S", line) and re.search(r"-\d+\.\d", line)]
    check("no negative time reaches the table", not negatives, negatives)

    # A record with no ok= field is truncated; defaulting it to succeeded is the unsafe direction.
    code, out, err = run("[compute-phase] submit=1 code=0x1 setup_ms=1.00 total_ms=1.00\n")
    check("record without ok= is not counted as succeeded", code == 1, out + err)

    # --since-submit cannot filter [compute-image] (no submit ordinal), so the image section must be
    # suppressed rather than divide a whole-log numerator by a filtered denominator.
    log = (phase(submit=1, dispatch_ms=100) + image(ms=90)
           + phase(submit=500, dispatch_ms=1) + image(ms=1))
    code, out, err = run(log, "--since-submit", "400")
    check("image section suppressed under --since-submit", "SUPPRESSED" in out, out)
    # Assert the *number* cannot appear, not a phrasing. The earlier form here was vacuous: its second
    # disjunct was implied by suppression working, so it was True whenever the feature worked AND
    # crashed with IndexError when it did not -- a check that could not fail, inside the PR that adds
    # trap 47. Mutation-checked: deleting the suppression makes this line fail, not error.
    check("no image share survives the filter", "of ALL dispatch wall" not in out, out)
    code, out, err = run(log)
    check("image section shown without the filter", "of ALL dispatch wall" in out, out)

    # Aliased bindings did no work and carry no sub-timers; averaging over them divides ms/binding by
    # the wrong denominator (they are 64% of Astro Bot's image records). Assert the ARITHMETIC, not
    # the label: with one real 9 ms binding and 99 aliases, prepare upload is 9.000 ms/binding if they
    # are excluded and 0.090 if they are not. A refactor that keeps the wording and restores the wrong
    # denominator must fail here.
    log = phase(setup_ms=10, dispatch_ms=1) + image(ms=9, prepare_ms=9) + "".join(
        image(alias=True, ms=0.01) for _ in range(99))
    code, out, err = run(log)
    check("aliases excluded from binding count", "1 real bindings" in out, out)
    check("aliases still reported", "99 further records were aliased folds" in out, out)
    check("ms/binding uses the real-binding denominator",
          image_row(out, "prepare upload")[1] == 9.000, image_row(out, "prepare upload"))
    # The alias records contribute 0.99 ms with no sub-timers, so if they are counted that whole
    # amount lands here. Parse the number and compare it; a substring test for "0.00" is true of
    # "0.001" as well and so passes with the bug in place, which is how the previous form of this
    # check survived its own mutation.
    check("alias ms does not land in image unattributed",
          abs(image_row(out, "unattributed")[0]) < 0.05, image_row(out, "unattributed"))

    # An all-alias log is ordinary input (e.g. --program on a kernel whose bindings all fold). The
    # image section must say so and STILL print the top-programs table below it -- an early return
    # here silently truncated the report.
    code, out, err = run(phase(dispatch_ms=1) + image(alias=True, ms=0.5))
    check("all-alias log reports the folds", "all 1 records are aliased folds" in out, out)
    check("all-alias log still prints top programs", "top 1 programs by total cost" in out, out)

    # `unattributed` is thresholded against its own PARENT, not the grand total: a small parent must
    # not lose its remainder just because the run is large. Here writeback_images_ms is 4 ms of a
    # 1004 ms run (0.4 % of grand, so the old grand-relative rule dropped it) while its own children
    # explain none of it. Mutation-checked: restoring the grand-relative rule makes this line fail.
    log = phase(dispatch_ms=1000, writeback_ms=4, writeback_images_ms=4)
    code, out, err = run(log)
    images_unattr = [l for l in out.splitlines()
                     if "unattributed" in l and l.startswith("      ")]
    check("small parent keeps its unattributed remainder", images_unattr, out)

    # The tool encodes a model of execute_item's phase structure; if that model stops matching, it
    # must say so rather than print a confident wrong table.
    code, out, err = run(phase(setup_ms=1, total_ms=99))
    check("stale phase model warns", "does not match these records" in err, err)
    # ...and warns on STDOUT too. The ctest gate feeds synthetic records, so it pins the tool against
    # its own model and can never notice the emitter drifting; this runtime check is the only real
    # drift detector, and `report.py run.log > table.txt` discards stderr while keeping a table that
    # still looks authoritative. The banner must travel with the table.
    check("stale model banner survives redirection", "NOT TRUSTWORTHY" in out, out)
    code, out, err = run(phase(setup_ms=1, dispatch_ms=8, writeback_ms=1))
    check("consistent records warn about nothing", "does not match" not in err, err)
    check("no banner on consistent records", "NOT TRUSTWORTHY" not in out, out)

    # "No records" must not blame the env switch when a filter is what excluded them.
    code, out, err = run(phase(submit=1), "--since-submit", "999")
    check("filter named when it excluded everything", "--since-submit 999" in err, err)
    code, out, err = run("noise\n")
    check("missing switch named when the log has no records",
          "PROSPER_COMPUTE_PHASE_TIMING" in err, err)

    # Degenerate inputs must exit cleanly rather than traceback.
    code, out, err = run(phase(ok=0, total_ms=1))
    check("all-failed log exits cleanly", code == 1 and "Traceback" not in err, err)
    code, out, err = run(phase(setup_ms=0, total_ms=0))
    check("zero-total log does not divide by zero", "Traceback" not in err, err)
    proc = subprocess.run([sys.executable, str(TOOL), "/nonexistent/log"],
                          capture_output=True, text=True)
    check("missing file is a message, not a traceback",
          "Traceback" not in proc.stderr and proc.returncode == 2, proc.stderr)

    # --csv and --program must survive the same records.
    code, out, err = run(phase(setup_ms=1, dispatch_ms=1), "--csv")
    check("--csv emits a header", out.startswith("phase,key,ms"), out)
    # CSV is the most-redirected output of all; a stale model must be visible in the file itself.
    code, out, err = run(phase(setup_ms=1, total_ms=99), "--csv")
    check("--csv carries the stale-model banner", "NOT TRUSTWORTHY" in out, out)
    code, out, err = run(phase(code=0xAA, dispatch_ms=1), "--program", "0xbb")
    check("--program miss names the filter", "--program 0xbb" in err, err)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} case(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
