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
    log = (phase(ok=1, setup_ms=1, dispatch_ms=8, writeback_ms=1)
           + phase(ok=0, setup_ms=5, dispatch_ms=-5, cleanup_ms=6, total_ms=6))
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
    check("no impossible image share printed", "%" not in out.split("SUPPRESSED")[1][:200]
          or "of ALL dispatch wall" not in out, out)
    code, out, err = run(log)
    check("image section shown without the filter", "of ALL dispatch wall" in out, out)

    # Aliased bindings did no work and carry no sub-timers; averaging over them divides ms/binding
    # by the wrong denominator (they are 64% of Astro Bot's image records).
    log = phase(setup_ms=10, dispatch_ms=1) + image(ms=9, prepare_ms=9) + "".join(
        image(alias=True, ms=0.001) for _ in range(99))
    code, out, err = run(log)
    check("aliases excluded from binding count", "1 real bindings" in out, out)
    check("aliases still reported", "99 further records were aliased folds" in out, out)

    # The tool encodes a model of execute_item's phase structure; if that model stops matching, it
    # must say so rather than print a confident wrong table.
    code, out, err = run(phase(setup_ms=1, total_ms=99))
    check("stale phase model warns", "does not match these records" in err, err)
    code, out, err = run(phase(setup_ms=1, dispatch_ms=8, writeback_ms=1))
    check("consistent records warn about nothing", "does not match" not in err, err)

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
