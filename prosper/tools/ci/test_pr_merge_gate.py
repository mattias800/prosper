#!/usr/bin/env python3
"""Self-checking tests for pr_merge_gate.py (exit code is truth).

Nothing here touches the network. `evaluate()` and `parse_checks()` are pure, so every arm drives
them with synthetic payloads shaped exactly like `gh pr checks --json name,state,bucket` emits.

The two arms that matter most are RECONSTRUCTIONS of merges that actually went wrong (#3259), not
invented cases: #3234's red `Windows MinGW` and #3243's frozen PR head. An arm built from a real
incident cannot be satisfied by a scenario nobody encounters.

HOW TO TELL A REAL ARM FROM ONE THAT REDDENS NOTHING (borrowed from test_trap_number.py, and
applied here):

    Ask "WHAT ELSE COULD SATISFY THIS ASSERTION?"

For a gate whose whole output is a boolean, that question bites hard: nearly every refusal arm is
satisfied by a gate that refuses EVERYTHING. `ok == False` alone therefore discriminates nothing.
So each refusal arm also asserts on the REASON, and the suite carries a positive control -- the
all-pass matching-head case -- whose failure would expose a gate stuck at "no". The two together
are what make a refusal meaningful; either alone is void.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pr_merge_gate import GateError, evaluate, parse_checks  # noqa: E402

FAILURES = []


def case(label, got, want):
    ok = got == want
    print(("  ok   " if ok else "  FAIL ") + label)
    if not ok:
        FAILURES.append("%s (got %r, want %r)" % (label, got, want))


def refused_for(res, needle):
    """True when the gate refused AND said so for the stated reason."""
    return (not res.ok) and any(needle in r for r in res.reasons)


HEAD = "04de4ac925d4503d71d3ac999feb0209f539bfbe"
OTHER = "fd1dbb92aa1c40e2b2f7a0f0e8ee2f0f3d5c1a77"


def rows(*pairs):
    return [{"name": n, "bucket": b, "state": "X"} for n, b in pairs]


print("\n-- the positive control (without it, every refusal arm below is void)")
res = evaluate(parse_checks(rows(
    ("Docs", "pass"), ("Linux", "pass"), ("Windows App", "pass"),
    ("macOS (x86_64 / Rosetta 2)", "pass"), ("Progress tracker", "skipping"),
)), HEAD, HEAD)
case("all-pass with a matching head is GREEN", res.ok, True)
case("...and counts the passes, not the words in their names", res.counts["pass"], 4)
case("...and a deliberately skipped job does not block", res.counts["skipping"], 1)

print("\n-- a check NAME is never read as a status (the #3234 class)")
# The historical failure: a shell gate split `Windows MinGW\tfail` on whitespace, read `MinGW`
# where the status belongs, and merged. Asserting on the count (not just on ok) is what makes this
# arm specific: a gate that refused for any other reason would still leave fail==1 unexplained.
res = evaluate(parse_checks(rows(
    ("Docs", "pass"), ("Linux", "pass"), ("Windows MinGW", "fail"),
)), HEAD, HEAD)
case("a failing multi-word check is counted as a FAILURE", res.counts["fail"], 1)
case("...and it refuses, naming failure as the reason", refused_for(res, "failed or were cancelled"), True)
case("...and it is named in the blocking list", ("Windows MinGW", "fail") in res.blocking, True)

# The mirror image, and the subtler half: multi-word names must not inflate the PASS count either.
res = evaluate(parse_checks(rows(
    ("Windows App", "pass"), ("Linux App Package", "pass"), ("macOS App (SDL3 + MoltenVK)", "pass"),
)), HEAD, HEAD)
case("multi-word passing names count once each, not per word", res.counts["pass"], 3)

print("\n-- CI green on a commit that would not merge (the #3243 class)")
# Every check passes. The only thing wrong is that they describe a commit the branch has moved
# past -- which is precisely the state that merged an unreviewed 239-byte overread.
res = evaluate(parse_checks(rows(("Docs", "pass"), ("Linux", "pass"))), OTHER, HEAD)
case("all-green on a stale head still REFUSES", res.ok, False)
case("...for the head-mismatch reason specifically", refused_for(res, "CI never saw what would merge"), True)
case("...and no check is blamed, because none failed", res.blocking, [])

print("\n-- an empty or unverified result is VOID, not green")
res = evaluate(parse_checks([]), HEAD, HEAD)
case("zero checks refuses", res.ok, False)
case("...calling itself void rather than failed", refused_for(res, "VOID, not green"), True)

res = evaluate(parse_checks(rows(("Progress tracker", "skipping"), ("Docs", "skipping"))), HEAD, HEAD)
case("an ALL-SKIPPED result refuses too (nothing was verified)", refused_for(res, "VOID, not green"), True)

print("\n-- pending and cancelled block")
res = evaluate(parse_checks(rows(("Linux", "pass"), ("Windows MinGW", "pending"))), HEAD, HEAD)
case("a pending check refuses", refused_for(res, "still pending"), True)
case("...and is named so the waiter knows what it waits on", res.pending, ["Windows MinGW"])

res = evaluate(parse_checks(rows(("Linux", "pass"), ("Windows App", "cancel"))), HEAD, HEAD)
case("a CANCELLED check blocks (nothing was verified by it)", refused_for(res, "failed or were cancelled"), True)

print("\n-- a bucket this tool has never heard of must not read as success")
res = evaluate(parse_checks(rows(("Linux", "pass"), ("Some Future Job", "quarantined"))), HEAD, HEAD)
case("an unknown bucket blocks", res.ok, False)
case("...and is reported as blocking, not silently dropped", ("Some Future Job", "quarantined") in res.blocking, True)

print("\n-- the parser refuses the TEXT form, so no loose fallback can creep back in")
text = "Docs\tpass\t34s\thttps://example\nWindows MinGW\tfail\t7m\thttps://example\n"
try:
    parse_checks(text)
    case("tab-separated text is rejected", "accepted", "GateError")
except GateError:
    case("tab-separated text is rejected", "GateError", "GateError")

for label, payload in [
    ("a JSON object instead of a list", '{"bucket": "pass"}'),
    ("rows missing 'bucket' (wrong --json fields)", '[{"name": "Docs", "state": "SUCCESS"}]'),
    ("a non-string bucket", '[{"name": "Docs", "bucket": 1}]'),
]:
    try:
        parse_checks(payload)
        case(label + " is rejected", "accepted", "GateError")
    except GateError:
        case(label + " is rejected", "GateError", "GateError")

print("\n-- head comparison is skipped only when the caller supplies no heads")
res = evaluate(parse_checks(rows(("Linux", "pass"))), None, None)
case("no head information given -> checks alone decide", res.ok, True)

print()
if FAILURES:
    for f in FAILURES:
        print("FAIL: %s" % f, file=sys.stderr)
    print("\n%d case(s) failed." % len(FAILURES), file=sys.stderr)
    sys.exit(1)
print("all cases passed")
