#!/usr/bin/env python3
"""Self-checking tests for pr_merge_gate.py (exit code is truth).

Nothing here touches the network. `evaluate()` and `parse_checks()` are pure, so every arm drives
them with synthetic payloads shaped exactly like `gh pr checks --json name,state,bucket` emits.

One arm is a RECONSTRUCTION of a merge that actually went wrong (#3259): #3234's red
`Windows MinGW`, read as green because a shell split the check NAME on whitespace. An arm built
from a real incident cannot be satisfied by a scenario nobody encounters.

The stale-head arm is NOT one, and saying so matters because an earlier version of this file
claimed it was. #3259 originally cited #3243 as a second incident; #3243's head was in fact the
branch tip when it merged, with every check green on that exact commit. That arm guards a state
this repository produces routinely -- every lane pushes after CI starts -- which is a weaker
warrant than a reconstruction, and it should not borrow one it does not have.

HOW TO TELL A REAL ARM FROM ONE THAT REDDENS NOTHING (borrowed from test_trap_number.py, and
applied here):

    Ask "WHAT ELSE COULD SATISFY THIS ASSERTION?"

For a gate whose whole output is a boolean, that question bites hard: nearly every refusal arm is
satisfied by a gate that refuses EVERYTHING. `ok == False` alone therefore discriminates nothing.
So each refusal arm also asserts on the REASON, and the suite carries a positive control -- the
all-pass matching-head case -- whose failure would expose a gate stuck at "no". The two together
are what make a refusal meaningful; either alone is void.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pr_merge_gate  # noqa: E402
from pr_merge_gate import GateError, evaluate, parse_checks, resolve_tip  # noqa: E402

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

print("\n-- CI green on a commit that would not merge")
# Every check passes. The only thing wrong is that the recorded head is a commit the branch has
# moved past, so nothing verified what would actually merge. No merge here is known to have been
# lost this way (see the docstring); the arm guards the state, not a scar.
res = evaluate(parse_checks(rows(("Docs", "pass"), ("Linux", "pass"))), OTHER, HEAD)
case("all-green on a stale head still REFUSES", res.ok, False)
case("...for the head-mismatch reason specifically", refused_for(res, "nothing verified what would merge"), True)
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

print("\n-- evaluate() without heads is the PURE contract, not a reachable gate state")
# evaluate() is deliberately usable without head information so the arms above can drive it. That
# is NOT a state a real run can reach: collect() raises before returning a null head, and the
# main() arms below pin that. Keeping this arm without that note previously read as blessing a
# silent GREEN with no head verification.
res = evaluate(parse_checks(rows(("Linux", "pass"))), None, None)
case("no head information given -> checks alone decide", res.ok, True)


# ---------------------------------------------------------------------------------------------
# The I/O layer. Everything above drives evaluate(), which is pure -- and a suite that stops there
# leaves collect() and main() pinned by NOTHING. Review demonstrated five mutations in those two
# functions that the pure-function arms could not see, two of them fatal:
#
#     return 2  -> return 0   in main's `except GateError`   : exit 0 on ANY tooling failure
#     `0 if res.ok else 1` -> `0`                            : a gate that APPROVES EVERYTHING
#
# The second is the mirror of the flaw the arms above defend against. Those arms guard against a
# gate stuck at "no"; nothing guarded against one stuck at "yes", because no arm ever called main().
# `_run` is the single process boundary, so monkeypatching it covers both functions with no
# production seam.
# ---------------------------------------------------------------------------------------------

print("\n-- resolve_tip matches the ref EXACTLY, not by suffix")
LS_ONE = "aaaa1111\trefs/heads/topic\n"
LS_DECOY = "bbbb2222\trefs/heads/other/refs/heads/topic\naaaa1111\trefs/heads/topic\n"
case("a single exact match resolves", resolve_tip(LS_ONE, "topic"), "aaaa1111")
# git ls-remote matches by SUFFIX, so this decoy really does come back on the same query. Taking
# line 1 would compare against a branch nobody asked about -- and report GREEN for it.
try:
    case("a suffix decoy on line 1 does not win", resolve_tip(LS_DECOY, "topic"), "aaaa1111")
except GateError as exc:
    # Reported as a named failure rather than an escaping traceback: a suffix-matching
    # implementation raises here (two hits) instead of returning the wrong sha, and a suite that
    # dies with a stack trace tells the reader far less than one that names the arm.
    case("a suffix decoy on line 1 does not win", "GateError: %s" % exc, "aaaa1111")
for label, out, br in [("no match refuses", "", "topic"),
                       ("only a suffix decoy refuses", "bbbb2222\trefs/heads/x/refs/heads/topic\n", "topic")]:
    try:
        resolve_tip(out, br); case(label, "returned", "GateError")
    except GateError:
        case(label, "GateError", "GateError")


class FakeGh:
    """Stands in for `_run`, keyed on the command being asked for."""

    def __init__(self, checks_json, view_json, ls_out, fail_on=None):
        self.checks_json, self.view_json, self.ls_out, self.fail_on = checks_json, view_json, ls_out, fail_on

    def __call__(self, cmd, cwd):
        joined = " ".join(cmd)
        if self.fail_on and self.fail_on in joined:
            raise GateError("simulated failure of %s" % self.fail_on)
        if cmd[0] == "git":
            return self.ls_out
        if "checks" in cmd:
            return self.checks_json
        return self.view_json


GREEN_CHECKS = '[{"name": "Windows App", "bucket": "pass", "state": "S"}]'
RED_CHECKS = '[{"name": "Windows MinGW", "bucket": "fail", "state": "F"}]'
VIEW = '{"headRefOid": "aaaa1111", "headRefName": "topic"}'


def run_main(fake, argv=("1",)):
    real = pr_merge_gate._run
    pr_merge_gate._run = fake
    try:
        return pr_merge_gate.main(list(argv))
    finally:
        pr_merge_gate._run = real


print("\n-- main() exit codes: the arms that pin a gate stuck at YES")
case("all green, head == tip -> exit 0",
     run_main(FakeGh(GREEN_CHECKS, VIEW, LS_ONE)), 0)
# Without this arm, replacing main's return with a bare `return 0` passes the whole suite.
case("a FAILING check -> exit 1, never 0",
     run_main(FakeGh(RED_CHECKS, VIEW, LS_ONE)), 1)
case("head behind the branch tip -> exit 1, never 0",
     run_main(FakeGh(GREEN_CHECKS, VIEW, "cccc3333\trefs/heads/topic\n")), 1)

print("\n-- main() distinguishes 'could not run' from 'said no'")
# `return 2 -> return 0` here means every auth failure, network failure and malformed payload
# reports a clean merge. This is the single most dangerous mutation in the file.
case("gh failure -> exit 2, not 0 and not 1",
     run_main(FakeGh(GREEN_CHECKS, VIEW, LS_ONE, fail_on="pr checks")), 2)
case("git ls-remote failure -> exit 2",
     run_main(FakeGh(GREEN_CHECKS, VIEW, LS_ONE, fail_on="ls-remote")), 2)
case("a malformed check payload -> exit 2",
     run_main(FakeGh("not json at all", VIEW, LS_ONE)), 2)

print("\n-- a PR with no usable head must REFUSE, not silently skip the head check")
# Previously headRefOid: null left pr_head=None, evaluate skipped the comparison, render() printed
# no head line, and the gate exited 0 -- a silent GREEN precisely where the verification belonged.
for label, view in [("headRefOid null", '{"headRefOid": null, "headRefName": "topic"}'),
                    ("headRefOid empty", '{"headRefOid": "", "headRefName": "topic"}'),
                    ("headRefName missing", '{"headRefOid": "aaaa1111"}')]:
    case(label + " -> exit 2, never 0", run_main(FakeGh(GREEN_CHECKS, view, LS_ONE)), 2)

print("\n-- --json is an exit path too, and nothing pinned it")
# The tenth mutation: `return 0` inserted before the --json print left the whole suite green,
# because no arm ever passed --json. That path is not decorative -- it is the machine-readable
# record of what a gate refused, so anything built on it inherits whatever this arm does not check.
import io  # noqa: E402
import contextlib  # noqa: E402


def run_main_capture(fake, argv):
    buf = io.StringIO()
    real = pr_merge_gate._run
    pr_merge_gate._run = fake
    try:
        with contextlib.redirect_stdout(buf):
            rc = pr_merge_gate.main(list(argv))
    finally:
        pr_merge_gate._run = real
    return rc, buf.getvalue()


rc, out = run_main_capture(FakeGh(RED_CHECKS, VIEW, LS_ONE), ["1", "--json"])
case("--json on a failing PR still exits 1", rc, 1)
try:
    payload = json.loads(out)
except ValueError:
    payload = None
case("...and emits parseable JSON", isinstance(payload, dict), True)
case("...whose ok is False", (payload or {}).get("ok"), False)
# The reasons list is the override record a caller would quote, so pin that it is populated.
case("...and whose reasons say why", bool((payload or {}).get("reasons")), True)
rc, out = run_main_capture(FakeGh(GREEN_CHECKS, VIEW, LS_ONE), ["1", "--json"])
case("--json on a green PR exits 0", rc, 0)
# Parsed defensively: a bare json.loads here CRASHES the suite under any mutant that suppresses
# the --json print, which costs every arm below it and reports as a traceback instead of a name.
try:
    green_ok = json.loads(out).get("ok")
except ValueError:
    green_ok = "<not JSON>"
case("...with ok True", green_ok, True)

print("\n-- a failure while REPORTING is 'could not evaluate', not 'said no'")
# The guard used to wrap collect() only, so an exception in render()/print() escaped and the
# interpreter exited 1 -- the exact masquerade the code comment warns about. Reachable without a
# mutation: `pr_merge_gate.py N | head -1` raises BrokenPipeError on the print.
class ExplodingRender:
    def __init__(self, inner):
        self.inner = inner

    def __call__(self, cmd, cwd):
        return self.inner(cmd, cwd)


_real_render = pr_merge_gate.GateResult.render
try:
    def _boom(self, pr):
        raise KeyError("render blew up")
    pr_merge_gate.GateResult.render = _boom
    # The escape is caught HERE and turned into a value, not left to propagate. Without this, a
    # build whose catch-all is missing kills the suite with a traceback -- which is red, but names
    # nothing and takes every arm below it with it. An arm should say what broke.
    try:
        rc_render = run_main(FakeGh(GREEN_CHECKS, VIEW, LS_ONE))
    except BaseException as exc:  # noqa: BLE001
        rc_render = "escaped %s" % type(exc).__name__
    case("an exception while rendering -> exit 2, never 1", rc_render, 2)
finally:
    pr_merge_gate.GateResult.render = _real_render
case("...and the gate still works afterwards",
     run_main(FakeGh(GREEN_CHECKS, VIEW, LS_ONE)), 0)

print()
if FAILURES:
    for f in FAILURES:
        print("FAIL: %s" % f, file=sys.stderr)
    print("\n%d case(s) failed." % len(FAILURES), file=sys.stderr)
    sys.exit(1)
print("all cases passed")
