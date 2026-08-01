#!/usr/bin/env python3
"""Unit tests for the Vulkan-validation scan's parsing and gating logic (#1704).

Registered as ctest `vkval_scan_logic`. This guard is only worth having if its parser really
recognises the layer's output, so each test below fails if the corresponding behaviour regresses:
a parser that quietly matches nothing reports a clean suite, which is the exact failure mode the
scan exists to prevent.
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vk_validation_scan as scan  # noqa: E402

FAILURES = []
CHECKS = 0
# Everything this test writes to stdout, kept so the final check can prove the test does not
# contaminate the very scan it is testing — see the note on run_main().
EMITTED = []


def say(line):
    EMITTED.append(line)
    print(line)


def check(cond, what):
    global CHECKS
    CHECKS += 1
    say(f"  [ok]   {what}" if cond else f"  [FAIL] {what}")
    if not cond:
        FAILURES.append(what)


def run_main(argv):
    """Call scan.main() with its console output SWALLOWED, returning only the exit code.

    This is not tidiness. `vk_validation_scan.py` prints a representative message body for every
    new finding, and ctest captures this test's stdout into the same LastTest.log the scan parses.
    Echoing it here would inject `VUID-vkCmdDispatch-viewType-07752` into the suite's log as though
    a real dispatch had produced it — the guard reading its own tail and failing on clean master.
    Measured: it did exactly that before this wrapper existed.
    """
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        return scan.main(argv)


# A faithful excerpt of a real ctest LastTest.log written under VK_LAYER_KHRONOS_validation, with
# the exact framing the layer and ctest produce (message header, spec sentence, Objects block).
SAMPLE_LOG = """\
Start testing: Aug 01 20:00 UTC
----------------------------------------------------------
1/3 Testing: vulkan_offscreen_clear
1/3 Test: vulkan_offscreen_clear
Command: "/b/test_vulkan_offscreen"
Output:
----------------------------------------------------------
== PASS ==
<end of output>
Test Passed.
2/3 Testing: texture_sample_render
2/3 Test: texture_sample_render
Command: "/b/test_texture_sample_render"
Output:
----------------------------------------------------------
Validation Error: [ VUID-vkCmdDraw-format-07753 ] | MessageID = 0x7be8f3b5
vkCmdDraw(): the storage image descriptor requires UINT component type, but bound descriptor \
format is VK_FORMAT_R8G8B8A8_UNORM.
The Vulkan spec states: ...
Objects: 1
    [0] VkPipeline 0x1

Validation Error: [ VUID-vkCmdDraw-format-07753 ] | MessageID = 0x7be8f3b5
vkCmdDraw(): again
<end of output>
Test Passed.
3/3 Testing: game_compute_exec
3/3 Test: game_compute_exec
Command: "/b/test_game_compute"
Output:
----------------------------------------------------------
Validation Error: [ VUID-vkCmdDispatch-viewType-07752 ] | MessageID = 0x6174abc7
vkCmdDispatch(): VkImageViewType is VK_IMAGE_VIEW_TYPE_3D but the OpTypeImage has (Dim = 2D).
<end of output>
Test Passed.
"""


# Validation layers 1.3.275 — what Ubuntu 24.04 ships, i.e. what the CI runner actually loads —
# print the whole message on ONE line, PREFIXED by the VUID and a msgNum. Verbatim framing from a
# real run in `podman run --rm ubuntu:24.04`.
#
# This constant exists because an anchored parser matched all 51 messages on layers 1.4.341 and
# 0 of 187 here, which would have made the CI gate permanently, invisibly green. Do not "simplify"
# this test by regenerating both samples from one layer version: the whole point is that two
# supported versions frame the same message differently.
SAMPLE_LOG_1_3_275 = """\
1/2 Testing: texture_sample_render
Output:
----------------------------------------------------------
VUID-vkCmdDraw-format-07753(ERROR / SPEC): msgNum: -1520283006 - Validation Error: \
[ VUID-vkCmdDraw-format-07753 ] Object 0: handle = 0x1, type = VK_OBJECT_TYPE_PIPELINE; | \
MessageID = 0x7be8f3b5 | vkCmdDraw(): the storage image descriptor requires UINT component type, \
but bound descriptor format is VK_FORMAT_R8G8B8A8_UNORM. The Vulkan spec states: ...
    Objects: 1
        [0] 0x1, type: 19, name: NULL
<end of output>
Test Passed.
2/2 Testing: game_compute_exec
Output:
----------------------------------------------------------
VUID-vkCmdDispatch-viewType-07752(ERROR / SPEC): msgNum: 1635426247 - Validation Error: \
[ VUID-vkCmdDispatch-viewType-07752 ] Object 0: handle = 0x2, type = VK_OBJECT_TYPE_PIPELINE; | \
MessageID = 0x6174abc7 | vkCmdDispatch(): VkImageViewType is VK_IMAGE_VIEW_TYPE_3D but the \
OpTypeImage has (Dim = 2D).
<end of output>
Test Passed.
"""


def main():
    say("== test_vk_validation_scan ==")

    findings = scan.parse_log(SAMPLE_LOG)
    check(set(findings) == {"VUID-vkCmdDraw-format-07753", "VUID-vkCmdDispatch-viewType-07752"},
          "parser finds exactly the two message IDs present in the log")
    check(findings["VUID-vkCmdDraw-format-07753"].count == 2,
          "repeat occurrences of one ID are counted, not deduplicated")
    check(findings["VUID-vkCmdDraw-format-07753"].tests == {"texture_sample_render"},
          "a message is attributed to the ctest test whose output contains it")
    check(findings["VUID-vkCmdDispatch-viewType-07752"].tests == {"game_compute_exec"},
          "a second test's messages are attributed to that test, not the first")
    check("VK_IMAGE_VIEW_TYPE_3D" in findings["VUID-vkCmdDispatch-viewType-07752"].sample,
          "a representative message body is retained for reporting")

    # The #1690 defect class specifically: this VUID must be recognised, because it is the whole
    # justification for the guard. If the parser stopped matching it the scan would go green.
    check("VUID-vkCmdDispatch-viewType-07752" in findings,
          "the view-type/module-Dim mismatch VUID (#1690's defect) is recognised")

    # The CI runner's layer version frames every message differently. This is the regression that
    # nearly shipped: an anchored pattern reads 0 of 187 messages here and the gate goes green.
    old = scan.parse_log(SAMPLE_LOG_1_3_275)
    check(set(old) == {"VUID-vkCmdDraw-format-07753", "VUID-vkCmdDispatch-viewType-07752"},
          "layers 1.3.275 framing (VUID-x(ERROR / SPEC): msgNum: N - Validation Error: [...]) parses")
    check(old["VUID-vkCmdDraw-format-07753"].count == 1
          and old["VUID-vkCmdDispatch-viewType-07752"].count == 1,
          "each 1.3.275 message is counted exactly once despite repeating its id on the same line")
    check(old["VUID-vkCmdDispatch-viewType-07752"].tests == {"game_compute_exec"},
          "1.3.275 messages are attributed to the right test")

    # Validation warnings are messages too, and must not be dropped.
    warned = scan.parse_log(
        "1/1 Testing: t\nValidation Warning: [ WARNING-cache-file-error ] | MessageID = 0x1\nbody\n")
    check(set(warned) == {"WARNING-cache-file-error"},
          "Validation Warning messages are captured as well as Errors")

    # Messages printed before ctest's first test header must still be reported.
    stray = scan.parse_log("preamble\nValidation Error: [ VUID-stray-1 ] | MessageID = 0x2\nbody\n")
    check(set(stray) == {"VUID-stray-1"} and stray["VUID-stray-1"].tests == {"<unattributed>"},
          "messages outside any test section are reported as <unattributed>, never dropped")

    with tempfile.TemporaryDirectory() as td:
        def allowfile(name, text):
            p = Path(td) / name
            p.write_text(text)
            return str(p)

        good = allowfile("ok.txt",
                         "# comment\n\nVUID-a | required | test_x, test_y | because reasons (#1)\n")
        allowed = scan.parse_allowlist(Path(good))
        check(set(allowed) == {"VUID-a"} and allowed["VUID-a"].expectation == "required"
              and allowed["VUID-a"].tests == ["test_x", "test_y"]
              and allowed["VUID-a"].reason == "because reasons (#1)",
              "allow-list parses 'id | expectation | tests | reason', skipping comments and blanks")
        check(allowed["VUID-a"].covers("test_x") and not allowed["VUID-a"].covers("test_z"),
              "an entry covers only the tests it names")
        star = scan.parse_allowlist(Path(allowfile("star.txt", "VUID-a | required | * | any (#1)\n")))
        check(star["VUID-a"].covers("anything at all"), "'*' covers any test")

        piped = scan.parse_allowlist(Path(allowfile(
            "piped.txt", "VUID-a | required | t | see A|B, and C|D (#1)\n")))
        check(piped["VUID-a"].reason == "see A|B, and C|D (#1)",
              "a reason may contain '|' — only the first three separators are structural")

        # Every malformation is fatal: a line nobody can read is indistinguishable from a filter
        # somebody added quietly, which is the thing this file exists to make impossible.
        for name, text, what in [
                ("bare.txt", "VUID-a\n", "a bare id with no fields"),
                ("noreason.txt", "VUID-a | required | test_x |   \n", "an empty reason"),
                ("notests.txt", "VUID-a | required |  | why (#1)\n", "an empty test list"),
                ("commatests.txt", "VUID-a | required | , , | why (#1)\n",
                 "a test list that is non-empty text but names nothing"),
                ("badexp.txt", "VUID-a | maybe | test_x | why (#1)\n", "an unknown expectation"),
                ("shortform.txt", "VUID-a | why (#1)\n", "the old two-field form")]:
            try:
                scan.parse_allowlist(Path(allowfile(name, text)))
                check(False, f"allow-list entry with {what} is rejected")
            except SystemExit:
                check(True, f"allow-list entry with {what} is rejected")

        try:
            scan.parse_allowlist(Path(td) / "does-not-exist.txt")
            check(False, "a MISSING allow-list is rejected, not treated as empty")
        except SystemExit:
            check(True, "a MISSING allow-list is rejected, not treated as empty")

        # End-to-end gating, driven through main() the way CI drives it.
        log = Path(td) / "LastTest.log"
        log.write_text(SAMPLE_LOG)

        covers_all = allowfile("all.txt",
            "VUID-vkCmdDraw-format-07753 | required | texture_sample_render | pre-existing (#x)\n"
            "VUID-vkCmdDispatch-viewType-07752 | required | game_compute_exec | pre-existing (#y)\n")
        rc = run_main(["--log", str(log), "--allowlist", covers_all])
        check(rc == 0, "a run whose every id AND test is on the ledger passes")

        partial = allowfile("partial.txt",
            "VUID-vkCmdDraw-format-07753 | required | texture_sample_render | pre-existing (#x)\n")
        rc = run_main(["--log", str(log), "--allowlist", partial])
        check(rc == 1, "a message id missing from the allow-list fails the scan")

        rc = run_main(["--log", str(log), "--allowlist", partial, "--report-only"])
        check(rc == 0, "--report-only measures without failing")

        # Scoping is the point of the tests column: a catch-all VUID appearing somewhere new is a
        # new defect wearing an old id, and an id-only ledger would wave it through.
        wrong_test = allowfile("wrongtest.txt",
            "VUID-vkCmdDraw-format-07753 | required | texture_sample_render | pre-existing (#x)\n"
            "VUID-vkCmdDispatch-viewType-07752 | required | some_other_test | pre-existing (#y)\n")
        rc = run_main(["--log", str(log), "--allowlist", wrong_test])
        check(rc == 1, "an allow-listed id arriving from an unrecorded test fails the scan")

        # A `required` entry that stops firing is either a fix nobody pruned or an observation that
        # broke. Failing on it is what catches a PARTIAL break — one id going quiet while the rest
        # still report — which the all-absent check alone lets through.
        stale_required = allowfile("stale.txt",
            "VUID-vkCmdDraw-format-07753 | required | texture_sample_render | pre-existing (#x)\n"
            "VUID-vkCmdDispatch-viewType-07752 | required | game_compute_exec | pre-existing (#y)\n"
            "VUID-never-seen | required | some_test | fixed, entry not yet pruned\n")
        rc = run_main(["--log", str(log), "--allowlist", stale_required])
        check(rc == 1, "a `required` id that produced no messages fails the scan")

        stale_env = allowfile("staleenv.txt",
            "VUID-vkCmdDraw-format-07753 | required | texture_sample_render | pre-existing (#x)\n"
            "VUID-vkCmdDispatch-viewType-07752 | required | game_compute_exec | pre-existing (#y)\n"
            "VUID-never-seen | environment-dependent | some_test | cannot fire on lavapipe (#z)\n")
        rc = run_main(["--log", str(log), "--allowlist", stale_env])
        check(rc == 0, "an `environment-dependent` id that produced no messages is reported, not fatal")

        # The total-break shape, still covered: every required id silent at once.
        empty_log = Path(td) / "empty.log"
        empty_log.write_text("Start testing\n1/1 Testing: t\n== PASS ==\n")
        check(scan.parse_log(empty_log.read_text()) == {},
              "a log with no validation messages yields no findings")
        rc = run_main(["--log", str(empty_log), "--allowlist", covers_all])
        check(rc == 1,
              "zero observed messages against a non-empty allow-list fails as a broken observation")

        # ... and the gate switches itself off once the defects are fixed and their lines deleted.
        empty_allow = allowfile("none.txt", "# every pre-existing finding has been fixed\n")
        rc = run_main(["--log", str(empty_log), "--allowlist", empty_allow])
        check(rc == 0, "an empty allow-list plus an empty log is the intended clean end state")

        # The real ledger must itself be well-formed, and must cover what the guard measured.
        real = scan.parse_allowlist(Path(__file__).resolve().parent / "allowlist.txt")
        check(len(real) >= 1 and all(a.reason and a.tests for a in real.values()),
              "the checked-in allowlist.txt parses, and every entry carries tests and a reason")
        check(sum(1 for a in real.values() if a.expectation == "environment-dependent") == 2,
              "exactly two checked-in entries claim to be environment-dependent")

    # The guard must not read its own tail. ctest captures this test's stdout into the same
    # LastTest.log that vk_validation_scan.py parses, so anything printed here that LOOKS like a
    # validation message is indistinguishable from one a real draw produced — and this file
    # necessarily traffics in validation-message text. Parse everything this run emitted and
    # require zero findings.
    check(scan.parse_log("\n".join(EMITTED)) == {},
          "this test's own output contains nothing the scan would read as a validation message")

    say(f"== {'PASS' if not FAILURES else 'FAIL'} == ({CHECKS} assertions executed)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
