#!/usr/bin/env python3
"""Unit tests for the Vulkan-validation scan's parsing and gating logic (#1704).

Registered as ctest `vkval_scan_logic`. This guard is only worth having if its parser really
recognises the layer's output, so each test below fails if the corresponding behaviour regresses:
a parser that quietly matches nothing reports a clean suite, which is the exact failure mode the
scan exists to prevent.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vk_validation_scan as scan  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, what):
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  [ok]   {what}")
    else:
        print(f"  [FAIL] {what}")
        FAILURES.append(what)


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


def main():
    print("== test_vk_validation_scan ==")

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
        good = Path(td) / "ok.txt"
        good.write_text("# comment\n\nVUID-a | because reasons (#1)\n")
        allowed = scan.parse_allowlist(good)
        check(allowed == {"VUID-a": "because reasons (#1)"},
              "allow-list parses 'id | reason', ignoring comments and blank lines")

        bare = Path(td) / "bare.txt"
        bare.write_text("VUID-a\n")
        try:
            scan.parse_allowlist(bare)
            check(False, "a bare allow-list ID with no reason is rejected")
        except SystemExit:
            check(True, "a bare allow-list ID with no reason is rejected")

        empty_reason = Path(td) / "empty.txt"
        empty_reason.write_text("VUID-a |   \n")
        try:
            scan.parse_allowlist(empty_reason)
            check(False, "an allow-list entry with an empty reason is rejected")
        except SystemExit:
            check(True, "an allow-list entry with an empty reason is rejected")

        # End-to-end gating, driven through main() the way CI drives it.
        log = Path(td) / "LastTest.log"
        log.write_text(SAMPLE_LOG)

        covers_all = Path(td) / "all.txt"
        covers_all.write_text("VUID-vkCmdDraw-format-07753 | pre-existing (#x)\n"
                              "VUID-vkCmdDispatch-viewType-07752 | pre-existing (#y)\n")
        rc = scan.main(["--log", str(log), "--allowlist", str(covers_all)])
        check(rc == 0, "a run whose every message ID is allow-listed passes")

        partial = Path(td) / "partial.txt"
        partial.write_text("VUID-vkCmdDraw-format-07753 | pre-existing (#x)\n")
        rc = scan.main(["--log", str(log), "--allowlist", str(partial)])
        check(rc == 1, "a message ID missing from the allow-list fails the scan")

        rc = scan.main(["--log", str(log), "--allowlist", str(partial), "--report-only"])
        check(rc == 0, "--report-only measures without failing")

        stale = Path(td) / "stale.txt"
        stale.write_text("VUID-vkCmdDraw-format-07753 | pre-existing (#x)\n"
                         "VUID-vkCmdDispatch-viewType-07752 | pre-existing (#y)\n"
                         "VUID-never-seen | fixed, entry not yet pruned\n")
        rc = scan.main(["--log", str(log), "--allowlist", str(stale)])
        check(rc == 0, "an allow-listed ID that no longer fires is reported but does not fail "
                       "(the observed set is driver-dependent)")

        # An empty log with a populated allow-list is the "layer never loaded" shape. The scan
        # cannot detect that from the log alone — prove_layer_loads() is what covers it — so this
        # only pins that the parser reports zero rather than throwing.
        empty_log = Path(td) / "empty.log"
        empty_log.write_text("Start testing\n1/1 Testing: t\n== PASS ==\n")
        check(scan.parse_log(empty_log.read_text()) == {},
              "a log with no validation messages yields no findings")

    print(f"== {'PASS' if not FAILURES else 'FAIL'} == ({CHECKS} assertions executed)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
