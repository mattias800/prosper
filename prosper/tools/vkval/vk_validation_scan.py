#!/usr/bin/env python3
"""Run prosper's ctest suite under VK_LAYER_KHRONOS_validation and gate on the findings.

Why this exists (#1704): prosper had no Vulkan validation coverage anywhere. #1690 was an invalid
descriptor binding — a VK_IMAGE_VIEW_TYPE_3D view bound to a descriptor whose SPIR-V declared
`OpTypeImage ... 2D` — which is undefined behaviour that three conformant drivers resolved three
different ways. It read as a Mesa version regression and cost a session of bisection plus 484
executed assertions' place on CI. The validation layer names that exact defect at the dispatch
(`VUID-vkCmdDispatch-viewType-07752`) the first time the dispatch runs.

The gate is deliberately shaped so that nothing is silently suppressed:

  * every message ID observed today is written down in `allowlist.txt` with a one-line reason and,
    where one exists, an issue number;
  * any message ID NOT on that list fails the scan;
  * a test that fails or crashes under the layer fails the scan;
  * the layer is proven to have actually loaded before any "clean" verdict is accepted, because a
    layer that silently did not load produces exactly the same empty output as a clean run.

That last point is the trap this tool was written around: the validation layer defaults to
`report_flags = error` only, so a misconfigured probe prints nothing and reads as success.

Usage
-----
    python3 tools/vkval/vk_validation_scan.py --build-dir prosper/build-linux
    python3 tools/vkval/vk_validation_scan.py --build-dir ... --report-only   # measure, never fail
    python3 tools/vkval/vk_validation_scan.py --log LastTest.log              # parse an existing log

Exit status: 0 when every observed message ID is allow-listed and ctest passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import subprocess
import sys
from pathlib import Path

LAYER = "VK_LAYER_KHRONOS_validation"

# ctest writes one of these before each test's captured output in Testing/Temporary/LastTest.log.
TEST_HEADER = re.compile(r"^\d+/\d+ Testing: (.+)$", re.M)

# A validation message carries "Validation Error: [ <id> ]" / "Validation Warning: [ <id> ]" exactly
# once, but WHERE on the line differs by layer version, and this pattern must not be anchored:
#
#   layers 1.4.341 (Fedora 44)   Validation Error: [ VUID-x ] | MessageID = 0x...\n<text>
#   layers 1.3.275 (Ubuntu 24.04, VUID-x(ERROR / SPEC): msgNum: N - Validation Error: [ VUID-x ] ...
#                   the CI runner)
#
# An earlier revision anchored this to the start of a line. It matched all 51 messages on 1.4.341
# and 0 of 187 on 1.3.275 — so the guard was green on CI while observing nothing, which is the
# precise failure this whole tool exists to prevent. The all-absent check in main() is the backstop
# for the next version whose framing nobody anticipated.
MESSAGE = re.compile(r"Validation (Error|Warning|Performance Warning): \[ ([^\]]*) \]")


class Finding:
    """One message ID and where it was seen."""

    def __init__(self, message_id: str) -> None:
        self.message_id = message_id
        self.count = 0
        self.tests: set[str] = set()
        self.sample = ""


def parse_log(text: str) -> dict[str, Finding]:
    """Group validation messages in a ctest LastTest.log by message ID.

    Output of tests that ctest did not attribute (preamble, ctest's own noise) is attributed to the
    pseudo-test "<unattributed>" rather than dropped, so nothing observed can go unreported.
    """
    findings: dict[str, Finding] = {}
    pieces = TEST_HEADER.split(text)
    # pieces = [preamble, name, body, name, body, ...]
    sections = [("<unattributed>", pieces[0])]
    for i in range(1, len(pieces) - 1, 2):
        sections.append((pieces[i].strip(), pieces[i + 1]))

    for test, body in sections:
        for match in MESSAGE.finditer(body):
            ident = match.group(2).strip()
            # The layer prints "[ VUID-foo ] | MessageID = 0x...".  Some messages carry no VUID and
            # print a bare name; keep whatever the layer used as the identity, unchanged.
            ident = ident.split("|")[0].strip() or "<unnamed>"
            entry = findings.setdefault(ident, Finding(ident))
            entry.count += 1
            entry.tests.add(test)
            if not entry.sample:
                # One representative message body: everything up to the blank line after "Objects:".
                tail = body[match.start(): match.start() + 1200]
                entry.sample = tail.split("\n\n")[0].strip()
    return findings


EXPECTATIONS = ("required", "environment-dependent")


class Allowed:
    """One deferred finding: which id, where it may appear, and whether it must still appear."""

    def __init__(self, message_id: str, expectation: str, tests: list[str], reason: str) -> None:
        self.message_id = message_id
        self.expectation = expectation          # one of EXPECTATIONS
        self.tests = tests                      # ctest names, or ["*"]
        self.reason = reason

    def covers(self, test: str) -> bool:
        return "*" in self.tests or test in self.tests


def parse_allowlist(path: Path) -> dict[str, Allowed]:
    """Read `<message id> | <expectation> | <tests> | <reason>` lines.

    Blank lines and `#` comments are ignored. Every other malformation is fatal: this file is the
    project's ledger of deferred defects, and a line nobody can read is indistinguishable from a
    filter somebody added quietly.

    * `<expectation>` is `required` (the id must still be observed — its disappearance means either
      the defect was fixed and this line should go, or the scan stopped seeing things) or
      `environment-dependent` (its absence on some drivers/layer versions is understood and the
      reason says why).
    * `<tests>` is a comma-separated list of ctest names, or `*`. Scoping matters: several VUIDs are
      catch-alls — `VUID-VkShaderModuleCreateInfo-pCode-08737` covers *any* spirv-val error at
      `vkCreateShaderModule` — so an unscoped entry would defer every future occurrence anywhere.
    """
    if not path.exists():
        # Returning {} here would silently disable the all-absent check in main() and hand back a
        # PASS on an empty observation, which is the compound silent-green this tool must not have.
        raise SystemExit(f"[vkval] allow-list {path} does not exist — refusing to scan without a "
                         f"ledger to compare against")
    allowed: dict[str, Allowed] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # maxsplit=3 so a reason may legitimately contain a pipe; too FEW fields still fails.
        fields = [f.strip() for f in line.split("|", 3)]
        if len(fields) != 4:
            raise SystemExit(
                f"{path}:{lineno}: every allow-list entry needs "
                f"'<message id> | <{'|'.join(EXPECTATIONS)}> | <tests or *> | <reason>' — "
                f"an unexplained filter is exactly what this file exists to prevent"
            )
        ident, expectation, tests, reason = fields
        if expectation not in EXPECTATIONS:
            raise SystemExit(f"{path}:{lineno}: allow-list entry '{ident}' has expectation "
                             f"'{expectation}'; it must be one of {', '.join(EXPECTATIONS)}")
        if not reason:
            raise SystemExit(f"{path}:{lineno}: allow-list entry '{ident}' has an empty reason")
        # Validate the PARSED list, not the raw field: ", ," is a non-empty string that names no
        # tests, and an entry covering nothing would otherwise fail later, far from the mistake.
        names = [t.strip() for t in tests.split(",") if t.strip()]
        if not names:
            raise SystemExit(f"{path}:{lineno}: allow-list entry '{ident}' names no tests (use '*' "
                             f"deliberately if the id really may appear anywhere)")
        allowed[ident] = Allowed(ident, expectation, names, reason)
    return allowed


def layer_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    # VK_LOADER_LAYERS_ENABLE is the loader's current force-enable variable (loader >= 1.3.234).
    # VK_INSTANCE_LAYERS is its deprecated predecessor; set both so the scan works on older loaders.
    env["VK_LOADER_LAYERS_ENABLE"] = LAYER
    env["VK_INSTANCE_LAYERS"] = LAYER
    return env


def prove_layer_loads(build_dir: Path, probe: str) -> str:
    """Fail loudly unless the loader reports inserting the validation layer.

    Without this, "no findings" is ambiguous between a clean suite and a layer that never loaded,
    and the second reading is the one that quietly retires the guard.
    """
    exe = build_dir / probe
    if not exe.exists():
        raise SystemExit(
            f"[vkval] probe binary {exe} is missing — the Vulkan-gated tests were not built, so "
            f"there is nothing for the validation layer to observe (see #1675)"
        )
    env = layer_env()
    env["VK_LOADER_DEBUG"] = "layer"
    proc = subprocess.run([str(exe)], cwd=build_dir, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    inserted = [l for l in proc.stdout.splitlines()
                if "Insert instance layer" in l and LAYER in l]
    if not inserted:
        raise SystemExit(
            f"[vkval] the Vulkan loader never inserted {LAYER} while running {probe}.\n"
            f"        Install it (Debian/Ubuntu: vulkan-validationlayers, Fedora: "
            f"vulkan-validation-layers) — a scan without the layer reports zero findings and "
            f"means nothing."
        )
    return inserted[0].strip()


def run_ctest(build_dir: Path, extra: list[str]) -> tuple[int, str, str]:
    # --no-tests=error closes the "build directory with nothing registered" variant of a silent
    # green; --output-on-failure means a test that fails only under the layer arrives in the CI log
    # with its output attached, not just a summary line.
    cmd = ["ctest", "--timeout", "600", "--no-tests=error", "--output-on-failure"] + extra
    proc = subprocess.run(cmd, cwd=build_dir, env=layer_env(),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = build_dir / "Testing" / "Temporary" / "LastTest.log"
    if not log.exists():
        raise SystemExit(f"[vkval] ctest produced no {log} — cannot scan its output")
    return proc.returncode, proc.stdout, log.read_text(encoding="utf-8", errors="replace")


def main(argv: list[str]) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", type=Path,
                    help="CMake build directory containing the built ctest suite")
    ap.add_argument("--log", type=Path,
                    help="parse an existing ctest LastTest.log instead of running ctest")
    ap.add_argument("--allowlist", type=Path, default=here / "allowlist.txt")
    ap.add_argument("--probe", default="test_vulkan_offscreen",
                    help="Vulkan test binary used to prove the layer loads")
    ap.add_argument("--report-only", action="store_true",
                    help="print the grouped findings and always exit 0 (measurement mode)")
    ap.add_argument("--ctest-arg", action="append", default=[],
                    help="extra argument forwarded to ctest (repeatable)")
    args = ap.parse_args(argv)

    if not args.log and not args.build_dir:
        ap.error("one of --build-dir or --log is required")

    ctest_rc, ctest_out, log_text = 0, "", ""
    if args.log:
        log_text = args.log.read_text(encoding="utf-8", errors="replace")
    else:
        build_dir = args.build_dir.resolve()
        print(f"[vkval] proving {LAYER} loads ...")
        print(f"[vkval]   {prove_layer_loads(build_dir, args.probe)}")
        print(f"[vkval] running ctest under {LAYER} ...")
        ctest_rc, ctest_out, log_text = run_ctest(build_dir, args.ctest_arg)
        print(ctest_out.rstrip())

    findings = parse_log(log_text)
    allowed = parse_allowlist(args.allowlist)

    total = sum(f.count for f in findings.values())
    print()
    print(f"[vkval] {len(findings)} distinct message ID(s), {total} message(s) total")

    # An observation is accounted for only if BOTH its id and the test it came from are on the
    # ledger. Scoping by test is not pedantry: several VUIDs are catch-alls (pCode-08737 is every
    # spirv-val error at vkCreateShaderModule; format-07753 is the graphics-side sibling of the
    # exact defect class this guard exists to catch), so an id-only allow-list would defer every
    # future occurrence of those anywhere in the suite.
    #
    # Two consequences worth knowing before adding to this ledger. The test list must be the UNION
    # across drivers, because attribution varies: an id can reach one extra test on hardware that
    # never appears on lavapipe. And "<unattributed>" — the pseudo-test for a message ctest could
    # not place, which today only happens if the layer prints outside any test's captured output —
    # is covered by nothing, so such a message always fails. That is deliberate (a message we cannot
    # locate is not a message we have deferred), but it is a failure mode scoping created.
    new = sorted(k for k in findings if k not in allowed)
    misplaced = sorted((k, t) for k in findings if k in allowed
                       for t in findings[k].tests if not allowed[k].covers(t))
    known = sorted(k for k in findings if k in allowed)
    absent = sorted(k for k in allowed if k not in findings)
    absent_required = [k for k in absent if allowed[k].expectation == "required"]
    absent_env = [k for k in absent if allowed[k].expectation != "required"]

    if known:
        print()
        print("[vkval] pre-existing, allow-listed (deferred, NOT suppressed):")
        for ident in known:
            f = findings[ident]
            print(f"  {ident}  x{f.count}  [{', '.join(sorted(f.tests))}]")
            print(f"      reason: {allowed[ident].reason}")
    if absent_env:
        print()
        print("[vkval] allow-listed, environment-dependent, not observed here (expected):")
        for ident in absent_env:
            print(f"  {ident}      {allowed[ident].reason}")
    if absent_required:
        print()
        print("[vkval] allow-listed as REQUIRED but not observed:")
        for ident in absent_required:
            print(f"  {ident}      reason on file: {allowed[ident].reason}")
    if misplaced:
        print()
        print("[vkval] allow-listed id, but from a test the ledger does not cover:")
        for ident, test in misplaced:
            print(f"  {ident}  in {test}  (recorded for: {', '.join(allowed[ident].tests)})")
    if new:
        print()
        print("[vkval] NEW — not on the allow-list:")
        for ident in new:
            f = findings[ident]
            print(f"  {ident}  x{f.count}  [{', '.join(sorted(f.tests))}]")
            for line in f.sample.splitlines():
                print(f"      {line}")

    failed = False
    # Instrument check, and the one that matters most. Every `required` entry was observed when the
    # guard was switched on. If one stops appearing, exactly two things can have happened: the
    # defect was fixed (delete the line, in the same change) or the observation broke — a layer
    # version whose framing this parser does not match, a renamed VUID, a ctest that wrote its log
    # somewhere else, a build without the Vulkan tests. All of the second kind report "nothing
    # here", which is indistinguishable from success unless something asks the question.
    #
    # Checking each `required` id separately rather than only the all-absent case is what catches a
    # PARTIAL break: a rename or a framing change that affects some message shapes and not others
    # would otherwise pass while silently halving what the guard can see. `environment-dependent`
    # entries are exempt, and their reason line has to say why (currently a check newer than layers
    # 1.3.275) — the exemption is written down, not assumed.
    if absent_required:
        print()
        print(f"[vkval] FAIL: {len(absent_required)} allow-listed id(s) marked `required` produced "
              f"no messages, while the layer demonstrably loaded.")
        print("[vkval] Either the defect was fixed — in which case delete the entry in the same "
              "change — or this scan is no longer seeing what it used to. Check that the parser "
              "matches your layer version's message framing and that the id was not renamed.")
        print("[vkval] If the absence is a property of this driver or layer version, mark the entry "
              "`environment-dependent` and say why on its reason line.")
        failed = True
    if misplaced:
        print()
        print(f"[vkval] FAIL: {len(misplaced)} allow-listed id(s) appeared in a test the ledger does "
              f"not record. A deferral is scoped to where it was measured, so this is a new SITE "
              f"until someone confirms it is the same defect — several of these VUIDs are catch-alls."
              )
        print("[vkval] If it is the same defect (a renamed test, a new test exercising the same "
              "code, another driver attributing it differently), add the test to that entry. If it "
              "is not, it is a new finding and needs its own line and its own issue.")
        failed = True
    if new:
        print()
        print(f"[vkval] FAIL: {len(new)} validation message ID(s) are not on the allow-list.")
        print("[vkval] Fix the defect, or add the ID to tools/vkval/allowlist.txt WITH an "
              "expectation, the tests it comes from, a reason and (where warranted) an issue.")
        failed = True
    if ctest_rc != 0:
        print()
        print(f"[vkval] FAIL: ctest exited {ctest_rc} under the validation layer. A test that "
              f"passes without the layer and fails with it is a real finding, not layer noise.")
        failed = True

    if args.report_only:
        print("[vkval] --report-only: exiting 0 regardless of findings")
        return 0
    if not failed:
        print()
        print("[vkval] PASS: every observed validation message ID is accounted for.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
