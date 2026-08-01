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
# A validation message begins with "Validation Error: [ <id> ]" / "Validation Warning: [ <id> ]".
MESSAGE = re.compile(r"^Validation (Error|Warning|Performance Warning): \[ ([^\]]*) \]", re.M)


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


def parse_allowlist(path: Path) -> dict[str, str]:
    """Read `MESSAGE_ID | reason` lines. Blank lines and `#` comments are ignored."""
    allowed: dict[str, str] = {}
    if not path.exists():
        return allowed
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            raise SystemExit(
                f"{path}:{lineno}: every allow-list entry needs '<message id> | <reason>' — "
                f"an unexplained filter is exactly what this file exists to prevent"
            )
        ident, reason = line.split("|", 1)
        ident, reason = ident.strip(), reason.strip()
        if not reason:
            raise SystemExit(f"{path}:{lineno}: allow-list entry '{ident}' has an empty reason")
        allowed[ident] = reason
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
    cmd = ["ctest", "--timeout", "600"] + extra
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

    new = sorted(k for k in findings if k not in allowed)
    known = sorted(k for k in findings if k in allowed)
    stale = sorted(k for k in allowed if k not in findings)

    if known:
        print()
        print("[vkval] pre-existing, allow-listed (deferred, NOT suppressed):")
        for ident in known:
            f = findings[ident]
            print(f"  {ident}  x{f.count}  [{', '.join(sorted(f.tests))}]")
            print(f"      reason: {allowed[ident]}")
    if stale:
        print()
        print("[vkval] allow-listed but not observed in this run — prune these entries, or record "
              "why they are driver-dependent:")
        for ident in stale:
            print(f"  {ident}      reason on file: {allowed[ident]}")
    if new:
        print()
        print("[vkval] NEW — not on the allow-list:")
        for ident in new:
            f = findings[ident]
            print(f"  {ident}  x{f.count}  [{', '.join(sorted(f.tests))}]")
            for line in f.sample.splitlines():
                print(f"      {line}")

    failed = False
    if new:
        print()
        print(f"[vkval] FAIL: {len(new)} validation message ID(s) are not on the allow-list.")
        print("[vkval] Fix the defect, or add the ID to tools/vkval/allowlist.txt WITH a reason "
              "and (where warranted) an issue. Do not add a bare ID.")
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
