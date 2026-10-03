#!/usr/bin/env python3
"""Enforce that pull request descriptions adhere to the required PR template.

Why this gate exists
--------------------
Prosper relies on structured pull request descriptions for automated triage, reviewer
clarity, verification proof, and auditability. The template defined in
`.github/pull_request_template.md` specifies required top-level sections:

    - Context
    - Higher Goal
    - Acceptance Criteria
    - Out of Scope
    - Summary of Changes
    - Verification
    - Checklist

When contributors omit sections or leave unfilled template placeholders (HTML comments),
reviewers must spend time requesting information that should have accompanied the PR from the
outset.

This tool checks that:
1. Every required heading is present as a Markdown level-2 header (`## <Heading>`).
2. Placeholder comments (`<!-- ... -->`) are ignored when determining substantive content.
3. Every required section contains substantive, non-whitespace content.
4. Acceptance criteria contains at least one checked or unchecked task checkbox or prose item.
5. The Verification section contains non-empty responses for the standard verification questions:
   - Tests added
   - Red without the fix
   - Commands run + results

Self-validation
---------------
Like sibling CI gates (`check_contribution_shape.py`, `check_ctest_gate.py`), this tool
fails silently when it breaks (a matcher that stops matching would pass every PR).
Therefore `--selftest` tests hand-written violating AND compliant PR bodies across multiple
failure modes (missing heading, unfilled template, empty section, whitespace only, etc.).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED_SECTIONS: list[str] = [
    "Context",
    "Higher Goal",
    "Acceptance Criteria",
    "Out of Scope",
    "Summary of Changes",
    "Verification",
    "Checklist",
]

VERIFICATION_SUBHEADINGS: list[str] = [
    "Tests added:",
    "Red without the fix",
    "Commands run + results:",
]


def strip_html_comments(text: str) -> str:
    """Remove HTML comments (<!-- ... -->) across multiple lines."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def extract_sections(body: str) -> dict[str, str]:
    """Parse Markdown level-2 sections (## Section Name).

    Returns a mapping of canonical section name (title-cased/normalized) to its body content.
    """
    sections: dict[str, str] = {}
    current_section: str | None = None
    section_lines: list[str] = []

    for line in body.splitlines():
        heading_match = re.match(r"^##\s+(.+?)\s*$", line)
        if heading_match:
            if current_section is not None:
                sections[current_section] = "\n".join(section_lines).strip()
            current_section = heading_match.group(1).strip()
            section_lines = []
        elif current_section is not None:
            section_lines.append(line)

    if current_section is not None:
        sections[current_section] = "\n".join(section_lines).strip()

    return sections


def check_pr_body(body: str) -> list[str]:
    """Check PR body against template rules.

    Returns a list of violation strings. Empty list indicates full compliance.
    """
    if not body or not body.strip():
        return ["PR description is empty."]

    raw_sections = extract_sections(body)
    # Normalize section titles case-insensitively for lookup, while preserving canonical names
    lower_to_key = {k.strip().lower(): k for k in raw_sections}

    problems: list[str] = []

    for req in REQUIRED_SECTIONS:
        req_lower = req.lower()
        if req_lower not in lower_to_key:
            problems.append(f"Missing required section: '## {req}'")
            continue

        raw_content = raw_sections[lower_to_key[req_lower]]
        cleaned_content = strip_html_comments(raw_content).strip()

        if not cleaned_content:
            problems.append(
                f"Section '## {req}' is empty (only placeholder comments or whitespace)."
            )
            continue

        # Acceptance Criteria check: must have at least one checklist item or item entry
        if req_lower == "acceptance criteria":
            # Strip empty checkbox placeholder `- [ ]` if nothing else follows
            meaningful_lines = [
                line.strip()
                for line in cleaned_content.splitlines()
                if line.strip() and line.strip() not in ("- [ ]", "* [ ]", "- []", "* []")
            ]
            if not meaningful_lines:
                problems.append("Section '## Acceptance Criteria' has no filled acceptance items.")

        # Verification check: check that standard prompts have substantive content
        elif req_lower == "verification":
            for sub_name, pattern in [
                ("Tests added:", r"[-*]\s*Tests added:[ \t]*([^\r\n]*)"),
                (
                    "Red without the fix",
                    r"[-*]\s*Red without the fix(?:\s*\([^)]*\))?:?[ \t]*([^\r\n]*)",
                ),
                (
                    "Commands run + results:",
                    r"[-*]\s*Commands run \+ results:[ \t]*([^\r\n]*)",
                ),
            ]:
                m = re.search(pattern, cleaned_content, re.IGNORECASE)
                if not m:
                    problems.append(
                        f"Section '## Verification' is missing required prompt: '{sub_name}'"
                    )
                else:
                    ans = m.group(1).strip()
                    # Check if answer was left completely blank on the same line
                    # and no indented / bullet continuation immediately following
                    if not ans:
                        # Inspect remaining text following match to see if subsequent lines have content
                        # before next top-level prompt bullet or section
                        rest = cleaned_content[m.end() :]
                        next_heading = re.search(
                            r"\n\s*[-*]\s*(?:Tests added|Red without the fix|Commands run \+ results|Could not verify)|"
                            r"\n\s*[-*]\s*[A-Z][^:\n]+:|\n##",
                            rest,
                            re.IGNORECASE,
                        )
                        sub_content = (
                            rest[: next_heading.start()].strip() if next_heading else rest.strip()
                        )
                        if not sub_content:
                            problems.append(
                                f"Section '## Verification' prompt '{sub_name}' is not answered."
                            )

        # Checklist check: ensure checklist has items
        elif req_lower == "checklist":
            meaningful_lines = [
                line.strip() for line in cleaned_content.splitlines() if line.strip()
            ]
            if not meaningful_lines:
                problems.append("Section '## Checklist' contains no checklist items.")

    return problems


def check_event_file(event_path: Path) -> list[str]:
    """Extract PR body from GitHub event json payload and validate."""
    if not event_path.is_file():
        return [f"GitHub event file not found: {event_path}"]
    try:
        data = json.loads(event_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Failed to read GitHub event file: {exc}"]

    pr_data = data.get("pull_request")
    if not pr_data:
        # Not a pull_request event
        return []

    body = pr_data.get("body") or ""
    return check_pr_body(body)


def selftest() -> int:
    """Verify check_pr_body against positive and negative hand-written examples."""
    valid_pr_body = """## Context

Root problem: fix memory leak in audio playback. Fixes #1234.

## Higher Goal

Protect host stability during long game sessions.

## Acceptance Criteria

- [x] Audio buffer freed on playback stop.
- [ ] Leak test passes under ASan.

## Out of Scope

Re-architecting SDL3 audio thread management.

## Summary of Changes

Free allocated buffer in `stop_audio()`.

## Verification

- Tests added: `test_audio_buffer_leak`
- Red without the fix (how confirmed): reverted fix, observed 40 MB leak in 100 iterations.
- Commands run + results: `ctest --test-dir build --no-tests=error` (350 passed, 0 failed).
- Could not verify (no dump / no GPU): PS5 device audio timing.

## Checklist

- [x] Behavior changes have a meaningful regression.
- [x] New source/test files live under `prosper/`.
- [x] No Sony code/keys/firmware.
- [x] No unconditional "owned" answers.
- [x] `git diff --check` clean.
"""

    valid_multiline_verification = """## Context

Fix shader compilation error. Refs #5678.

## Higher Goal

Render compute passes correctly.

## Acceptance Criteria

* Compute dispatch finishes without validation error.

## Out of Scope

Fragment shader changes.

## Summary of Changes

Added missing SPIR-V descriptor mapping.

## Verification

- Tests added:
  - `compute_authority_census_policy`
- Red without the fix (how confirmed):
  Threw unhandled exception on missing binding.
- Commands run + results:
  Ran test suite with spirv-val enabled. All passed.
- Could not verify (no dump / no GPU): N/A

## Checklist

- [x] Compliant.
"""

    must_pass: list[tuple[str, str]] = [
        ("Fully compliant standard PR body", valid_pr_body),
        ("Compliant with multiline verification items", valid_multiline_verification),
    ]

    must_fail: list[tuple[str, str, str]] = [
        ("Empty body", "", "PR description is empty"),
        ("Whitespace body", "   \n\t  \n", "PR description is empty"),
        (
            "Unedited raw template (only HTML comments)",
            """## Context
<!-- Root problem, failure scenario... -->
## Higher Goal
<!-- What this unlocks... -->
## Acceptance Criteria
- [ ]
## Out of Scope
<!-- Boundaries and deferred work... -->
## Summary of Changes
<!-- Files, behavior contract... -->
## Verification
<!-- Exact commands... -->
- Tests added:
- Red without the fix (how confirmed):
- Commands run + results:
- Could not verify (no dump / no GPU):
## Checklist
- [ ] Check item
""",
            "is empty",
        ),
        (
            "Missing '## Context' section",
            valid_pr_body.replace("## Context", "## Background"),
            "Missing required section: '## Context'",
        ),
        (
            "Missing '## Higher Goal' section",
            valid_pr_body.replace("## Higher Goal", "## Motivation"),
            "Missing required section: '## Higher Goal'",
        ),
        (
            "Missing '## Acceptance Criteria' section",
            valid_pr_body.replace("## Acceptance Criteria", "## Criteria"),
            "Missing required section: '## Acceptance Criteria'",
        ),
        (
            "Missing '## Out of Scope' section",
            valid_pr_body.replace("## Out of Scope", "## Future Work"),
            "Missing required section: '## Out of Scope'",
        ),
        (
            "Missing '## Summary of Changes' section",
            valid_pr_body.replace("## Summary of Changes", "## Changes"),
            "Missing required section: '## Summary of Changes'",
        ),
        (
            "Missing '## Verification' section",
            valid_pr_body.split("## Verification")[0] + "## Checklist\n- [x] ok\n",
            "Missing required section: '## Verification'",
        ),
        (
            "Missing '## Checklist' section",
            valid_pr_body.split("## Checklist")[0],
            "Missing required section: '## Checklist'",
        ),
        (
            "Empty '## Summary of Changes' section",
            valid_pr_body.replace("Free allocated buffer in `stop_audio()`.", ""),
            "Section '## Summary of Changes' is empty",
        ),
        (
            "Empty '## Context' section",
            valid_pr_body.replace(
                "Root problem: fix memory leak in audio playback. Fixes #1234.", ""
            ),
            "Section '## Context' is empty",
        ),
        (
            "Empty '## Higher Goal' section",
            valid_pr_body.replace("Protect host stability during long game sessions.", ""),
            "Section '## Higher Goal' is empty",
        ),
        (
            "Empty '## Out of Scope' section",
            valid_pr_body.replace("Re-architecting SDL3 audio thread management.", ""),
            "Section '## Out of Scope' is empty",
        ),
        (
            "Unfilled Acceptance Criteria checkbox",
            valid_pr_body.replace(
                "- [x] Audio buffer freed on playback stop.\n- [ ] Leak test passes under ASan.",
                "- [ ]",
            ),
            "Section '## Acceptance Criteria' has no filled acceptance items",
        ),
        (
            "Unanswered Verification prompt Tests added",
            valid_pr_body.replace("`test_audio_buffer_leak`", ""),
            "prompt 'Tests added:' is not answered",
        ),
        (
            "Unanswered Verification prompt Red without the fix",
            valid_pr_body.replace("reverted fix, observed 40 MB leak in 100 iterations.", ""),
            "prompt 'Red without the fix' is not answered",
        ),
        (
            "Unanswered Verification prompt Commands run + results",
            valid_pr_body.replace(
                "`ctest --test-dir build --no-tests=error` (350 passed, 0 failed).", ""
            ),
            "prompt 'Commands run + results:' is not answered",
        ),
        (
            "Empty Checklist",
            valid_pr_body.replace(
                "- [x] Behavior changes have a meaningful regression.\n"
                "- [x] New source/test files live under `prosper/`.\n"
                "- [x] No Sony code/keys/firmware.\n"
                '- [x] No unconditional "owned" answers.\n'
                "- [x] `git diff --check` clean.\n",
                "",
            ),
            "Section '## Checklist' is empty",
        ),
    ]

    failures = 0

    for label, body, expected_fragment in must_fail:
        problems = check_pr_body(body)
        if not problems:
            print(f"selftest: FAILED to reject -- {label}", file=sys.stderr)
            failures += 1
        elif not any(expected_fragment.lower() in p.lower() for p in problems):
            print(
                f"selftest: rejected '{label}' but missing expected fragment '{expected_fragment}':\n    got: {problems}",
                file=sys.stderr,
            )
            failures += 1

    for label, body in must_pass:
        problems = check_pr_body(body)
        if problems:
            print(
                f"selftest: wrongly rejected -- {label}\n    problems: {problems}", file=sys.stderr
            )
            failures += 1

    total = len(must_fail) + len(must_pass)
    print(
        f"selftest: {total} arms ({len(must_fail)} must-fail, {len(must_pass)} must-pass), {failures} failed"
    )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--body", help="PR description text string")
    parser.add_argument("--body-file", help="Path to file containing PR description")
    parser.add_argument(
        "--event-json", help="Path to GitHub Actions event.json file ($GITHUB_EVENT_PATH)"
    )
    parser.add_argument(
        "--github",
        action="store_true",
        help="Format violations with GitHub Actions ::error:: annotations",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run selftest against hand-written compliant and violating fixtures",
    )
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    if args.event_json:
        problems = check_event_file(Path(args.event_json))
    elif args.body_file:
        path = Path(args.body_file)
        if not path.is_file():
            print(f"error: body file not found: {path}", file=sys.stderr)
            return 2
        problems = check_pr_body(path.read_text(encoding="utf-8", errors="replace"))
    elif args.body is not None:
        problems = check_pr_body(args.body)
    else:
        parser.error("One of --body, --body-file, --event-json, or --selftest is required.")

    if not problems:
        print("PR description structure: all required sections and verification fields verified.")
        return 0

    print(
        f"error: PR description does not conform to the template ({len(problems)} problem(s)):\n",
        file=sys.stderr,
    )
    for p in problems:
        if args.github:
            print(f"::error::{p}")
        print(f"  - {p}", file=sys.stderr)
    print(
        "\nPlease update the PR description to follow .github/pull_request_template.md.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
