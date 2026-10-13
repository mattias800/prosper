#!/usr/bin/env python3
"""Read a prosper-app --report bundle for triage without reproducing the run.

The bundle format is versioned (see frontends/prosper-app/report_bundle.hpp); this reader
accepts versions it knows and refuses anything else loudly, never by guessing. Later schema
versions add sections (perf-alarm summary, unimplemented NIDs, log tail, fault backtraces);
absence of a section the reader knows is reported, not defaulted.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

KNOWN_SCHEMA = 1


class BundleError(Exception):
    """A bundle that cannot be triaged (as opposed to one with an empty section)."""


def load_bundle(path: str | Path) -> dict:
    """Parse and validate the bundle. Raises BundleError on any refusal."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise BundleError(f"{path}: cannot be read ({error.strerror})") from error
    except UnicodeError as error:
        raise BundleError(f"{path}: not readable UTF-8") from error
    try:
        record = json.loads(text)
    except ValueError as error:
        raise BundleError(f"{path}: not valid JSON ({error})") from error
    if not isinstance(record, dict):
        raise BundleError(f"{path}: bundle root must be an object")
    if record.get("type") != "report":
        raise BundleError(f"{path}: not a report bundle (type={record.get('type')!r})")
    schema = record.get("schema")
    if type(schema) is not int or schema != KNOWN_SCHEMA:
        raise BundleError(f"{path}: schema {schema!r} is not readable by this tool")
    for key in ("tool", "build_revision", "build_source_identity", "os", "title_id", "switches"):
        if key not in record:
            raise BundleError(f"{path}: bundle lacks {key!r}")
    for key in ("tool", "build_revision", "build_source_identity"):
        if not isinstance(record[key], str) or not record[key]:
            raise BundleError(f"{path}: {key} must be a nonempty string")
    if record["os"] not in ("linux", "windows", "macos"):
        raise BundleError(f"{path}: os must name a supported host")
    title = record["title_id"]
    if not isinstance(title, str) or (title and not re.fullmatch(r"PPSA[0-9]{5}", title)):
        raise BundleError(f"{path}: title_id must be empty or a title identifier")
    if not isinstance(record["switches"], dict):
        raise BundleError(f"{path}: switches must be an object")
    for name, value in record["switches"].items():
        if not re.fullmatch(r"PROSPER_[A-Z0-9_]+", name) or not isinstance(value, str):
            raise BundleError(f"{path}: switches must map PROSPER_* names to strings")
    return record


def triage_lines(record: dict) -> list[str]:
    """One human line per bundle section a triager reads first."""
    switches = record["switches"]
    return [
        f"tool: {record['tool']}",
        f"build: {record['build_revision'][:12]} (source identity present: "
        f"{bool(record['build_source_identity'])})",
        f"os: {record['os']}",
        f"title: {record['title_id'] or '(no title identity)'}",
        f"switches: {len(switches)} PROSPER_* in force",
    ]


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: read_bundle.py <report.json>", file=sys.stderr)
        return 2
    try:
        record = load_bundle(args[0])
    except BundleError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    for line in triage_lines(record):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
