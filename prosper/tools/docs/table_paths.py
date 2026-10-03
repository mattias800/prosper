"""Resolve the orchestration table's known rename without inventing a missing baseline."""

from __future__ import annotations

import subprocess
import sys

CURRENT_FILE = "prosper/docs/process/GAME_COMPAT_ORCHESTRATION.md"
LEGACY_FILE = "prosper/docs/GAME_COMPAT_ORCHESTRATION.md"


def table_paths(path: str) -> tuple[str, ...]:
    """Aliases apply only to this logical table; custom paths retain exact-path behavior."""
    if path in (CURRENT_FILE, LEGACY_FILE):
        return CURRENT_FILE, LEGACY_FILE
    return (path,)


def select_table_path(path: str, existing: list[str]) -> str:
    """Require one authenticated path, refusing both missing and ambiguous copies."""
    if not isinstance(existing, list) or any(not isinstance(p, str) for p in existing):
        raise ValueError("table path listing is not a list of strings")
    found = [candidate for candidate in table_paths(path) if candidate in existing]
    if len(found) != 1:
        raise ValueError(f"expected exactly one table path for {path}, found {found}")
    return found[0]


def git_table_path(ref: str, path: str, run) -> str:
    """Resolve aliases against this revision's tree, without reading a different revision."""
    paths = run(["git", "ls-tree", "-r", "--name-only", "-z", ref, "--", *table_paths(path)])
    if paths and not paths.endswith("\0"):
        raise ValueError("table path listing is not NUL-terminated")
    return select_table_path(path, paths.split("\0")[:-1])


def main() -> int:
    """Print the actual path for a workflow's baseline revision, or fail closed."""
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 3:
        print("usage: table_paths.py REF PATH", file=sys.stderr)
        return 2

    def run(cmd: list[str]) -> str:
        return subprocess.run(cmd, check=True, capture_output=True, encoding="utf-8").stdout

    try:
        print(git_table_path(sys.argv[1], sys.argv[2], run))
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
