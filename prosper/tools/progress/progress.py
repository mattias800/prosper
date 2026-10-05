"""Static HLE coverage census.

Scans prosper/src/hle/**/*.cpp for HLE(name) definitions and classifies each as
implemented or placeholder. Groups by hle/<area>/. Writes progress.json, prints a
text summary, or diffs two JSON files. No dump, no build, no GPU.

Done means a real implementation: the body does not call prosper_on_unimpl and the
handler is not placeholder-only. A handler implemented anywhere counts as done even
when another file placeholder-registers the same name, matching the runtime rule
that a later real register_fn overrides a placeholder.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFINITION = re.compile(r"\bHLE\((\w+)\)")
PLACEHOLDER_REG = re.compile(r"register_placeholder\s*\(\s*[^,]+,\s*\(HleFn\)\s*&?(\w+)")
STUB_MARK = "prosper_on_unimpl"


def body_end(text: str, start: int) -> int:
    """Index of the closing brace matching the opening brace at start."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(text)


def placeholder_handlers(text: str) -> set[str]:
    """Handler names installed via register_placeholder in one file."""
    return set(PLACEHOLDER_REG.findall(text))


def scan_area(path: Path) -> dict:
    """Classify every HLE(name) under one hle/<area>/ directory."""
    done: set[str] = set()
    todo: set[str] = set()
    ph_only: set[str] = set()
    real_body: set[str] = set()
    for source in sorted(path.rglob("*.cpp")):
        text = source.read_text(errors="ignore")
        ph_only |= placeholder_handlers(text)
        for match in DEFINITION.finditer(text):
            name = match.group(1)
            open_idx = text.find("{", match.end())
            if open_idx < 0:
                continue
            body = text[open_idx : body_end(text, open_idx)]
            if STUB_MARK in body:
                todo.add(name)
            else:
                real_body.add(name)
    done |= real_body
    # Placeholder-registered but never really implemented stays todo.
    todo |= ph_only - done
    # A real body anywhere beats a placeholder anywhere (runtime override rule).
    todo -= done
    return {
        "name": path.name,
        "done": len(done),
        "todo": len(todo),
        "done_names": sorted(done),
        "todo_names": sorted(todo),
    }


def collect(root: Path) -> dict:
    """Scan every hle/<area>/ under the prosper source root."""
    hle = root / "prosper" / "src" / "hle"
    groups = [scan_area(p) for p in sorted(hle.iterdir()) if p.is_dir()]
    done = sum(g["done"] for g in groups)
    total = done + sum(g["todo"] for g in groups)
    percent = round(100 * done / total, 2) if total else 0.0
    return {
        "done": done,
        "todo": total - done,
        "total": total,
        "percent": percent,
        "groups": groups,
    }


def text(data: dict) -> str:
    """One text line per area plus a total, for stdout and logs."""
    lines = [
        f"{g['name']}: {g['done']}/{g['done'] + g['todo']}"
        for g in sorted(data["groups"], key=lambda g: g["name"])
    ]
    lines.append(f"Total: {data['done']}/{data['total']} ({data['percent']}%)")
    return "\n".join(lines) + "\n"


def names(data: dict, state: str) -> set[tuple[str, str]]:
    """(area, handler) pairs in one state for diffing."""
    return {(g["name"], n) for g in data["groups"] for n in g.get(f"{state}_names", [])}


def compare(base: dict, head: dict) -> str:
    """Markdown delta between two progress.json files."""
    base_done, head_done = names(base, "done"), names(head, "done")
    base_all = base_done | names(base, "todo")
    head_all = head_done | names(head, "todo")
    implemented = head_done - base_done
    declared = head_all - base_all - head_done
    regressed = base_done & (head_all - head_done)
    removed = base_all - head_all
    delta = round(head["percent"] - base["percent"], 2)
    # ASCII verdict tag: Windows consoles decode stdout as cp1252, where the
    # obvious emoji/icons are unencodable and crash the diff (measured).
    icon = "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT"
    counts = ", ".join(
        f"{n:+} {label}"
        for n, label in (
            (len(implemented), "implemented"),
            (len(declared), "declared as stubs"),
            (-len(regressed), "reverted"),
            (-len(removed), "removed"),
        )
        if n
    )
    head_line = f"{icon} HLE: {head['percent']}% ({delta:+}%"
    head_line += f", {counts}" if counts else ""
    lines = [head_line + ")", ""]
    for title, items in (
        ("implemented", implemented),
        ("declared as stubs", declared),
        ("went back to stubs", regressed),
        ("removed", removed),
    ):
        if not items:
            continue
        lines.append(f"{title} ({len(items)}):")
        lines += [f"- {area}::{name}" for area, name in sorted(items)[:50]]
        if len(items) > 50:
            lines.append(f"- ... {len(items) - 50} more")
        lines.append("")
    if len(lines) == 2:
        lines.append("No HLE coverage change.")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Parse arguments and run the census, diff, or both."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("."), help="checkout root holding prosper/"
    )
    parser.add_argument("--output", type=Path, help="directory receiving progress.json")
    parser.add_argument(
        "--compare",
        type=Path,
        nargs=2,
        metavar=("BASE", "HEAD"),
        help="diff two progress.json files as markdown",
    )
    args = parser.parse_args()
    if args.compare:
        # Local dev tool: operator-supplied paths are the whole job (precedent:
        # new_probe/verify_report.py). NOSONAR marks the taint-accepted lines.
        base = json.loads(args.compare[0].read_text())  # NOSONAR
        head = json.loads(args.compare[1].read_text())  # NOSONAR
        print(compare(base, head), end="")
        return
    data = collect(args.root)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)  # NOSONAR
        (args.output / "progress.json").write_text(json.dumps(data, indent=2) + "\n")
    print(text(data), end="")


if __name__ == "__main__":
    main()
