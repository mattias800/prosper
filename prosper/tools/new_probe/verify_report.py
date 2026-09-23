#!/usr/bin/env python3
"""Native positive control: discriminate caller, TID, byte count and ELF mapping."""

import argparse
import os
import re
import subprocess
from pathlib import Path


def input_file(path: Path, root: Path) -> Path:
    """Restrict report inputs to the caller's scratch tree, including symlinks."""
    # Refuse a lexical escape before probing the filesystem, then resolve symlinks
    # and refuse any resolved escape. This CLI intentionally inspects operator files.
    candidate = Path(os.path.abspath(path))
    if not candidate.is_relative_to(root):
        raise ValueError(f"probe input is outside the report root: {path}")
    # Resolve a symlink without probing whether its target exists. Containment
    # must be decided before the file-type check, including for dangling links.
    resolved = candidate.resolve(strict=False)  # NOSONAR
    if not resolved.is_relative_to(root):
        raise ValueError(f"probe input is outside the report root: {path}")
    if not resolved.is_file():  # NOSONAR
        raise ValueError(f"probe input is not a file: {path}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control_output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--root", type=Path, required=True,
                        help="scratch directory containing both probe input files")
    parser.add_argument("--elf", choices=("EXEC", "DYN"), required=True)
    parser.add_argument("--space-path", action="store_true")
    parser.add_argument("--tab-path", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)  # NOSONAR
    if not root.is_dir():
        parser.error("--root must be a directory")
    control_file = input_file(args.control_output, root)
    report_file = input_file(args.report, root)

    control = control_file.read_text(encoding="ascii").splitlines()[0]
    match = re.fullmatch(r"main_tid=(\d+) worker_tid=(\d+) calls_per_site_per_tid=(\d+)", control)
    assert match, control
    main_tid, worker_tid, expected = map(int, match.groups())
    assert main_tid != worker_tid

    lines = report_file.read_text(encoding="utf-8").splitlines()
    header = lines[0]
    assert re.search(r"\boverflow_calls=0\b", header), header
    assert re.search(r"\bresolution_failures=0\b", header), header
    assert re.search(r"\bfailed_calls=1\b", header), header
    assert lines[1] == (
        "# tid\tkind\tcaller\tcalls\trequested_bytes\tdso\tdso_offset\tsymbol"
    ), lines[1]

    rows = {}
    for line in lines[2:]:
        fields = line.split("\t")
        assert len(fields) == 8, (len(fields), line)
        tid, kind, caller, calls, size, dso, offset, symbol = fields
        if symbol in {"new_probe_scalar_site", "new_probe_array_site"}:
            key = (int(tid), symbol)
            assert key not in rows, key
            rows[key] = (kind, int(calls), int(size), dso, offset, caller)

    assert set(rows) == {
        (main_tid, "new_probe_scalar_site"), (main_tid, "new_probe_array_site"),
        (worker_tid, "new_probe_scalar_site"), (worker_tid, "new_probe_array_site"),
    }, rows
    for (_, symbol), (kind, calls, size, dso, offset, caller) in rows.items():
        unit = 37 if symbol == "new_probe_scalar_site" else 59
        assert kind == ("scalar" if unit == 37 else "array")
        assert calls == expected, (symbol, calls, expected)
        assert size == expected * unit, (symbol, size)
        assert (" " in dso) == args.space_path, dso
        assert ("\\t" in dso) == args.tab_path, dso
        real_dso = dso.replace("\\t", "\t")
        # Report text is data, not an executable selector. Its resolved ELF must live in
        # the explicit scratch root, and `--` prevents option-looking paths becoming flags.
        elf_file = input_file(Path(real_dso), root)
        elf = subprocess.check_output(["readelf", "-hW", "--", str(elf_file)], text=True)  # NOSONAR
        assert re.search(rf"Type:\s+{args.elf}\b", elf), (args.elf, dso)
        mapped = int(caller, 16) - 1 if args.elf == "EXEC" else int(offset, 16) - 1
        assert mapped > 0
        # Same confined ELF path and fixed argument vector; no shell is involved.
        resolved = subprocess.check_output(  # NOSONAR
            ["addr2line", "-e", str(elf_file), "-f", "-C", hex(mapped)], text=True
        ).splitlines()[0]
        assert resolved == symbol, (caller, offset, resolved, symbol)

    print(f"PASS: {args.elf} caller mapping, two sites x two TIDs, exact calls/bytes")


if __name__ == "__main__":
    main()
