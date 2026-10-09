#!/usr/bin/env python3
"""coverage_report — console oracle coverage and candidate inventory.

Answers: which library exports exist in the verified NID database, are not yet covered
by any committed console oracle cases file, and are plausible candidates for a human
with a console to measure next?

Offline and read-only: inspects database, committed cases files, and Prosper HLE registration
tables, and never modifies any data, test, or source files.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Classifiers for uncovered exports
CLASS_STATE = "excluded-state"
CLASS_IDENTITY = "excluded-identity"
CLASS_NETWORK = "excluded-network"
CLASS_FP_VARIADIC = "excluded-fp-variadic"
CLASS_CANDIDATE = "candidate"

# Configurable pattern lists kept in a single table at the top of the file
# so a reviewer can inspect or change them easily.
# Patterns are applied in order of specificity.
STATE_PATTERNS = [
    "Save",
    "Setting",
    "Reboot",
    "Install",
    "Format",
    "Delete",
    "Remove",
    "Write",
]

IDENTITY_PATTERNS = [
    "Account",
    "UserId",
    "Serial",
    "Mac",
    "Uuid",
    "DeviceId",
]

NETWORK_PATTERNS = [
    "Net",
    "Http",
    "Ssl",
    "Socket",
    "Dns",
    "Curl",
    "Psn",
    "Np",
]

# Specific function names or substrings that match sceKernelDebugRaiseException
EXCEPTION_PATTERNS = [
    "DebugRaiseException",
]

# Patterns for floating-point and variadic functions:
# - printf and scanf families (printf, scanf, vprintf, vfprintf, snprintf, sscanf, etc.)
# - names ending in 'f' that are printf-like (e.g. asprintf, dprintf) or math-float functions
PRINTF_SCANF_RE = re.compile(r"(?:^|_)v?(?:f|s|sn|d|as|w)?(?:printf|scanf)", re.IGNORECASE)
PRINTF_LIKE_ENDINGS = ("printf", "scanf", "dprintf", "asprintf", "snprintf")

# When the library is known, state modification exclusion applies to storage and system libraries.
# If library is unknown ("?"), any state keyword is treated as state-modifying to be conservative.
STORAGE_SYSTEM_LIBS = {
    "libkernel.sprx",
    "libkernel",
    "libSceSaveData.sprx",
    "libSceSaveData",
    "libSceAppContent.sprx",
    "libSceAppContent",
    "libSceSystemService.sprx",
    "libSceSystemService",
    "libSceFs.sprx",
    "libSceFs",
    "?",
}


def classify_export(name: str, lib: str = "?") -> str:
    """Classify an uncovered export by name pattern and optional library attribution.

    Classes:
      excluded-state: names containing Save, Setting, Reboot, Install, Format, Delete, Remove, Write
                      when the library is a storage/system library (or '?').
      excluded-identity: names containing Account, UserId, Serial, Mac, Uuid, DeviceId.
      excluded-network: names containing Net, Http, Ssl, Socket, Dns, Curl, Psn, Np.
      excluded-fp-variadic: names ending in 'f' that are printf-like, or printf/scanf families.
      candidate: everything else.
    """
    # Exclude fatal exception raisers (key safety rule)
    for pat in EXCEPTION_PATTERNS:
        if pat in name:
            return CLASS_STATE

    # 1. State modification patterns (storage / system libs or unknown lib)
    is_storage_or_system = (lib in STORAGE_SYSTEM_LIBS) or any(
        s in lib.lower() for s in ("kernel", "save", "system", "appcontent", "fs")
    )
    if is_storage_or_system:
        for pat in STATE_PATTERNS:
            if pat in name:
                return CLASS_STATE

    # 2. Identity patterns
    for pat in IDENTITY_PATTERNS:
        if pat in name:
            return CLASS_IDENTITY

    # 3. Network patterns
    for pat in NETWORK_PATTERNS:
        if pat in name:
            return CLASS_NETWORK

    # 4. FP / variadic patterns
    if PRINTF_SCANF_RE.search(name):
        return CLASS_FP_VARIADIC
    if any(name.endswith(end) for end in PRINTF_LIKE_ENDINGS):
        return CLASS_FP_VARIADIC
    # Names ending in 'f' that are printf-like
    if name.endswith("f") and ("print" in name.lower() or "scan" in name.lower()):
        return CLASS_FP_VARIADIC

    return CLASS_CANDIDATE


def parse_nid_db(path: pathlib.Path) -> Tuple[List[Tuple[str, str]], List[Tuple[int, int, str]]]:
    """Parse NID database (whitespace-separated: NID <name>).

    Returns (valid_entries, bad_lines) where valid_entries is [(nid, name), ...]
    and bad_lines is [(line_no, part_count, line_str), ...].
    """
    entries: List[Tuple[str, str]] = []
    bad_lines: List[Tuple[int, int, str]] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) == 2:
                entries.append((parts[0], parts[1]))
            else:
                bad_lines.append((idx, len(parts), s))

    return entries, bad_lines


def load_cases_coverage(data_dir: pathlib.Path) -> Dict[str, str]:
    """Load all committed cases files and map function name -> library name.

    Returns {func_name: lib_name}.
    """
    # Import run_oracle for parsing cases
    tools_dir = pathlib.Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import run_oracle as ro

    cases_funcs: Dict[str, str] = {}
    cases_files = sorted(data_dir.glob("*.cases.tsv"))
    for cf in cases_files:
        try:
            content = cf.read_text(encoding="utf-8")
            cases = ro.parse_cases(content)
            for c in cases:
                if c.func not in cases_funcs:
                    cases_funcs[c.func] = c.lib
        except Exception as e:
            sys.stderr.write(f"Warning: could not parse cases file {cf.name}: {e}\n")

    return cases_funcs


def load_registered_handlers(prosper_root: pathlib.Path) -> Tuple[Set[str], int]:
    """Extract registered Sony names using prosper/tools/re/hle_handler_map.py.

    Returns (set_of_registered_names, unresolved_site_count).
    """
    re_dir = prosper_root / "prosper" / "tools" / "re"
    if str(re_dir) not in sys.path:
        sys.path.insert(0, str(re_dir))
    import hle_handler_map as hmap

    hle_src = prosper_root / "prosper" / "src" / "hle"
    sc = hmap.scan_tree(str(hle_src), "windows")

    reg_names: Set[str] = set()
    for r in sc.regs:
        if r.name:
            reg_names.add(r.name)

    return reg_names, len(sc.unresolved)


def load_nid_libs(path: pathlib.Path) -> Dict[str, str]:
    """Load optional NID -> library mapping TSV."""
    nid_to_lib: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                nid_to_lib[parts[0].strip()] = parts[1].strip()
    return nid_to_lib


def generate_coverage(
    db_entries: List[Tuple[str, str]],
    cases_funcs: Dict[str, str],
    reg_names: Set[str],
    nid_libs: Optional[Dict[str, str]] = None,
    only_lib: Optional[str] = None,
    registered_only: bool = False,
) -> Dict[str, Any]:
    """Compute coverage statistics and classify uncovered exports."""
    nid_libs = nid_libs or {}

    # Unique exports from DB
    seen_db_funcs: Set[str] = set()
    per_lib_data: Dict[str, Dict[str, Any]] = {}

    def get_lib_bucket(lib_name: str) -> Dict[str, Any]:
        if lib_name not in per_lib_data:
            per_lib_data[lib_name] = {
                "db_exports": 0,
                "cases_covered": 0,
                "registered": 0,
                "neither": 0,
                "classes": {
                    CLASS_STATE: 0,
                    CLASS_IDENTITY: 0,
                    CLASS_NETWORK: 0,
                    CLASS_FP_VARIADIC: 0,
                    CLASS_CANDIDATE: 0,
                },
                "candidates": [],
            }
        return per_lib_data[lib_name]

    for nid, func in db_entries:
        if func in seen_db_funcs:
            continue
        seen_db_funcs.add(func)

        # Determine library:
        # 1. From cases if already covered
        # 2. From nid_libs if provided
        # 3. Otherwise "?"
        if func in cases_funcs:
            lib = cases_funcs[func]
        elif nid in nid_libs:
            lib = nid_libs[nid]
        else:
            lib = "?"

        if only_lib and lib != only_lib:
            continue

        in_cases = func in cases_funcs
        in_prosper = func in reg_names
        in_neither = (not in_cases) and (not in_prosper)

        bucket = get_lib_bucket(lib)
        bucket["db_exports"] += 1
        if in_cases:
            bucket["cases_covered"] += 1
        if in_prosper:
            bucket["registered"] += 1
        if in_neither:
            bucket["neither"] += 1

        # Check if candidate analysis applies
        if not in_cases:
            classification = classify_export(func, lib)
            bucket["classes"][classification] += 1

            is_candidate = classification == CLASS_CANDIDATE
            if registered_only and not in_prosper:
                is_candidate = False

            if is_candidate:
                bucket["candidates"].append(func)

    return per_lib_data


def format_text_report(
    per_lib_data: Dict[str, Any],
    unresolved_count: int,
    db_line_count: int,
    bad_line_count: int,
    only_lib: Optional[str] = None,
    all_exports: bool = False,
) -> str:
    """Format human-readable coverage report."""
    lines: List[str] = []
    lines.append("Console Oracle Coverage Report")
    lines.append("==============================")
    lines.append(f"NID database rows: {db_line_count} (anomalies/non-2-part lines: {bad_line_count})")
    lines.append(f"Unresolved Prosper HLE registration sites: {unresolved_count}")
    if only_lib:
        lines.append(f"Filter library: {only_lib}")
    if not all_exports:
        lines.append("Candidates: registered handlers only (default; pass --all-candidates to show all)")
    else:
        lines.append("Candidates: all uncovered exports (--all-candidates)")
    lines.append("")

    header = (
        f"{'Library':<28} "
        f"{'Exports':>8} {'Cases':>8} {'RegHle':>8} {'Neither':>8} "
        f"{'State':>7} {'Ident':>7} {'Net':>7} {'FP/Var':>7} {'Cand':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    # Sort libraries: known libraries alphabetically, '?' at the end
    sorted_libs = sorted(
        per_lib_data.keys(),
        key=lambda l: (1 if l == "?" else 0, l),
    )

    total_exports = 0
    total_cases = 0
    total_registered = 0
    total_neither = 0
    total_classes = {c: 0 for c in (CLASS_STATE, CLASS_IDENTITY, CLASS_NETWORK, CLASS_FP_VARIADIC, CLASS_CANDIDATE)}
    total_candidates_eligible = 0

    for lib in sorted_libs:
        b = per_lib_data[lib]
        c = b["classes"]
        cand_count = len(b["candidates"])

        total_exports += b["db_exports"]
        total_cases += b["cases_covered"]
        total_registered += b["registered"]
        total_neither += b["neither"]
        for k in total_classes:
            total_classes[k] += c[k]
        total_candidates_eligible += cand_count

        lines.append(
            f"{lib:<28} "
            f"{b['db_exports']:>8} {b['cases_covered']:>8} {b['registered']:>8} {b['neither']:>8} "
            f"{c[CLASS_STATE]:>7} {c[CLASS_IDENTITY]:>7} {c[CLASS_NETWORK]:>7} {c[CLASS_FP_VARIADIC]:>7} "
            f"{cand_count:>7}"
        )

    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<28} "
        f"{total_exports:>8} {total_cases:>8} {total_registered:>8} {total_neither:>8} "
        f"{total_classes[CLASS_STATE]:>7} {total_classes[CLASS_IDENTITY]:>7} {total_classes[CLASS_NETWORK]:>7} "
        f"{total_classes[CLASS_FP_VARIADIC]:>7} {total_candidates_eligible:>7}"
    )
    lines.append("")

    # First 20 candidates per library
    lines.append("Candidate Exports (up to 20 per library):")
    lines.append("-----------------------------------------")
    for lib in sorted_libs:
        cands = per_lib_data[lib]["candidates"]
        if not cands:
            continue
        lines.append(f"[{lib}] ({len(cands)} total candidates):")
        for sample in cands[:20]:
            lines.append(f"  - {sample}")
        if len(cands) > 20:
            lines.append(f"  ... and {len(cands) - 20} more")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Console oracle coverage and candidate export analyzer."
    )
    parser.add_argument(
        "--db",
        type=pathlib.Path,
        default=pathlib.Path(os.environ["PROSPER_NID_DB"]) if "PROSPER_NID_DB" in os.environ else None,
        help="Path to verified NID CSV database <NID_DB> (or set via PROSPER_NID_DB env var).",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=None,
        help="Path to console_oracle data directory (default: prosper/tests/data/console_oracle).",
    )
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=None,
        help="Path to repository root.",
    )
    parser.add_argument(
        "--nid-libs",
        type=pathlib.Path,
        default=None,
        help="Optional TSV mapping NID to library name (NID<TAB>library).",
    )
    parser.add_argument(
        "--only",
        dest="only_lib",
        metavar="LIB",
        default=None,
        help="Limit report to a single library.",
    )
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="Include all uncovered candidate exports (by default, candidates are limited to registered HLE handlers).",
    )
    parser.add_argument(
        "--registered-only",
        action="store_true",
        default=True,
        help="Limit candidate listing and candidate count to functions Prosper already registers (default behaviour).",
    )
    parser.add_argument(
        "--json",
        dest="json_file",
        type=pathlib.Path,
        default=None,
        help="Write machine-readable JSON report to file.",
    )

    args = parser.parse_args()

    if not args.db:
        sys.stderr.write("Error: --db <NID_DB> is required (or set PROSPER_NID_DB environment variable).\n")
        return 1

    repo_root = args.root or pathlib.Path(__file__).resolve().parents[3]
    data_dir = args.data_dir or (repo_root / "prosper" / "tests" / "data" / "console_oracle")

    if not args.db.is_file():
        sys.stderr.write(f"Error: NID database file not found: {args.db}\n")
        return 1

    entries, bad_lines = parse_nid_db(args.db)
    cases_funcs = load_cases_coverage(data_dir)
    reg_names, unresolved_count = load_registered_handlers(repo_root)

    nid_libs = None
    if args.nid_libs and args.nid_libs.is_file():
        nid_libs = load_nid_libs(args.nid_libs)

    registered_only = not args.all_candidates

    per_lib_data = generate_coverage(
        db_entries=entries,
        cases_funcs=cases_funcs,
        reg_names=reg_names,
        nid_libs=nid_libs,
        only_lib=args.only_lib,
        registered_only=registered_only,
    )

    report_text = format_text_report(
        per_lib_data=per_lib_data,
        unresolved_count=unresolved_count,
        db_line_count=len(entries) + len(bad_lines),
        bad_line_count=len(bad_lines),
        only_lib=args.only_lib,
        all_exports=args.all_candidates,
    )

    print(report_text)

    if args.json_file:
        # Machine-readable output (no host paths inside)
        json_obj = {
            "summary": {
                "db_total_lines": len(entries) + len(bad_lines),
                "db_valid_entries": len(entries),
                "db_bad_lines": len(bad_lines),
                "unresolved_registration_sites": unresolved_count,
                "filter_only_lib": args.only_lib,
                "filter_registered_only": registered_only,
            },

            "libraries": {
                lib: {
                    "exports": b["db_exports"],
                    "cases_covered": b["cases_covered"],
                    "registered": b["registered"],
                    "neither": b["neither"],
                    "classes": b["classes"],
                    "candidate_count": len(b["candidates"]),
                    "candidates": b["candidates"][:20],
                }
                for lib, b in per_lib_data.items()
            },
        }
        args.json_file.write_text(json.dumps(json_obj, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
