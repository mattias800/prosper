#!/usr/bin/env python3
"""dump_hygiene — report (and optionally remove) files in a game dump that prosper must never load.

Some dumps in circulation ship third-party replacements of Sony libraries: `dlc_emu` builds
libSceAppContent / libSceNpEntitlementAccess / libSceGameUpdate, and `ampr_emu` builds libSceAmpr.
They exist because on real hardware you cannot reimplement the platform — the only lever is
substituting a library. prosper *is* the platform layer, so those substitutions are redundant, and
worse than redundant: linking one would shadow prosper's own implementation and answer the guest
with somebody else's stub, so every entitlement or asset result measured afterwards would be theirs
rather than ours.

`src/host/image/module_path_policy.hpp` already makes the loader refuse them. This tool is the
other half: it finds them, says what each one is, and can take them off the disk entirely, so the
question does not depend on a runtime guard staying correct forever.

Four categories, and the distinction between the last two is the whole reason this is a tool rather
than an `rm`:

  REFUSED   a module prosper will never link (`fakelib/`, or any module outside the permitted
            directories). Removing it changes nothing prosper does. `--strip` removes these.
  MARKER    release-group provenance — an `.nfo`, a group directory. Read by nothing. `--strip`
            removes these.
  CONSUMED  prosper genuinely reads this. `dlc_emu.ini` is the declared add-content inventory that
            `src/hle/service/hle_addcontent.cpp` parses. NEVER removed, at any flag combination.
  RETAINED  third-party data prosper does not currently read, but which describes where the dump's
            real content lives — `ampr_emu.index` maps the APR file-id API onto a repacked tree.
            Reported so you know it is there; never removed, because deleting a description of
            where assets are is not reversible by re-running anything.

usage:
    dump_hygiene.py <dump>...              # report only
    dump_hygiene.py --strip --yes <dump>   # remove REFUSED + MARKER
    dump_hygiene.py --check <dump>...      # exit nonzero if any REFUSED remain (onboarding gate)
    dump_hygiene.py --json <dump>...       # machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Must stay equal to kPermittedModuleDirs in src/host/image/module_path_policy.hpp. A drift here
# would make the tool report a module as REFUSED that the loader in fact links, or vice versa — so
# tests/tools/test_dump_hygiene.py reads the C++ header and asserts the two agree, rather than
# trusting this comment.
PERMITTED_MODULE_DIRS = ("Media/Modules", "Media/Plugins", "sce_module")

# Only real module extensions. `.bin` was here briefly and was a disaster: it is the most generic
# data extension there is, and a sweep of the corpus reported 1,012 `Media/StreamingAssets/**/*.bin`
# and 914 `COMMON/ui/uv/*.bin` game-data files as refused modules. A tool whose output is thousands
# of false rows is a tool nobody reads.
MODULE_SUFFIXES = (".prx", ".sprx")

# The signal this tool exists to find: a module that IMPERSONATES a Sony library. A title's own
# native plugins are game content and are none of our business, even in directories prosper does not
# link from -- `prx/akaudioinput.prx` is Wwise, shipped by the developer. What matters is a `libSce*`
# somewhere it could shadow prosper's own implementation of that same library.
SONY_LIBRARY_PREFIX = "libsce"

# Sony's own metadata directory. Authentic dump content, present in every dump including ones with
# no third-party additions at all. prosper does not LINK from here -- `sce_sys/about/right.sprx` is
# not a module the loader wants -- but "prosper does not link it" and "it does not belong in the
# dump" are different claims, and only the second is this tool's business. Caught by dry-running
# against a known-clean dump: without this exclusion every one of the 50 dumps reported a REFUSED
# row for right.sprx, and --strip would have deleted an authentic Sony file from all of them.
SONY_DATA_DIRS = ("sce_sys",)

# Files prosper parses. Never removed.
CONSUMED = {
    "dlc_emu.ini": "declared add-content inventory, parsed by hle_addcontent.cpp",
}

# Not read by prosper, and still never removed -- for two quite different reasons.
RETAINED_FILES = {
    "ampr_emu.index": "maps the APR file-id API onto a repacked asset tree; nothing else "
                      "describes that layout",
}
RETAINED_DIRS = {
    "_original_files": "holds the dump's ORIGINAL Sony files, including an encrypted eboot.bin "
                       "(entropy 8.000). This is the most provenance-valuable thing in the dump "
                       "and the only local evidence of what the untouched file looked like",
}

# Directory names that carry no content, only provenance.
MARKER_DIRS = ("_duplex_", "_unlimited_")
MARKER_SUFFIXES = (".nfo", ".diz", ".sfv")

REFUSED, MARKER, CONSUMED_K, RETAINED_K = "REFUSED", "MARKER", "CONSUMED", "RETAINED"


def classify(rel: str, is_dir: bool) -> tuple[str, str] | None:
    """Classify one dump-relative path. Returns (category, why) or None if unremarkable.

    Pure: takes a relative path string, touches no filesystem. `rel` uses forward slashes.
    """
    parts = [p for p in rel.replace("\\", "/").split("/") if p]
    if not parts:
        return None
    name = parts[-1]
    lname = name.lower()
    top = parts[0].lower()
    directory = "/".join(parts[:-1])

    if top in SONY_DATA_DIRS:
        return None

    if is_dir:
        if lname in RETAINED_DIRS:
            return (RETAINED_K, RETAINED_DIRS[lname])
        if lname in MARKER_DIRS:
            return (MARKER, "release-group directory; read by nothing")
        return None

    if name in CONSUMED:
        return (CONSUMED_K, CONSUMED[name])
    if name in RETAINED_FILES:
        return (RETAINED_K, RETAINED_FILES[name])

    if top in MARKER_DIRS or lname.endswith(MARKER_SUFFIXES):
        return (MARKER, "release-group provenance; read by nothing")

    if not lname.endswith(MODULE_SUFFIXES):
        return None

    if top == "fakelib":
        return (REFUSED, "third-party replacement of a Sony library; prosper implements these "
                         "APIs itself and must not delegate them to a bundled module")

    # A Sony-named module anywhere but sce_module/ is the thing worth finding: it can only either
    # shadow prosper's own implementation or sit there pretending to be a platform library.
    if lname.startswith(SONY_LIBRARY_PREFIX) and directory.lower() != "sce_module":
        where = f"{directory}/" if directory else "the dump root"
        return (REFUSED, f"a Sony-named library in {where}; prosper implements the Sony libraries "
                         f"itself and links them only from sce_module/")

    return None


def scan(dump: Path) -> list[dict]:
    findings = []
    for dirpath, dirnames, filenames in os.walk(dump, followlinks=False):
        rel_dir = os.path.relpath(dirpath, dump)
        rel_dir = "" if rel_dir == "." else rel_dir.replace(os.sep, "/")

        # Classify directories, and do not descend into one we are going to report wholesale —
        # otherwise a marker directory's contents each produce their own redundant row.
        for d in list(dirnames):
            rel = f"{rel_dir}/{d}" if rel_dir else d
            got = classify(rel, is_dir=True)
            if got:
                cat, why = got
                findings.append({"path": rel, "kind": "dir", "category": cat, "why": why,
                                 "bytes": dir_size(Path(dirpath) / d)})
                dirnames.remove(d)

        for f in filenames:
            rel = f"{rel_dir}/{f}" if rel_dir else f
            got = classify(rel, is_dir=False)
            if got:
                cat, why = got
                p = Path(dirpath) / f
                findings.append({"path": rel, "kind": "file", "category": cat, "why": why,
                                 "bytes": p.stat().st_size if p.is_file() else 0})
    findings.sort(key=lambda x: (x["category"], x["path"]))
    return findings


def dir_size(p: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(p, followlinks=False):
        for f in filenames:
            fp = Path(dirpath) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GiB"


def looks_like_a_dump(p: Path) -> bool:
    # Guard for --strip: refuse to delete inside anything that is not recognisably a game dump.
    return (p / "sce_sys").is_dir() and (
        (p / "sce_sys" / "param.json").is_file() or (p / "sce_sys" / "param.sfo").is_file()
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+", type=Path)
    ap.add_argument("--strip", action="store_true",
                    help="remove REFUSED and MARKER entries (never CONSUMED or RETAINED)")
    ap.add_argument("--yes", action="store_true", help="do not prompt (required with --strip when "
                                                       "stdin is not a terminal)")
    ap.add_argument("--check", action="store_true",
                    help="exit nonzero if any REFUSED entry is present")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    report, any_refused, exit_code = {}, False, 0

    for dump in args.dumps:
        if not dump.is_dir():
            print(f"not a directory: {dump}", file=sys.stderr)
            exit_code = 2
            continue
        findings = scan(dump)
        report[str(dump)] = findings
        if any(f["category"] == REFUSED for f in findings):
            any_refused = True

        if args.json:
            continue

        print(f"\n=== {dump.name} ===")
        if not findings:
            print("  clean — nothing prosper refuses, no provenance markers")
            continue
        for cat in (REFUSED, MARKER, CONSUMED_K, RETAINED_K):
            rows = [f for f in findings if f["category"] == cat]
            if not rows:
                continue
            verb = {REFUSED: "prosper will not link",
                    MARKER: "read by nothing",
                    CONSUMED_K: "prosper reads this — kept",
                    RETAINED_K: "not read by prosper, but kept"}[cat]
            print(f"  {cat} ({verb}):")
            for r in rows:
                print(f"    {r['path']}{'/' if r['kind'] == 'dir' else ''}  [{human(r['bytes'])}]")
                print(f"        {r['why']}")

        if args.strip:
            removable = [f for f in findings if f["category"] in (REFUSED, MARKER)]
            if not removable:
                print("  nothing to strip")
                continue
            if not looks_like_a_dump(dump):
                print(f"  REFUSING to strip: {dump} has no sce_sys/param.* — this does not look "
                      f"like a game dump", file=sys.stderr)
                exit_code = 2
                continue
            if not args.yes:
                if not sys.stdin.isatty():
                    print("  --strip needs --yes when stdin is not a terminal", file=sys.stderr)
                    exit_code = 2
                    continue
                if input(f"  remove {len(removable)} entries from {dump.name}? [y/N] ").lower() != "y":
                    print("  skipped")
                    continue
            for r in removable:
                target = dump / r["path"]
                try:
                    if r["kind"] == "dir":
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                    print(f"  removed {r['path']}")
                except OSError as e:
                    print(f"  FAILED to remove {r['path']}: {e}", file=sys.stderr)
                    exit_code = 2
            # Removing every file out of `fakelib/` leaves the directory behind, which still reads
            # as "this dump carries a fakelib" to anyone who looks. Prune directories that the strip
            # emptied -- only those, and only if they really are empty, so a rmdir can never take
            # content with it.
            for r in removable:
                parent = (dump / r["path"]).parent
                while parent != dump and parent.is_dir():
                    try:
                        next(parent.iterdir())
                        break                      # not empty: stop, and leave it alone
                    except StopIteration:
                        parent.rmdir()
                        print(f"  removed emptied directory {parent.relative_to(dump)}")
                        parent = parent.parent
                    except OSError:
                        break

    if args.json:
        print(json.dumps(report, indent=2))

    if args.check and any_refused:
        return 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
