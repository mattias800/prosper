#!/usr/bin/env python3
"""move_module.py -- relocate source modules into folders, mechanically.

A 116k-line restructure cannot be hand-edited, and hand-moving is how this project's recorded traps
happen: a whole-file `git checkout` that silently reverts another lane's edits, and "clean merge,
broken file", where two branches both merge cleanly into something wrong. So the moves run through
this, and the diff a reviewer reads is `git mv` plus include rewrites -- never retyped code.

What it does, for each module NAME assigned to folder F under a root R (default prosper/src/gpu):

  1. `git mv R/NAME.{cpp,hpp,h}` -> `R/F/NAME.{cpp,hpp,h}`, preserving history.
  2. Rewrites every `#include` of those headers, repo-wide, into ONE canonical form.
  3. Rewrites the paths in CMakeLists.txt.

The canonical include form is `gpu/F/NAME.hpp`, i.e. relative to `src`, which `prosper_core` puts on
the include path (`target_include_directories(prosper_core PUBLIC src)`). That form resolves
identically from src/gpu, frontends/, tools/ and tests/, so a file can move again later without
touching its consumers' spelling. Bare same-directory includes (`"NAME.hpp"`) stop working the moment
a file moves into a subfolder, which is exactly why they are normalised away rather than patched.

VERIFICATION IS NOT THIS TOOL'S JOB, and that is deliberate. It reports what it changed; the build and
the test suite decide whether it was right. A pure move has two independent checks available --
`verify_pure_move.py` for textual identity and ctest for behavioural identity -- and neither depends
on anyone reading a large diff carefully.

DRY-RUN COUNTS OVERSTATE THE PATH PASS, and the mechanism is worth knowing before you read one as
a preview. The passes run in order -- includes, then paths -- and a dry run applies neither, so the
path pass still sees include lines the include pass would have already rewritten. `#include
"../src/hle/dispatch/dispatch.hpp"` contains `src/hle/dispatch/dispatch.hpp` as a substring, so it is counted twice.
Measured on the src/hle move: 240 path citations predicted, 45 actual. The dry run is exact for the
moves and the includes; treat its path number as an upper bound.

  # dry run first, always
  python3 prosper/tools/refactor/move_module.py --plan prosper/tools/refactor/gpu_layout.txt --dry-run
  python3 prosper/tools/refactor/move_module.py --plan prosper/tools/refactor/gpu_layout.txt

A plan file is `folder: name name name`, one group per line, `#` comments ignored.
"""

import argparse
import pathlib
import re
import subprocess
import sys

SUFFIXES = (".cpp", ".hpp", ".h")


def read_exact(path: pathlib.Path) -> str:
    """Read without newline translation.

    `Path.read_text()` opens in universal-newline mode, which turns every CRLF into LF in the string
    it returns; writing that back produces LF. So merely LOADING a CRLF file to rewrite one include
    rewrites every line in it. Measured: docs/ROADMAP.md came back as 312 insertions and 312
    deletions for a one-line path citation, and the resulting diff is unreviewable -- which is
    exactly when a real change hides.
    """
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write_exact(path: pathlib.Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def repo_root() -> pathlib.Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True, check=True)
    return pathlib.Path(out.stdout.strip())


def read_plan(path: pathlib.Path) -> dict[str, str]:
    """folder -> [names] flattened to name -> folder, rejecting duplicates loudly."""
    assignment: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            sys.exit(f"{path}:{lineno}: expected 'folder: name name', got {raw!r}")
        folder, names = line.split(":", 1)
        for name in names.split():
            if name in assignment:
                sys.exit(f"{path}:{lineno}: {name} already assigned to {assignment[name]}")
            assignment[name] = folder.strip()
    return assignment


def tracked_text_files(root: pathlib.Path, keep: tuple[str, ...]) -> list[pathlib.Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
    return [root / p for p in out.stdout.decode().split("\0")
            if p and p.endswith(keep)]


SOURCE_SUFFIXES = (".cpp", ".hpp", ".h")
# The path pass reaches further than the include pass on purpose: the ledgers it must keep honest
# are .txt and .py, and the citations it must keep resolvable are .md.
CITING_SUFFIXES = SOURCE_SUFFIXES + (".txt", ".cmake", ".md", ".py", ".ps1", ".sh", ".json",
                                    ".yml", ".yaml")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, type=pathlib.Path)
    ap.add_argument("--source-root", default="prosper/src/gpu",
                    help="directory the modules currently live in")
    ap.add_argument("--include-prefix", default="gpu",
                    help="canonical include prefix, relative to the include path root")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--paths-only", action="store_true",
                    help="skip the git mv; rewrite includes and path citations for files already "
                         "moved. Exists because the path-citation pass was added after a move had "
                         "already landed, and re-running the whole tool would have found nothing "
                         "to move and therefore nothing to rewrite.")
    args = ap.parse_args()

    root = repo_root()
    src_root = root / args.source_root
    if not src_root.is_dir():
        sys.exit(f"no such directory: {src_root}")

    assignment = read_plan(args.plan)

    # Only move what exists; report the rest rather than silently doing nothing, because a typo'd
    # module name would otherwise look like a successful no-op.
    moves: list[tuple[pathlib.Path, pathlib.Path]] = []
    missing: list[str] = []
    for name, folder in sorted(assignment.items()):
        found = False
        for suffix in SUFFIXES:
            # Search the WHOLE subtree, not just its top level. A plan is re-run after it is edited
            # -- a test placed in the wrong folder has to move from where it IS, not from where it
            # started -- and a top-level-only lookup reports every already-moved module as missing,
            # which prints as a warning and then does nothing.
            here = src_root / f"{name}{suffix}"
            if not here.exists():
                matches = sorted(src_root.rglob(f"{name}{suffix}"))
                if len(matches) > 1:
                    sys.exit(f"{name}{suffix} exists at {len(matches)} paths under {src_root}: "
                             f"{', '.join(str(m) for m in matches)}; the plan is keyed by module "
                             f"name and cannot express that")
                if not matches:
                    continue
                here = matches[0]
            there = src_root / folder / f"{name}{suffix}"
            found = True
            if here.resolve() != there.resolve():
                moves.append((here, there))
            elif args.paths_only:
                moves.append((src_root / f"{name}{suffix}", there))
        if not found:
            missing.append(name)
    if missing:
        print(f"  WARNING {len(missing)} module(s) in the plan do not exist: {' '.join(missing)}")

    header_names = {p.name for p, _ in moves if p.suffix in (".hpp", ".h")}
    # An EMPTY prefix is meaningful, not a mistake: prosper/tests is itself the include root (it is
    # already one for 8 targets), so its canonical form is `fixtures/render_runner.h` with no
    # subsystem component. Joining unconditionally produced a leading slash and an absolute path.
    def canonical(name: str, folder: str) -> str:
        return "/".join(x for x in (args.include_prefix, folder, name) if x)

    include_target = {p.name: canonical(p.name, assignment[p.stem])
                      for p, _ in moves if p.suffix in (".hpp", ".h")}

    print(f"  {len(moves)} file(s) to move into {len(set(assignment.values()))} folder(s)")

    # --- the guard that makes the include rewrite safe ------------------------------------------
    # The include rewrite anchors on the header's BASENAME, which is what lets it collapse every
    # spelling a consumer happened to use -- `"x.hpp"`, `"../src/gpu/x.hpp"`, `"../../hle/x.hpp"` --
    # into one canonical form without knowing which. That is only sound while a basename names
    # exactly one header in the repository. If two directories both hold `audio.hpp`, the rewrite
    # would silently repoint the OTHER one's consumers at the moved file: a different translation
    # unit, no error, and it still compiles. Nothing downstream can see that -- not the build, not
    # the suite, not a reviewer reading a diff of 400 identical-looking include lines.
    #
    # So the precondition is checked rather than assumed. Measured on this repository: zero
    # duplicate header basenames, which is why the rewrite is safe here and why an abort is the
    # right response to that ceasing to be true.
    all_headers: dict[str, list[str]] = {}
    for path in tracked_text_files(root, (".hpp", ".h")):
        all_headers.setdefault(path.name, []).append(str(path.relative_to(root)))
    collisions = {n: v for n, v in all_headers.items()
                  if n in {p.name for p, _ in moves if p.suffix in (".hpp", ".h")} and len(v) > 1}
    if collisions:
        for name, paths in sorted(collisions.items()):
            print(f"  [FAIL] {name} names {len(paths)} headers: {', '.join(paths)}")
        sys.exit("a moved header's basename is not unique; the include rewrite would silently "
                 "repoint the other one's consumers. Rename one, or move them together.")

    # --- 1. the moves ---------------------------------------------------------------------------
    for old, new in moves:
        if args.dry_run or args.paths_only:
            print(f"    git mv {old.relative_to(root)} -> {new.relative_to(root)}")
            continue
        new.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "mv", str(old.relative_to(root)), str(new.relative_to(root))],
                       cwd=root, check=True)

    # --- 2. include rewrites --------------------------------------------------------------------
    # Match any existing spelling of the header -- bare, `gpu/X`, `../src/gpu/X`, deeper relatives --
    # and collapse them all to the canonical form. Anchoring on the basename is what makes this
    # independent of how many `../` a given consumer happened to use.
    include_re = re.compile(r'#include\s+"([^"]*?)([A-Za-z0-9_]+\.(?:hpp|h))"')
    rewritten_files = 0
    rewritten_lines = 0
    for path in tracked_text_files(root, SOURCE_SUFFIXES):
        try:
            text = read_exact(path)
        except UnicodeDecodeError:
            continue

        def replace(m: re.Match) -> str:
            nonlocal rewritten_lines
            prefix, base = m.group(1), m.group(2)
            if base not in header_names:
                return m.group(0)
            canonical = include_target[base]
            if prefix + base == canonical:
                return m.group(0)
            rewritten_lines += 1
            return f'#include "{canonical}"'

        new_text = include_re.sub(replace, text)
        if new_text != text:
            rewritten_files += 1
            if not args.dry_run:
                write_exact(path, new_text)
    print(f"  {rewritten_lines} include(s) rewritten across {rewritten_files} file(s)")

    # --- 2b. relative includes inside the moved files -------------------------------------------
    # Pass 2 fixes how the WORLD includes a moved header. This fixes how a moved file includes
    # everything else: `#include "../self/module.hpp"` resolved from src/hle, and resolves from
    # nowhere once the file sits in src/hle/dispatch. Every such include is broken by the move, and
    # each one costs a full build to discover -- the compiler reports the first and stops.
    #
    # Resolved, not pattern-matched. Each relative include is resolved against the file's OLD
    # directory; if the result is a real file under src, it is rewritten to its src-relative form,
    # which is depth-independent and so survives the next move too. A hardcoded list of subsystem
    # names would have been shorter and would silently skip anything not on the list -- and an
    # include the tool declines to fix is one the build reports as the tool's success.
    rel_re = re.compile(r'#include\s+"([^"<>]+)"')
    src_dir = root / "prosper" / "src"
    fixed_rel = 0
    unresolved: list[str] = []
    for old, new_path in moves:
        if new_path.suffix not in SOURCE_SUFFIXES or args.dry_run or not new_path.exists():
            continue
        text = read_exact(new_path)

        def fix(m: re.Match) -> str:
            nonlocal fixed_rel
            spelling = m.group(1)
            if not spelling.startswith((".", "/")) and "/" in spelling:
                return m.group(0)          # already a src-relative form
            target = (old.parent / spelling).resolve()
            if not target.exists():
                if spelling.startswith("."):
                    unresolved.append(f"{new_path.name}: {spelling}")
                return m.group(0)
            try:
                canonical = target.relative_to(src_dir.resolve())
            except ValueError:
                return m.group(0)          # outside src: leave it to the author
            if str(canonical) == spelling:
                return m.group(0)
            fixed_rel += 1
            return f'#include "{canonical}"'

        new_text = rel_re.sub(fix, text)
        if new_text != text:
            write_exact(new_path, new_text)
    print(f"  {fixed_rel} relative include(s) inside moved files re-anchored to src")
    if unresolved:
        print(f"  WARNING {len(unresolved)} relative include(s) could not be resolved and were left "
              f"alone: {', '.join(unresolved[:6])}")

    # --- 3. path citations ----------------------------------------------------------------------
    # A move invalidates far more than `#include` lines, and the remainder is where a move goes
    # quietly wrong rather than loudly. The repository carries PATH-KEYED LEDGERS whose keys are
    # `src/gpu/NAME.cpp` -- check_ascii_output.py's quarantine and tools/env/diag_gate_baseline.txt.
    # A stale key does not error. It reads as "that file has no findings", so the ledger silently
    # stops describing the tree it exists to describe, which is the precise failure both gates were
    # built to prevent. (Measured: this restructure orphaned 7 quarantine rows and 34 baseline rows.)
    # Docs and code comments that cite a moved file are the same defect with a slower fuse.
    #
    # `\b` after the extension is what separates `.h` from the `.h` inside `.hpp`: in
    # `gpu_execute.hpp` that `h` is followed by `p`, both word characters, so no boundary exists and
    # no match is made. Anchoring on the TAIL means a `prosper/`-prefixed citation is rewritten by
    # the same rule, and so is a CMake source list -- which is why this replaces the CMake-only pass
    # it grew out of rather than sitting beside it.
    path_rules = [(re.compile(re.escape(str(old.relative_to(root / "prosper"))) + r"\b"),
                   str(new.relative_to(root / "prosper")))
                  for old, new in moves]
    path_files = 0
    path_edits = 0
    for path in tracked_text_files(root, CITING_SUFFIXES):
        try:
            text = read_exact(path)
        except UnicodeDecodeError:
            continue
        new_text, edits = text, 0
        for rx, repl in path_rules:
            new_text, n = rx.subn(repl, new_text)
            edits += n
        if edits:
            path_files += 1
            path_edits += edits
            if not args.dry_run:
                write_exact(path, new_text)
    print(f"  {path_edits} path citation(s) rewritten across {path_files} file(s)")

    if args.dry_run:
        print("  (dry run -- nothing written)")
    else:
        print("  now BUILD and run ctest; this tool does not verify its own work")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
