#!/usr/bin/env python3
"""Gate C++ formatting and clang-tidy on the lines a change touches, never on the whole tree.

The tree predates both tools. Measured 2026-10-02: the root `.clang-format` would rewrite ~30% of
the 258,100 lines under prosper/src and prosper/frontends, and the root `.clang-tidy` candidate set
reported 7,196 findings. Failing on either would block every PR or force a mass rewrite. So only
the lines a change ADDS OR MODIFIES are checked, against the merge base with `--base`:

  * clang-format: `--dry-run -Werror --lines=` over each changed hunk. A violation prints the
    replacement the formatter wants and fails. The command that applies it is
    `git clang-format <merge-base>`, which touches changed lines only.
  * clang-tidy: each changed C++ file is analysed with `--line-filter` naming the changed lines of
    EVERY changed file, so a finding in a changed header line is reported from a changed TU too.
    A file that is neither in compile_commands.json nor a header is reported as SKIPPED, by name.
    Error-level findings (WarningsAsErrors in `.clang-tidy`) fail; warnings are printed only.

The working tree is compared, not just HEAD, so uncommitted edits are checked locally.

EXIT STATUS IS THE VERDICT: 0 clean, 1 violations, 2 could not evaluate (a missing tool, a bad
ref, a missing compile database). `--selftest` builds a throwaway git repository, commits a clean
file, then adds one violation per enabled check family and asserts the gate fails on each and
passes on a clean edit, so a gate that quietly stopped discriminating fails there.

Run from anywhere inside the checkout:
    uv sync --group cpp-lint
    uv run --no-sync python prosper/tools/ci/check_cpp_lint.py --build prosper/build-linux
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CPP_SUFFIXES = (".cpp", ".cc", ".cxx", ".hpp", ".h", ".hh", ".inl")
HEADER_SUFFIXES = (".hpp", ".h", ".hh", ".inl")
EXCLUDED_PREFIXES = ("third_party/", "prosper/third_party/")
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
FINDING = re.compile(r"^(.*?):(\d+):(\d+): (warning|error): (.*) \[([\w.,-]+)\]$")
SAFE_REF = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._/@~^-]*")


def is_cpp(path):
    """True for a C++ source this gate owns."""
    return path.endswith(CPP_SUFFIXES) and not path.startswith(EXCLUDED_PREFIXES)


def parse_diff(text):
    """Map each new-side path in a `git diff -U0` to its list of (first, last) changed lines.

    Pure deletions (`+N,0`) add no lines and are dropped: there is nothing on the new side to check.
    """
    ranges, path = {}, None
    for line in text.splitlines():
        if line.startswith("+++ "):
            target = line[4:]
            path = target[2:] if target.startswith("b/") else None
            continue
        m = HUNK.match(line)
        if m and path is not None:
            first, count = int(m.group(1)), int(m.group(2) or "1")
            if count:
                ranges.setdefault(path, []).append((first, first + count - 1))
    return {p: r for p, r in ranges.items() if is_cpp(p)}


def parse_findings(output):
    """(path, line, level, check, message) for every clang-tidy diagnostic line in `output`."""
    found = []
    for line in output.splitlines():
        m = FINDING.match(line.rstrip())
        if m:
            found.append((m.group(1), int(m.group(2)), m.group(4), m.group(6), m.group(5)))
    return found


def validate_ref(ref):
    """Reject anything git could parse as an option rather than a revision."""
    if not SAFE_REF.fullmatch(ref):
        raise ValueError(f"refusing base ref {ref!r}: must be a plain revision name")
    return ref


def git(root, *args):
    """Run git in `root`; return stdout, raising on failure."""
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
        encoding="utf-8",
        errors="replace",
    ).stdout


def tool(name):
    """Locate a pinned tool: the active environment's scripts dir first, then PATH."""
    here = Path(sys.executable).parent
    for cand in (here / name, here / f"{name}.exe"):
        if cand.is_file():
            return str(cand)
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} not found: run `uv sync --group cpp-lint`")
    return found


def check_format(root, ranges, clang_format):
    """clang-format violations on the changed lines, one entry per file."""
    problems = []
    for path, spans in sorted(ranges.items()):
        args = [clang_format, "--dry-run", "-Werror", "--style=file"]
        args += [f"--lines={a}:{b}" for a, b in spans]
        proc = subprocess.run(
            args + [path],
            capture_output=True,
            text=True,
            cwd=root,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode:
            problems.append(f"{path}: clang-format\n{proc.stderr.rstrip()}")
    return problems


def compile_db_files(build):
    """Absolute, normalised paths of every TU in `build`/compile_commands.json."""
    db = Path(build) / "compile_commands.json"
    if not db.is_file():
        raise RuntimeError(f"{db} not found: configure the build first")
    entries = json.loads(db.read_text(encoding="utf-8"))
    return {os.path.normcase(str(Path(e["directory"], e["file"]).resolve())) for e in entries}


def check_tidy(root, ranges, build, clang_tidy, extra_args, jobs):
    """clang-tidy over each changed file; returns (problems, warnings, skipped)."""
    in_db = compile_db_files(build)
    line_filter = json.dumps(
        [
            {"name": str((Path(root) / p).resolve()), "lines": [list(s) for s in spans]}
            for p, spans in ranges.items()
        ]
    )
    targets, skipped = [], []
    for path in sorted(ranges):
        absolute = os.path.normcase(str((Path(root) / path).resolve()))
        if absolute in in_db or path.endswith(HEADER_SUFFIXES):
            targets.append(path)
        else:
            skipped.append(f"{path}: not in {build}/compile_commands.json (not built here)")

    def one(path):
        args = [clang_tidy, "-p", str(build), "--quiet", f"--line-filter={line_filter}"]
        args += [f"--extra-arg={a}" for a in extra_args]
        proc = subprocess.run(
            args + [path],
            capture_output=True,
            text=True,
            cwd=root,
            encoding="utf-8",
            errors="replace",
        )
        return path, proc.stdout + proc.stderr

    errors, warnings, seen = [], [], set()
    with ThreadPoolExecutor(max(1, jobs)) as ex:
        for path, out in ex.map(one, targets):
            findings = parse_findings(out)
            if any(check == "clang-diagnostic-error" for _, _, _, check, _ in findings):
                skipped.append(
                    f"{path}: does not compile under clang-tidy here (see output)\n{out}"
                )
                continue
            for where, line, level, check, message in findings:
                key = (os.path.normcase(where), line, check)
                if key in seen:
                    continue
                seen.add(key)
                text = f"{where}:{line}: {message} [{check}]"
                (errors if level == "error" else warnings).append(text)
    return errors, warnings, skipped


def check_db(root, build, extra_args):
    """Prove clang-tidy can compile a real prosper/src TU from this build's database.

    The tidy gate SKIPS a file clang cannot compile rather than failing the PR, so a database whose
    flags clang rejects would turn the whole gate into a list of skips. This is the positive control
    for that: it must find a prosper/src TU and analyse it with no clang-diagnostic-error.
    """
    src = os.path.normcase(str((Path(root) / "prosper" / "src").resolve()))
    candidates = sorted(f for f in compile_db_files(build) if f.startswith(src))
    if not candidates:
        raise RuntimeError(f"no prosper/src translation unit in {build}/compile_commands.json")
    args = [tool("clang-tidy"), "-p", str(build), "--quiet", "--checks=-*,modernize-use-nullptr"]
    args += [f"--extra-arg={a}" for a in extra_args]
    proc = subprocess.run(
        args + [candidates[0]],
        capture_output=True,
        text=True,
        cwd=root,
        encoding="utf-8",
        errors="replace",
    )
    out = proc.stdout + proc.stderr
    if any(c == "clang-diagnostic-error" for _, _, _, c, _ in parse_findings(out)):
        print(f"compile database NOT analysable: {candidates[0]}\n{out}")
        return 1
    print(
        f"compile database analysable ({len(candidates)} prosper/src TUs; probed {candidates[0]})"
    )
    return 0


def run(root, base, build, extra_args, jobs, do_format=True, do_tidy=True):
    """Evaluate the change in `root`; return (problems, warnings, skipped)."""
    merge_base = git(root, "merge-base", validate_ref(base), "HEAD").strip()
    ranges = parse_diff(git(root, "diff", "-U0", "--no-color", "--no-ext-diff", merge_base, "--"))
    print(
        f"checking {sum(len(s) for s in ranges.values())} changed hunk(s) "
        f"in {len(ranges)} C++ file(s) against {merge_base[:12]}"
    )
    problems, warnings, skipped = [], [], []
    if not ranges:
        return problems, warnings, skipped
    if do_format:
        problems += check_format(root, ranges, tool("clang-format"))
    if do_tidy:
        e, w, s = check_tidy(root, ranges, build, tool("clang-tidy"), extra_args, jobs)
        problems += e
        warnings += w
        skipped += s
    return problems, warnings, skipped


CLEAN = """#include <cstdio>

struct Base {
    virtual ~Base() = default;
    virtual int value() const;
};

int report(int x) {
    if (x > 0) return x;
    std::printf("%d\\n", x);
    return 0;
}
"""

# One violation per enabled family. Each is appended to the clean file in its own commit-less edit,
# so the gate must fire on exactly the added lines.
VIOLATIONS = {
    "clang-format": "int   badly_spaced( int x ){return x;}\n",
    "bugprone": "long long widen(int a, int b) { return a * b; }\n",
    "performance": "#include <string>\nstd::string cat(const std::string& s) {\n"
    "    std::string r;\n    for (int i = 0; i < 3; ++i) r = r + s + s;\n"
    "    return r;\n}\n",
    "concurrency-mt-unsafe": '#include <cstdlib>\nconst char* home() { return std::getenv("HOME"); }\n',
    "readability-function-size": "int huge() {\n    int v = 0;\n"
    + "    v += 1;\n" * 410
    + "    return v;\n}\n",
    "modernize-use-override": "struct Derived : Base {\n    virtual int value() const;\n};\n",
    "modernize-use-nullptr": "int* none() { return 0; }\n",
}


def selftest(build_root=None, extra_args=()):
    """Drive the real gate over a throwaway repo; each violation must fail, a clean edit must pass."""
    repo_root = Path(__file__).resolve().parents[3]
    failures = []

    def expect(label, got, want):
        if bool(got) != want:
            failures.append(f"{label}: got {got!r}, wanted violation={want}")

    expect(
        "diff parse", parse_diff("+++ b/a.cpp\n@@ -1,0 +2,3 @@\n+x\n") == {"a.cpp": [(2, 4)]}, True
    )
    expect("pure deletion", parse_diff("+++ b/a.cpp\n@@ -3,2 +2,0 @@\n"), False)
    expect("non-C++ dropped", parse_diff("+++ b/a.py\n@@ -1 +1 @@\n"), False)
    expect(
        "third_party dropped", parse_diff("+++ b/prosper/third_party/x.cpp\n@@ -1 +1 @@\n"), False
    )
    expect("finding parse", parse_findings("a.cpp:3:4: error: msg [bugprone-x]"), True)

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        for cfg in (".clang-format", ".clang-tidy"):
            shutil.copy(repo_root / cfg, repo / cfg)
        src = repo / "prosper" / "src" / "lint_probe.cpp"
        src.parent.mkdir(parents=True)
        src.write_text(CLEAN, encoding="utf-8", newline="\n")
        build = repo / "build"
        build.mkdir()
        (build / "compile_commands.json").write_text(
            json.dumps(
                [
                    {
                        "directory": str(repo),
                        "file": str(src),
                        "arguments": ["clang++", "-std=c++20", "-c", str(src)],
                    }
                ]
            ),
            encoding="utf-8",
        )
        git(repo, "init", "-q", "-b", "main")
        git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
        git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "clean")

        def gate(appended, **kw):
            src.write_text(CLEAN + appended, encoding="utf-8", newline="\n")
            problems, _, skipped = run(repo, "main", build, list(extra_args), 4, **kw)
            if skipped:
                failures.append(f"probe was skipped, not analysed: {skipped}")
            return problems

        expect("untouched tree", gate(""), False)
        expect("clean addition", gate("int twice(int x) {\n    return 2 * x;\n}\n"), False)
        for family, text in VIOLATIONS.items():
            is_format = family == "clang-format"
            # Tidy probes are checked with format off, so a formatting slip in a probe cannot be
            # what makes it fail; the format probe with tidy off, for the same reason.
            problems = gate(text, do_format=is_format, do_tidy=not is_format)
            named = [p for p in problems if is_format or family in p]
            expect(f"violation: {family}", named, True)

        # The property the whole gate exists for: violations already in the base, on lines the
        # change does not touch, never fail it -- neither formatting nor tidy.
        legacy = VIOLATIONS["concurrency-mt-unsafe"] + VIOLATIONS["clang-format"]
        src.write_text(CLEAN + legacy, encoding="utf-8", newline="\n")
        git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-am", "legacy")
        src.write_text(
            CLEAN + legacy + "int twice(int x) {\n    return 2 * x;\n}\n",
            encoding="utf-8",
            newline="\n",
        )
        problems, _, _ = run(repo, "main", build, list(extra_args), 4)
        expect("legacy violations on untouched lines", problems, False)
    if failures:
        print("SELFTEST FAILED\n" + "\n".join(failures))
        return 1
    print("selftest ok")
    return 0


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="origin/main")
    ap.add_argument(
        "--build",
        default="prosper/build-linux",
        help="build directory holding compile_commands.json",
    )
    ap.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="passed to clang-tidy, e.g. --target=x86_64-w64-mingw32 for MinGW builds",
    )
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--no-tidy", action="store_true", help="format only (no build needed)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument(
        "--check-db",
        action="store_true",
        help="only prove clang-tidy can compile a real TU from --build",
    )
    args = ap.parse_args()
    try:
        if args.selftest:
            return selftest(extra_args=args.extra_arg)
        root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
        build = Path(args.build)
        build = build if build.is_absolute() else root / build
        if args.check_db:
            return check_db(root, build, args.extra_arg)
        problems, warnings, skipped = run(
            root, args.base, build, args.extra_arg, args.jobs, do_tidy=not args.no_tidy
        )
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"COULD NOT EVALUATE: {exc}", file=sys.stderr)
        return 2
    for s in skipped:
        print(f"SKIPPED {s}")
    for w in warnings:
        print(f"warning (not gating) {w}")
    if problems:
        print("\n".join(problems))
        print(
            f"{len(problems)} violation(s) on changed lines. Fix, or suppress one finding with "
            "`// NOLINT(check-name): reason`. Formatting: `git clang-format <merge-base>`."
        )
        return 1
    print("clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
