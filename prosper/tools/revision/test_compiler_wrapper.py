#!/usr/bin/env python3
"""Regression test: compiler_dependencies.py sees through a compiler-cache wrapper (#4356, #4187).

A build configured with ccache as its compiler names `/usr/lib64/ccache/c++` (a symlink to ccache)
in compile_commands.json. The scanner used to refuse that as an unsupported driver, the embedded
source identity became "unknown", and every fragment compile case was then marked incomplete, so
`fragment_compile_case`, `fragment_compile_case_cli` and `spv_validate` failed on such a build
while passing in CI.

The arms build real wrapper layouts on disk (a fake `ccache` and a `c++` symlink to it, in front of
a directory holding the real compiler) and assert both directions: each supported spelling resolves
to the real compiler, and each layout with no real compiler behind the wrapper still fails closed.
The end-to-end arm runs `dependencies()` itself, so it fails if the unwrapping is not wired in.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import tempfile

SKIP = 77
failures = 0
checks = 0


def check(condition: bool, label: str) -> None:
    global failures, checks
    checks += 1
    if not condition:
        failures += 1
        print(f"[FAIL] {label}")


def refuses(call, label: str) -> None:
    try:
        call()
    except ValueError:
        check(True, label)
    else:
        check(False, label)


def main() -> int:
    if os.name == "nt":
        print("compiler wrapper: SKIP (POSIX symlink layouts only)")
        return SKIP
    # A real compiler, not a wrapper: a ccache build host may well put the masquerade first.
    real = None
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        for name in ("c++", "g++", "clang++"):
            candidate = pathlib.Path(entry or ".") / name
            if real is None and candidate.is_file() and os.access(candidate, os.X_OK):
                target = candidate.resolve(strict=True)
                if target.name not in ("ccache", "sccache"):
                    real = target
    if real is None:
        print("compiler wrapper: SKIP (no C++ compiler on PATH)")
        return SKIP

    module_path = pathlib.Path(__file__).with_name("compiler_dependencies.py")
    spec = importlib.util.spec_from_file_location("compiler_dependencies", module_path)
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as scratch:
        root = pathlib.Path(scratch)
        wrap, real_dir, empty = root / "wrap", root / "real", root / "empty"
        for d in (wrap, real_dir, empty):
            d.mkdir()
        ccache = wrap / "ccache"
        ccache.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        ccache.chmod(0o755)
        (wrap / "c++").symlink_to(ccache)
        (real_dir / "c++").symlink_to(real)
        search = os.pathsep.join([str(wrap), str(real_dir)])

        # Undecorated drivers pass through untouched.
        compiler, args = scanner.unwrap_compiler([str(real_dir / "c++"), "-c", "x.cpp"], search)
        check(compiler == real and args[0] == str(real_dir / "c++"), "a plain driver is unchanged")

        # Masquerade: the wrapper's own directory is skipped and the later `c++` is the compiler.
        compiler, args = scanner.unwrap_compiler([str(wrap / "c++"), "-O2", "-c", "x.cpp"], search)
        check(compiler == real, "masquerade resolves to the real compiler")
        check(args == [str(real), "-O2", "-c", "x.cpp"], "masquerade keeps every flag")

        # Explicit: `ccache c++ ...` drops the wrapper argument.
        compiler, args = scanner.unwrap_compiler(
            [str(ccache), "c++", "-O2", "-c", "x.cpp"], os.pathsep.join([str(real_dir)])
        )
        check(compiler == real, "explicit wrapper resolves to the real compiler")
        check(args == [str(real), "-O2", "-c", "x.cpp"], "explicit wrapper argument is dropped")

        # Fail closed where the wrapper has nothing real behind it.
        refuses(
            lambda: scanner.unwrap_compiler([str(wrap / "c++"), "-c", "x.cpp"], str(wrap)),
            "masquerade with only the wrapper on PATH refuses",
        )
        refuses(
            lambda: scanner.unwrap_compiler([str(wrap / "c++"), "-c", "x.cpp"], str(empty)),
            "masquerade with no compiler on PATH refuses",
        )
        refuses(
            lambda: scanner.unwrap_compiler([str(ccache), "-O2", "-c", "x.cpp"], search),
            "explicit wrapper followed by a flag refuses",
        )
        refuses(
            lambda: scanner.unwrap_compiler([str(ccache), "c++", "-c", "x.cpp"], str(wrap)),
            "explicit wrapper naming another wrapper refuses",
        )

        # End to end through the scanner's real dependency scan, with the masquerade on PATH.
        source = root / "unit.cpp"
        header = root / "unit.hpp"
        header.write_text("constexpr int unit_value = 3;\n", encoding="utf-8")
        source.write_text(
            '#include "unit.hpp"\nint unit() { return unit_value; }\n', encoding="utf-8"
        )
        command = {
            "directory": str(root),
            "arguments": [str(wrap / "c++"), "-c", str(source), "-o", str(root / "unit.o")],
        }
        saved = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join([search, saved])
        try:
            paths = scanner.dependencies(command)
        except ValueError as error:
            check(False, f"masqueraded command scans (refused: {error})")
        else:
            check(real in paths, "the real compiler is part of the identity")
            check(ccache.resolve() not in paths, "the wrapper is not")
            check(header.resolve() in paths, "the scan still sees the source's headers")
        finally:
            os.environ["PATH"] = saved

        # A work tree reached through a symlink still owns its sources: fingerprint() resolves
        # every source path, so it has to resolve the root it compares them against as well.
        tree = root / "tree"
        (tree / "prosper/src").mkdir(parents=True)
        unit = tree / "prosper/src/unit.cpp"
        unit.write_text("int unit() { return 4; }\n", encoding="utf-8")
        commands = tree / "compile_commands.json"
        commands.write_text(
            json.dumps(
                [
                    {
                        "directory": str(tree),
                        "file": str(unit),
                        "arguments": [str(real), "-c", str(unit)],
                    }
                ]
            ),
            encoding="utf-8",
        )
        alias = root / "alias"
        alias.symlink_to(tree)
        try:
            digest = scanner.fingerprint(alias / "compile_commands.json", alias)
        except ValueError as error:
            check(False, f"a symlinked work tree fingerprints (refused: {error})")
        else:
            check(
                digest == scanner.fingerprint(commands, tree),
                "a symlinked work tree has the same identity as its target",
            )

    print(f"compiler wrapper: {checks} checks, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
