#!/usr/bin/env python3
"""Regression test: compiler_dependencies.py sees through a compiler-cache wrapper (#4356, #4187).

A build configured with ccache as its compiler names `/usr/lib64/ccache/c++` (a symlink to ccache)
in compile_commands.json. The scanner used to refuse that as an unsupported driver, the embedded
source identity became "unknown", and every fragment compile case was then marked incomplete, so
`fragment_compile_case`, `fragment_compile_case_cli` and `spv_validate` failed on such a build
while passing in CI.

The tests build real wrapper layouts on disk (a fake `ccache` and a `c++` symlink to it, in front of
a directory holding the real compiler) and assert both directions: each supported spelling resolves
to the real compiler, and each layout with no real compiler behind the wrapper still fails closed.
The end-to-end test runs `dependencies()` itself, so it fails if the unwrapping is not wired in.

pytest collects the `test_*` functions. ctest runs this file directly, without pytest, through the
small runner at the bottom; a host with no POSIX symlinks or no C++ compiler skips (exit 77).
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from types import SimpleNamespace

SKIP = 77
MODULE_PATH = pathlib.Path(__file__).with_name("compiler_dependencies.py")


class _Skip(Exception):
    pass


# Set by the ctest runner below. The skip mechanism follows how the file was started, never whether
# pytest happens to import: pytest.skip raises a BaseException the runner must not have to catch.
_UNDER_RUNNER = False


def _skip(reason: str) -> None:
    if _UNDER_RUNNER:
        raise _Skip(reason)
    import pytest

    pytest.skip(reason)


def _scanner():
    spec = importlib.util.spec_from_file_location("compiler_dependencies", MODULE_PATH)
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    return scanner


def _real_compiler() -> pathlib.Path:
    """A real compiler, not a wrapper: a ccache build host may well put the masquerade first."""
    if os.name == "nt":
        _skip("POSIX symlink layouts only")
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        for name in ("c++", "g++", "clang++"):
            candidate = pathlib.Path(entry or ".") / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                target = candidate.resolve(strict=True)
                if target.name not in ("ccache", "sccache"):
                    return target
    _skip("no C++ compiler on PATH")
    raise AssertionError("unreachable")


@contextlib.contextmanager
def _layout():
    """wrap/ccache (answers `-k <key>` from FAKE_CCACHE_<KEY>), wrap/c++ -> ccache, real/c++."""
    real = _real_compiler()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as scratch:
        root = pathlib.Path(scratch)
        wrap, real_dir, empty = root / "wrap", root / "real", root / "empty"
        for d in (wrap, real_dir, empty):
            d.mkdir()
        ccache = wrap / "ccache"
        # Like ccache's own config query; anything else is a compile, which this fixture must never
        # be asked to do.
        ccache.write_text(
            '#!/bin/sh\nif [ "$1" = "-k" ]; then\n'
            '  case "$2" in\n'
            '    compiler) echo "$FAKE_CCACHE_COMPILER" ;;\n'
            '    path) echo "$FAKE_CCACHE_PATH" ;;\n'
            '    prefix_command) echo "$FAKE_CCACHE_PREFIX_COMMAND" ;;\n'
            "    *) exit 1 ;;\n"
            "  esac\n  exit 0\nfi\nexit 99\n",
            encoding="utf-8",
        )
        ccache.chmod(0o755)
        (wrap / "c++").symlink_to(ccache)
        (real_dir / "c++").symlink_to(real)
        yield SimpleNamespace(
            root=root,
            wrap=wrap,
            real_dir=real_dir,
            empty=empty,
            ccache=ccache,
            real=real,
            search=os.pathsep.join([str(wrap), str(real_dir)]),
            scanner=_scanner(),
        )


def _refuses(call) -> bool:
    try:
        call()
    except ValueError:
        return True
    return False


def test_a_plain_driver_is_unchanged():
    with _layout() as lay:
        driver = str(lay.real_dir / "c++")
        compiler, args = lay.scanner.unwrap_compiler([driver, "-c", "x.cpp"], lay.search)
        assert compiler == lay.real and args[0] == driver


def test_masquerade_resolves_to_the_real_compiler_and_keeps_every_flag():
    with _layout() as lay:
        compiler, args = lay.scanner.unwrap_compiler(
            [str(lay.wrap / "c++"), "-O2", "-c", "x.cpp"], lay.search
        )
        assert compiler == lay.real
        assert args == [str(lay.real), "-O2", "-c", "x.cpp"]


def test_explicit_wrapper_drops_its_argument():
    with _layout() as lay:
        compiler, args = lay.scanner.unwrap_compiler(
            [str(lay.ccache), "c++", "-O2", "-c", "x.cpp"], str(lay.real_dir)
        )
        assert compiler == lay.real
        assert args == [str(lay.real), "-O2", "-c", "x.cpp"]
        compiler, _ = lay.scanner.unwrap_compiler(
            [str(lay.ccache), str(lay.real_dir / "c++"), "-c"], ""
        )
        assert compiler == lay.real, "an absolute compiler path after the wrapper"


def test_a_wrapper_with_nothing_real_behind_it_refuses():
    with _layout() as lay:
        s = lay.scanner
        masquerade = [str(lay.wrap / "c++"), "-c", "x.cpp"]
        assert _refuses(lambda: s.unwrap_compiler(masquerade, str(lay.wrap))), "only the wrapper"
        assert _refuses(lambda: s.unwrap_compiler(masquerade, str(lay.empty))), "no compiler"
        assert _refuses(
            lambda: s.unwrap_compiler([str(lay.ccache), "-O2", "-c", "x.cpp"], lay.search)
        ), "explicit wrapper followed by a flag"
        assert _refuses(
            lambda: s.unwrap_compiler([str(lay.ccache), "c++", "-c", "x.cpp"], str(lay.wrap))
        ), "explicit wrapper that finds only itself"


def test_a_second_wrapper_behind_the_masquerade_refuses():
    with _layout() as lay:
        # A DIFFERENT wrapper later on PATH is the next link of a chain, not something to skip.
        other = lay.root / "other"
        other.mkdir()
        sccache = other / "sccache"
        sccache.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
        sccache.chmod(0o755)
        (other / "c++").symlink_to(sccache)
        chain = os.pathsep.join([str(lay.wrap), str(other), str(lay.real_dir)])
        assert _refuses(
            lambda: lay.scanner.unwrap_compiler([str(lay.wrap / "c++"), "-c", "x.cpp"], chain)
        )


def test_a_relative_or_empty_path_entry_ahead_of_the_compiler_refuses():
    # It would resolve against the compile's directory, not this process's.
    with _layout() as lay:
        for entry in ("bin", ""):
            search = os.pathsep.join([str(lay.wrap), entry, str(lay.real_dir)])
            assert _refuses(
                lambda search=search: lay.scanner.unwrap_compiler(
                    [str(lay.wrap / "c++"), "-c", "x.cpp"], search
                )
            ), repr(entry)


def test_ccache_overrides_refuse():
    # ccache's own settings replace the lookup; with any of them set the identity refuses.
    with _layout() as lay:
        for key in ("COMPILER", "PATH", "PREFIX_COMMAND"):
            os.environ[f"FAKE_CCACHE_{key}"] = "clang++" if key != "PATH" else str(lay.real_dir)
            lay.scanner.wrapper_overrides_unset.cache_clear()
            try:
                assert _refuses(
                    lambda: lay.scanner.unwrap_compiler(
                        [str(lay.wrap / "c++"), "-c", "x.cpp"], lay.search
                    )
                ), key
            finally:
                del os.environ[f"FAKE_CCACHE_{key}"]
        lay.scanner.wrapper_overrides_unset.cache_clear()


def test_the_dependency_scan_sees_through_the_masquerade():
    with _layout() as lay:
        source = lay.root / "unit.cpp"
        header = lay.root / "unit.hpp"
        header.write_text("constexpr int unit_value = 3;\n", encoding="utf-8")
        source.write_text('#include "unit.hpp"\nint unit() { return unit_value; }\n', "utf-8")
        command = {
            "directory": str(lay.root),
            "arguments": [str(lay.wrap / "c++"), "-c", str(source), "-o", str(lay.root / "u.o")],
        }
        saved = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join([lay.search, saved])
        try:
            paths = lay.scanner.dependencies(command)
        finally:
            os.environ["PATH"] = saved
        assert lay.real in paths, "the real compiler is part of the identity"
        assert lay.ccache.resolve() not in paths, "the wrapper is not"
        assert header.resolve() in paths, "the scan still sees the source's headers"


def test_a_symlinked_work_tree_has_the_same_identity():
    # fingerprint() resolves every source path, so it must resolve the root it compares against.
    with _layout() as lay:
        tree = lay.root / "tree"
        (tree / "prosper/src").mkdir(parents=True)
        unit = tree / "prosper/src/unit.cpp"
        unit.write_text("int unit() { return 4; }\n", encoding="utf-8")
        commands = tree / "compile_commands.json"
        entry = {
            "directory": str(tree),
            "file": str(unit),
            "arguments": [str(lay.real), "-c", str(unit)],
        }
        commands.write_text(json.dumps([entry]), encoding="utf-8")
        alias = lay.root / "alias"
        alias.symlink_to(tree)
        assert lay.scanner.fingerprint(alias / "compile_commands.json", alias) == (
            lay.scanner.fingerprint(commands, tree)
        )


def test_the_cli_reports_a_path_free_reason():
    # The build log's only diagnostic: a missing commands file is reported by exception TYPE, never
    # with the path its OSError carries.
    with _layout() as lay:
        missing = lay.root / "private-name" / "compile_commands.json"
        p = subprocess.run(
            [sys.executable, str(MODULE_PATH), str(missing), str(lay.root)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert p.returncode == 2
        assert p.stderr.strip() == "compiler dependency identity unavailable: FileNotFoundError"


def _main() -> int:
    """Run every test without pytest (ctest has none); 77 when the host cannot run them."""
    global _UNDER_RUNNER
    _UNDER_RUNNER = True
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = skipped = 0
    for name, fn in tests:
        try:
            fn()
        except _Skip as skip:
            skipped += 1
            print(f"[SKIP] {name}: {skip}")
        except Exception as error:  # an unexpected error is a failure, not an abort of the run
            failures += 1
            print(f"[FAIL] {name}: {type(error).__name__}: {error}")
    print(f"compiler wrapper: {len(tests)} tests, {failures} failures, {skipped} skipped")
    if failures:
        return 1
    return SKIP if skipped == len(tests) else 0


if __name__ == "__main__":
    sys.exit(_main())
