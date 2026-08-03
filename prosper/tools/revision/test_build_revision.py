#!/usr/bin/env python3
"""Regression for build-time revision refreshes across source-identical commits."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip()


def write_project(source: Path, module: Path) -> None:
    source.mkdir(parents=True)
    module_path = module.as_posix().replace('"', '\\"')
    (source / "CMakeLists.txt").write_text(
        f"""cmake_minimum_required(VERSION 3.20)
project(build_revision_regression LANGUAGES CXX)
include(\"{module_path}\")
prosper_add_build_revision_library(test_build_revision WORK_TREE \"${{CMAKE_CURRENT_SOURCE_DIR}}\")
add_executable(revision_query main.cpp)
target_link_libraries(revision_query PRIVATE test_build_revision)
set_target_properties(revision_query PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY \"${{CMAKE_CURRENT_BINARY_DIR}}/out\"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG \"${{CMAKE_CURRENT_BINARY_DIR}}/out\"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE \"${{CMAKE_CURRENT_BINARY_DIR}}/out\"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO \"${{CMAKE_CURRENT_BINARY_DIR}}/out\"
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL \"${{CMAKE_CURRENT_BINARY_DIR}}/out\")
""",
        encoding="utf-8",
    )
    (source / "main.cpp").write_text(
        '#include "build_revision.hpp"\n#include <cstdlib>\n#include <iostream>\n'
        "int main() {\n"
        '  const char* override_revision = std::getenv("PROSPER_CAPTURE_REVISION");\n'
        "  std::cout << (override_revision ? override_revision : prosper::embedded_build_revision());\n"
        "}\n",
        encoding="utf-8",
    )


def configure(cmake: str, source: Path, build: Path, generator: str | None) -> None:
    command = [cmake, "-S", str(source), "-B", str(build)]
    if generator:
        command.extend(["-G", generator])
    run(command, cwd=source)


def build(cmake: str, build_dir: Path) -> None:
    run(
        [cmake, "--build", str(build_dir), "--config", "Release", "--target", "revision_query"],
        cwd=build_dir,
    )


def executable_path(build_dir: Path) -> Path:
    return build_dir / "out" / ("revision_query.exe" if os.name == "nt" else "revision_query")


def query(build_dir: Path, *, override: str | None = None) -> str:
    executable = executable_path(build_dir)
    env = os.environ.copy()
    if override is None:
        env.pop("PROSPER_CAPTURE_REVISION", None)
    else:
        env["PROSPER_CAPTURE_REVISION"] = override
    return run([str(executable)], cwd=build_dir, env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--git", required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--generator")
    args = parser.parse_args()

    scratch = args.scratch_root.resolve()
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    source = scratch / "repo"
    build_dir = scratch / "build"
    write_project(source, args.module.resolve())
    run([args.git, "init", "--quiet"], cwd=source)
    run([args.git, "config", "user.name", "Prosper Build Test"], cwd=source)
    run([args.git, "config", "user.email", "build-test@invalid"], cwd=source)
    run([args.git, "add", "."], cwd=source)
    run([args.git, "commit", "--quiet", "-m", "tree"], cwd=source)
    revision_a = run([args.git, "rev-parse", "HEAD"], cwd=source)

    configure(args.cmake, source, build_dir, args.generator)
    build(args.cmake, build_dir)
    if query(build_dir) != revision_a:
        raise AssertionError("initial binary does not contain the current source revision")

    generated_source = build_dir / "generated" / "test_build_revision" / "build_revision.cpp"
    generated_mtime = generated_source.stat().st_mtime_ns
    executable_mtime = executable_path(build_dir).stat().st_mtime_ns
    time.sleep(1.1)  # Make an accidental rewrite observable even on coarse timestamp filesystems.
    build(args.cmake, build_dir)
    if generated_source.stat().st_mtime_ns != generated_mtime:
        raise AssertionError("unchanged revision rewrote the generated source")
    if executable_path(build_dir).stat().st_mtime_ns != executable_mtime:
        raise AssertionError("unchanged revision relinked the consumer")

    run([args.git, "commit", "--quiet", "--allow-empty", "-m", "same tree, new revision"], cwd=source)
    revision_b = run([args.git, "rev-parse", "HEAD"], cwd=source)
    tree_a = run([args.git, "show", "-s", "--format=%T", revision_a], cwd=source)
    tree_b = run([args.git, "show", "-s", "--format=%T", revision_b], cwd=source)
    if revision_a == revision_b or tree_a != tree_b:
        raise AssertionError("test setup did not create two revisions with an identical source tree")

    # This is the decisive arm: no configure and no clean between source-identical commits.
    build(args.cmake, build_dir)
    observed_b = query(build_dir)
    if observed_b != revision_b:
        raise AssertionError(
            f"incremental build kept stale revision: expected {revision_b}, got {observed_b}"
        )

    override = "explicit-capture-revision"
    if query(build_dir, override=override) != override:
        raise AssertionError("PROSPER_CAPTURE_REVISION did not remain higher priority")

    # Linked worktrees store a .git indirection file rather than a directory. They must resolve the
    # worktree's own HEAD just like an ordinary clone.
    linked_source = scratch / "linked-worktree"
    linked_build = scratch / "linked-build"
    run([args.git, "worktree", "add", "--quiet", "--detach", str(linked_source), revision_b], cwd=source)
    configure(args.cmake, linked_source, linked_build, args.generator)
    build(args.cmake, linked_build)
    if query(linked_build) != revision_b:
        raise AssertionError("linked-worktree build did not embed its own HEAD")

    # The directory is deliberately nested below the outer checkout. The helper must not let Git
    # walk up into that unrelated repository; exported source trees retain the unknown fallback.
    nongit_source = scratch / "nongit"
    nongit_build = scratch / "nongit-build"
    write_project(nongit_source, args.module.resolve())
    configure(args.cmake, nongit_source, nongit_build, args.generator)
    build(args.cmake, nongit_build)
    if query(nongit_build) != "unknown":
        raise AssertionError("non-Git build did not use the unknown revision fallback")

    print(f"build revision refreshed {revision_a[:12]} -> {revision_b[:12]} with identical trees")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
