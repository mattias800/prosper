#!/usr/bin/env python3
"""Small real build proves immutable source/command/dependency identity, not HEAD alone."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from test_build_revision import build as build_target, configure, executable_path, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--generator")
    args = parser.parse_args()
    module = args.module.resolve(strict=True)
    repository = module.parent.parent
    scratch = Path(os.environ.get("PROSPER_TEST_SCRATCH_DIR") or Path.cwd() / "test-scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    checks = 0

    def check(ok, name):
        nonlocal checks
        checks += 1
        if not ok:
            raise AssertionError(name)

    # Compact child/target names also keep the supported Windows worktree path below MAX_PATH.
    with tempfile.TemporaryDirectory(prefix="i-", dir=scratch) as temporary:
        root = Path(temporary)
        source = root / "prosper"
        (source / "src").mkdir(parents=True)
        (source / "tools/revision").mkdir(parents=True)
        shutil.copyfile(repository / "tools/revision/compiler_dependencies.py",
                        source / "tools/revision/compiler_dependencies.py")
        build_dir = root / "b"
        project = f'''cmake_minimum_required(VERSION 3.20)
project(compiler_source_identity LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include("{module.as_posix()}")
prosper_add_build_revision_library(r WORK_TREE "${{CMAKE_CURRENT_SOURCE_DIR}}/..")
file(GLOB units CONFIGURE_DEPENDS "src/*.cpp")
add_executable(q ${{units}})
target_link_libraries(q PRIVATE r)
set_target_properties(q PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_CURRENT_BINARY_DIR}}/out"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${{CMAKE_CURRENT_BINARY_DIR}}/out")
'''
        (source / "CMakeLists.txt").write_text(project, encoding="utf-8")
        dependency = source / "src/dep.hpp"
        dependency.write_text("constexpr int dependency_value = 1;\n", encoding="utf-8")
        external = root / "external/header.hpp"
        external.parent.mkdir()
        external.write_text("constexpr int external_dependency = 1;\n", encoding="utf-8")
        program = source / "src/main.cpp"
        program.write_text('#include "build_revision.hpp"\n#include "dep.hpp"\n'
                           '#include "../../external/header.hpp"\n#include <iostream>\n'
                           'int main(){std::cout << prosper::embedded_build_source_identity();}\n',
                           encoding="utf-8")
        configure(args.cmake, source, build_dir, args.generator)

        def build(cmake, directory):
            build_target(cmake, directory, "q")

        def linked():
            return run([str(executable_path(build_dir, "q"))], cwd=build_dir)

        build(args.cmake, build_dir)
        original = linked()
        # MSVC or unsupported command syntax must visibly decline identity, never use a
        # configure-only or header-blind fallback. GNU/Clang are the supported positive slice.
        if original == "unknown":
            commands = build_dir / "compile_commands.json"
            driver = commands.read_text(encoding="utf-8") if commands.exists() else ""
            if any(word in driver for word in ("g++", "clang", "c++.exe")):
                raise AssertionError("supported dependency scan unexpectedly unavailable")
            check(original == "unknown", "unsupported compiler fails closed")
            print("source identity: unsupported compiler guard verified; no positive fidelity claim")
            return 0
        check(len(original) == 64 and all(x in "0123456789abcdef" for x in original),
              "supported linked identity is a full digest")
        build(args.cmake, build_dir)
        check(linked() == original, "unchanged incremental build retains identity")

        # Dirty edits do not require a commit, reconfigure or clean. The already linked binary
        # must remain immutable until the actual consumer is rebuilt.
        program.write_text(program.read_text(encoding="utf-8") + "// dirty source\n", encoding="utf-8")
        check(linked() == original, "post-build source edit cannot relabel old binary")
        build(args.cmake, build_dir)
        dirty = linked()
        check(dirty != original, "incremental dirty source edit changes identity")
        dependency.write_text("constexpr int dependency_value = 2;\n", encoding="utf-8")
        check(linked() == dirty, "post-build header edit cannot relabel old binary")
        build(args.cmake, build_dir)
        header = linked()
        check(header != dirty, "actual header dependency changes incremental identity")
        external.write_text("constexpr int external_dependency = 2;\n", encoding="utf-8")
        build(args.cmake, build_dir)
        external_header = linked()
        check(external_header != header, "dependency outside source glob changes identity")

        # This target-only definition is declared AFTER the revision library. It must be
        # observed in resolved compile commands, not only an early configure variable census.
        (source / "CMakeLists.txt").write_text(project +
            "target_compile_definitions(q PRIVATE IDENTITY_TARGET_ONLY=17)\n",
            encoding="utf-8")
        build(args.cmake, build_dir)
        definition = linked()
        check(definition != external_header, "late target-only definition changes identity")
        untracked = source / "src/untracked.cpp"
        untracked.write_text("int untracked_compiled_source(){return 31;}\n", encoding="utf-8")
        build(args.cmake, build_dir)
        check(linked() != definition, "untracked globbed compiled source changes identity")

        # Invoke the actual generator with unavailable commands and then corrupt the supported
        # command layout. Neither is allowed to claim a complete fingerprint.
        generated = root / "guard.cpp"
        base = [args.cmake, f"-DPROSPER_REVISION_WORK_TREE={root}",
                "-DPROSPER_REVISION_CONFIG_ID=test",
                f"-DPROSPER_REVISION_TEMPLATE={module.parent / 'build_revision.cpp.in'}",
                f"-DPROSPER_REVISION_OUTPUT={generated}",
                f"-DPROSPER_REVISION_PYTHON={sys.executable}"]
        def generated_identity():
            text = generated.read_text(encoding="utf-8")
            return text.split("embedded_build_source_identity()", 1)[1].split("}", 1)[0]

        script = ["-P", str(module.parent / "GenerateBuildRevision.cmake")]
        run(base + script, cwd=root)
        check('return "unknown";' in generated_identity(), "missing commands fail closed")
        malformed = root / "malformed.json"
        malformed.write_text("[]", encoding="utf-8")
        run(base + [f"-DPROSPER_REVISION_COMPILE_COMMANDS={malformed}",
                    "-P", str(module.parent / "GenerateBuildRevision.cmake")], cwd=root)
        check('return "unknown";' in generated_identity(), "empty dependency inventory fails closed")
        actual_commands = build_dir / "compile_commands.json"
        run(base + [f"-DPROSPER_REVISION_COMPILE_COMMANDS={actual_commands}"] + script, cwd=root)
        original_command_identity = generated_identity()
        check('return "unknown";' not in original_command_identity, "actual resolved command inventory is available")
        altered = json.loads(actual_commands.read_text(encoding="utf-8"))
        for command in altered:
            if Path(command["file"]).resolve() == program.resolve():
                if "arguments" in command:
                    command["arguments"].append("-DIDENTITY_ONLY_COMMAND=3")
                else:
                    command["command"] += " -DIDENTITY_ONLY_COMMAND=3"
        changed = root / "changed-command.json"
        changed.write_text(json.dumps(altered), encoding="utf-8")
        run(base + [f"-DPROSPER_REVISION_COMPILE_COMMANDS={changed}"] + script, cwd=root)
        check(generated_identity() != original_command_identity and
              'return "unknown";' not in generated_identity(),
              "target-only resolved definition changes identity without source/config edits")
    print(f"source identity: {checks} real incremental/config/dependency guards passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as error:
        print(error.stdout or "", file=sys.stderr)
        print(f"FAIL: real build exited {error.returncode}", file=sys.stderr)
        sys.exit(1)
    except (AssertionError, OSError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
