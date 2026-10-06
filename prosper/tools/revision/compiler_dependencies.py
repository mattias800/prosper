"""Bounded build-time dependency fingerprint for GCC/Clang prosper_core commands.

This runs before compilation, using current resolved compile commands, not stale .d files or a
configure-only source census. The output is ONLY a digest. Unsupported invocation/dependency
syntax fails closed; CMake embeds unknown rather than claiming a complete compiler case.
"""
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

MAX_UNITS, MAX_PATHS, MAX_BYTES = 1024, 16384, 256 * 1024 * 1024


def argv(command):
    if os.name != "nt":
        return shlex.split(command)
    import ctypes
    from ctypes import wintypes
    split = ctypes.windll.shell32.CommandLineToArgvW
    split.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    split.restype = ctypes.POINTER(wintypes.LPWSTR)
    count = ctypes.c_int()
    result = split(command, ctypes.byref(count))
    if not result:
        raise ValueError("unsupported compiler command")
    try:
        return [result[k] for k in range(count.value)]
    finally:
        ctypes.windll.kernel32.LocalFree(ctypes.cast(result, ctypes.c_void_p))


DRIVER = re.compile(r"(?:[\w.-]+-)?(?:g\+\+|gcc|c\+\+|clang\+\+|clang)(?:-\d+(?:\.\d+)*)?(?:\.exe)?")
WRAPPER = re.compile(r"ccache(?:\.exe)?")
# ccache settings that replace the compiler lookup this module reproduces. Any of them set means the
# compiler that actually ran is not the one PATH names, so the identity refuses rather than guess.
WRAPPER_OVERRIDES = ("compiler", "path", "prefix_command")


@lru_cache(maxsize=8)   # once per wrapper binary, not once per compile command
def wrapper_overrides_unset(wrapper):
    for key in WRAPPER_OVERRIDES:
        p = subprocess.run([str(wrapper), "-k", key], capture_output=True, text=True, timeout=10)
        if p.returncode != 0:
            raise ValueError("compiler wrapper configuration unavailable")
        if p.stdout.strip():
            raise ValueError("compiler wrapper configuration overrides the compiler")


def find_on_path(name, search_path, wrapper):
    """`name` on PATH as the wrapper finds it: the first match that is not the wrapper itself.

    A relative or empty entry ahead of the match would be resolved against the compile's working
    directory, not ours, and a DIFFERENT wrapper would be the next link of a chain; both refuse.
    """
    for entry in search_path.split(os.pathsep):
        if not entry or not os.path.isabs(entry):
            raise ValueError("relative PATH entry ahead of the wrapped compiler")
        candidate = Path(entry) / name
        if not (candidate.is_file() and os.access(candidate, os.X_OK)):
            continue
        target = candidate.resolve(strict=True)
        if target == wrapper:
            continue
        if WRAPPER.fullmatch(target.name) or target.name == "sccache":
            raise ValueError("compiler wrapper chain unsupported")
        return target
    raise ValueError("wrapped compiler not found")


def unwrap_compiler(args, search_path=None):
    """The real compiler behind a ccache wrapper, and the argv that compiler receives.

    A cache wrapper does not change what is compiled, so the identity is the compiler it forwards
    to. Two spellings reach compile_commands.json (#4356): the explicit `ccache g++ ...`, and the
    masquerade `/usr/lib64/ccache/c++ ...`, a symlink to ccache that finds the real `c++` by
    searching PATH and skipping links to itself. Both are resolved here the way ccache resolves
    them, and only when none of ccache's own overrides (`compiler`, `path`, `prefix_command`) is
    set; anything else fails closed. Other wrappers (sccache) are not recognised and so refuse as
    unsupported drivers.
    """
    first = Path(args[0])
    resolved = first.resolve(strict=True)
    if not WRAPPER.fullmatch(resolved.name):
        return resolved, args
    wrapper_overrides_unset(resolved)
    search = search_path if search_path is not None else os.environ.get("PATH", "")
    if WRAPPER.fullmatch(first.name):
        # Explicit form: the compiler is the next argument.
        if len(args) < 2 or args[1].startswith("-"):
            raise ValueError("compiler wrapper without a compiler")
        named = Path(args[1])
        if named.is_absolute():
            real = named.resolve(strict=True)
            if real == resolved or WRAPPER.fullmatch(real.name) or real.name == "sccache":
                raise ValueError("compiler wrapper chain unsupported")
        elif len(named.parts) != 1:
            raise ValueError("relative wrapped compiler path unsupported")
        else:
            real = find_on_path(args[1], search, resolved)
        rest = args[1:]
    else:
        # Masquerade: the same basename, later on PATH.
        real = find_on_path(first.name, search, resolved)
        rest = args
    return real, [str(real)] + list(rest[1:])


def dependencies(command):
    args = command.get("arguments") or argv(command["command"])
    if not args or any(x.startswith("@") for x in args):
        raise ValueError("response/wrapped compiler invocation unsupported")
    compiler, args = unwrap_compiler(args)
    if not DRIVER.fullmatch(compiler.name):
        raise ValueError("compiler driver unsupported")
    filtered = [str(compiler)]
    skip = False
    for x in args[1:]:
        if skip:
            skip = False
            continue
        if x in ("-o", "-MF", "-MT", "-MQ"):
            skip = True
        elif x not in ("-c", "-MD", "-MMD", "-MP"):
            filtered.append(x)
    if skip or any(x.startswith("-fplugin") or x in ("-include-pch", "-fmodules") for x in filtered):
        raise ValueError("compiler dependency extensions unsupported")
    p = subprocess.run(filtered + ["-M", "-MT", "prosper"], cwd=command["directory"],
                       capture_output=True, timeout=30)
    if p.returncode or len(p.stdout) > 8 * 1024 * 1024:
        raise ValueError("compiler dependency scan unavailable")
    text = p.stdout.decode("utf-8", "strict").replace("\\\r\n", " ").replace("\\\n", " ")
    if not text.startswith("prosper:") or "\n" in text.strip():
        raise ValueError("compiler dependency layout unsupported")
    # GCC Make-style escaping, not shell syntax. Preserve Windows separators; only escapes
    # of whitespace/#/$/backslash are decoded. Literal dollar paths are unsupported rather than
    # mistaken for Make expansion. Dependency existence is checked before any hashing.
    text = text[len("prosper:"):].strip()
    tokens, current = [], ""
    k = 0
    while k < len(text):
        ch = text[k]
        if ch == "\\" and k + 1 < len(text) and text[k + 1] in " \t#\\":
            current += text[k + 1]
            k += 2
            continue
        if ch.isspace():
            if current:
                tokens.append(current)
                current = ""
        else:
            if ch == "$":
                raise ValueError("Make expansion dependency unsupported")
            current += ch
        k += 1
    if current:
        tokens.append(current)
    if not tokens or len(tokens) > MAX_PATHS:
        raise ValueError("compiler dependency count")
    paths = {compiler}
    for token in tokens:
        path = Path(token)
        if not path.is_absolute():
            path = Path(command["directory"]) / path
        paths.add(path.resolve(strict=True))
    # GCC's actual front-end is separate from its driver; clang normally uses its driver binary.
    front = subprocess.run([str(compiler), "-print-prog-name=cc1plus"], capture_output=True,
                           text=True, timeout=10)
    if front.returncode == 0 and front.stdout.strip() != "cc1plus":
        paths.add(Path(front.stdout.strip()).resolve(strict=True))
    return paths


def secondary_commands(commands_path, root, required):
    """Same-sysroot producer argv emitted by the opt-in mixed Windows build seam."""
    # Sources are compared after resolve(), so the root must be resolved too: a work tree reached
    # through a symlink would otherwise own none of its own files.
    root = Path(root).resolve(strict=True)
    directory = commands_path.parent / "gnu-variadic"
    manifests = sorted(directory.glob("*.obj.argv"))
    if len(manifests) > 16:
        raise ValueError("secondary command count")
    commands, encoded = [], bytearray()
    for manifest in manifests:
        if manifest.stat().st_size > 64 * 1024:
            raise ValueError("secondary command size")
        data = manifest.read_bytes()
        if b"\0" in data or b"\r" in data:
            raise ValueError("secondary command layout")
        arguments = [line for line in data.decode("utf-8", "strict").split("\n") if line]
        if len(arguments) > 1024 or arguments.count("-c") != 1:
            raise ValueError("secondary command arguments")
        index = arguments.index("-c")
        if index + 1 >= len(arguments):
            raise ValueError("secondary source missing")
        source = Path(arguments[index + 1]).resolve(strict=True)
        if not source.is_relative_to(root / "prosper"):
            raise ValueError("secondary source ownership")
        if source.is_relative_to(root / "prosper/src"):
            commands.append({"file": str(source), "directory": str(commands_path.parent),
                             "arguments": arguments})
            encoded.extend(manifest.name.encode("utf-8") + b"\0" + data)
    producing_source = root / "prosper/src/hle/libc/libc_variadic_capture.cpp"
    if required and not any(Path(command["file"]) == producing_source for command in commands):
        raise ValueError("secondary producing command unavailable")
    return commands, encoded


def fingerprint(commands_file, root, secondary_required=False):
    commands_path = Path(commands_file)
    root = Path(root).resolve(strict=True)   # see secondary_commands
    if commands_path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("compile command budget")
    commands = json.loads(commands_path.read_text(encoding="utf-8"))
    selected = [c for c in commands if Path(c["file"]).resolve().is_relative_to(root / "prosper" / "src")]
    additional, secondary = secondary_commands(commands_path, root, secondary_required)
    selected.extend(additional)
    if not selected or len(selected) > MAX_UNITS:
        raise ValueError("core command count")
    paths = set()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for dependency_set in pool.map(dependencies, selected):
            paths.update(dependency_set)
            if len(paths) > MAX_PATHS:
                raise ValueError("aggregate dependency count")
    h = hashlib.sha256(commands_path.read_bytes())
    h.update(secondary)
    size = 0
    for path in sorted(paths, key=lambda p: p.as_posix()):
        size += path.stat().st_size
        if size > MAX_BYTES:
            raise ValueError("aggregate dependency byte budget")
        # Project paths are sorted/relative. External names remain local to this build-time digest;
        # neither paths nor the manifest are embedded/exported into cases or public diagnostics.
        name = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()
        h.update(name.encode("utf-8") + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


if __name__ == "__main__":
    try:
        if len(sys.argv) not in (3, 4):
            raise ValueError("arguments")
        required = False
        if len(sys.argv) == 4:
            value = sys.argv[3].upper()
            if value not in ("", "0", "1", "OFF", "ON", "FALSE", "TRUE"):
                raise ValueError("secondary policy argument")
            required = value in ("1", "ON", "TRUE")
        print(fingerprint(sys.argv[1], Path(sys.argv[2]).resolve(strict=True), required))
    except Exception as error:
        # This module's own refusals are plain ValueErrors with fixed, path-free phrases, so their
        # reason is safe to show. Anything else (OSError, JSON/Unicode decode errors, subprocess
        # failures) can carry a private path or file content and is named by type only.
        reason = str(error) if type(error) is ValueError else type(error).__name__
        print(f"compiler dependency identity unavailable: {reason}", file=sys.stderr)
        sys.exit(2)
