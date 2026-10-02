"""Audited TEST_LOCAL member-fact declarations, not a CMake/C++ reachability evaluator.

The maintained source-role contract and its limits are in MEMBER_FACT_DOMAINS.md. Every refusal
retains shared facts. The caller still collects, inventories and locally scans every source.
"""
from __future__ import annotations

import os
import re
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

PROPERTY = "PROSPER_DIAG_MEMBER_FACT_DOMAIN"
IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9.-]*\Z")
BRACKET = re.compile(r"\[(=*)\[")


@dataclass(frozen=True)
class Command:
    name: str
    args: tuple[str, ...]
    literal: tuple[bool, ...]
    path: Path
    line: int
    definition: bool = False
    complete: bool = True


@dataclass(frozen=True)
class Role:
    target: str
    source: str
    location: str
    accepted: bool
    reason: str

    def report(self) -> str:
        verdict = "TEST_LOCAL" if self.accepted else "REFUSED (shared)"
        return (f"member-domain: {verdict} target={self.target or '<unknown>'} "
                f"source={self.source or '<unknown>'} at {self.location}: {self.reason}")


def commands(text: str, path: Path) -> list[Command]:
    """Tokenize command syntax; comments/quoted/bracket payloads cannot execute commands.

    This deliberately does not expand variables, execute conditions, includes or functions. A
    marker in a function/macro definition is refused, rather than asserted to have run.
    """
    tokens: list[tuple[str, bool, int]] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        comment = ch == "#"
        start = i + int(comment)
        bracket = BRACKET.match(text, start)
        if bracket:
            end = text.find("]" + bracket.group(1) + "]", bracket.end())
            end = len(text) if end < 0 else end + len(bracket.group(1)) + 2
            if not comment:
                tokens.append((text[i:end], False, i))
            i = end
        elif comment:
            end = text.find("\n", i)
            i = len(text) if end < 0 else end
        elif ch == '"':
            start = i
            i += 1
            value = []
            while i < len(text) and text[i] != '"':
                if text[i] == "\\" and i + 1 < len(text):
                    i += 1
                value.append(text[i])
                i += 1
            i += int(i < len(text))
            tokens.append(("".join(value), False, start))
        elif ch in "()":
            tokens.append((ch, True, i))
            i += 1
        else:
            start = i
            while i < len(text) and not text[i].isspace() and text[i] not in '()#"':
                i += 1
            tokens.append((text[start:i], True, start))

    result = []
    definitions = 0
    i = 0
    while i + 1 < len(tokens):
        name, plain, offset = tokens[i]
        if not plain or not IDENT.fullmatch(name) or tokens[i + 1][0] != "(":
            i += 1
            continue
        depth, j = 1, i + 2
        while j < len(tokens) and depth:
            if tokens[j][1]:
                depth += {"(": 1, ")": -1}.get(tokens[j][0], 0)
            j += 1
        args = tokens[i + 2:j - 1] if not depth else tokens[i + 2:]
        name = name.lower()
        result.append(Command(name, tuple(t[0] for t in args), tuple(t[1] for t in args),
                              path, text.count("\n", 0, offset) + 1, bool(definitions), not depth))
        if name in ("function", "macro"):
            definitions += 1
        elif name in ("endfunction", "endmacro"):
            definitions = max(0, definitions - 1)
        i = j
    return result


def _identity(root: Path, scope: Path, token: str) -> Path | None:
    # A quoted literal path is fine for reuse checks, but variable/genex/list sources are opaque.
    if not token or any(ch in token for ch in "$;[]*"):
        return None
    path = scope / token
    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return resolved


def discover(root: Path, files: dict[Path, list[str]], excluded: Callable[[str], bool]
             ) -> tuple[set[Path], list[Role]]:
    """Return admitted single test TUs and visible accepted/refused role records.

    Enumerated guards catch literal misuse and constructions mentioning the marked target or
    source. Unrelated opaque CMake/macro generation is not interpreted: the assertion must still
    be true under build review. All unannotated files and all headers remain shared.
    """
    # Every path below is compared in CANONICAL form (`_identity` resolves), but callers key
    # `files` under their own spelling of the root. Those differ whenever the root sits under a
    # symlink -- on macOS every temporary directory does (/var -> /private/var) -- and then no
    # resolved source is ever "in files" and `relative_to(root)` raises (#4043). So translate the
    # keys once on the way in and translate the admitted set back on the way out.
    caller_root, root = root, root.resolve()
    canonical = {root / path.relative_to(caller_root): path for path in files}
    cmake = []
    include_files = set(canonical)
    for directory, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not excluded(d) and d != ".git")
        for name in sorted(names):
            path = Path(directory) / name
            if name == "CMakeLists.txt" or path.suffix == ".cmake":
                cmake.extend(commands(path.read_text(encoding="utf-8"), path))
            elif path.suffix == ".inc":
                include_files.add(path)  # literal inclusion may pass through a non-scanned .inc
    markers = [c for c in cmake if PROPERTY in c.args]
    roles: list[Role] = []
    local: set[Path] = set()
    for marker in markers:
        args = marker.args
        target = args[1] if len(args) > 1 and args[0] == "TARGET" else ""
        source = None
        failures = []
        supported = (marker.name == "set_property" and len(args) == 5
                     and args[:1] == ("TARGET",) and args[2:] == ("PROPERTY", PROPERTY, "TEST_LOCAL")
                     and all(marker.literal) and re.fullmatch(r"test_[A-Za-z_0-9]+", target))
        if not supported or marker.definition or not marker.complete:
            failures.append("unsupported marker syntax/value or definition scope")
        if sum(1 for c in markers if len(c.args) > 1 and c.args[1] == target) != 1:
            failures.append("duplicate marker")
        declarations = [c for c in cmake if c.name in ("add_executable", "add_library")
                        and c.args and c.args[0] == target]
        if len(declarations) != 1:
            failures.append("missing or duplicate target declaration")
        else:
            declaration = declarations[0]
            if (declaration.name != "add_executable" or len(declaration.args) != 2
                    or not all(declaration.literal) or declaration.definition or not declaration.complete):
                failures.append("target requires one literal executable source")
            else:
                spelling = declaration.args[1]
                source = _identity(root, declaration.path.parent, spelling)
                rel = source.relative_to(root).as_posix() if source else ""
                lexical = Path(os.path.abspath(declaration.path.parent / spelling))
                if (source is None or source != lexical or not source.is_file()
                        or not rel.startswith("tests/") or source.suffix != ".cpp"
                        or source not in canonical):
                    failures.append("source must be an existing collected tests/*.cpp without symlinks")
        registered = [c for c in cmake if c.name == "add_test" and not c.definition
                      and c.complete
                      and len(c.args) == 4 and c.args[0] == "NAME" and c.args[2] == "COMMAND"
                      and c.args[3] == target and all(c.literal)]
        if len(registered) != 1:
            failures.append("requires one direct literal add_test(NAME ... COMMAND target)")
        if source:
            for command in cmake:
                if command is marker or command in declarations or command in registered:
                    continue
                a = command.args
                target_mentioned = any(target == v or target in re.findall(r"[A-Za-z_0-9.-]+", v)
                                       for v in a)
                paths = [_identity(root, command.path.parent, v) for v in a]
                source_mentioned = source in paths or any(source.name in v for v in a)
                # These commands can add/reuse/export sources or alter the declared role. The
                # marker cannot exempt a target that is also a library dependency or alias.
                if command.name in ("target_sources", "add_executable", "add_library",
                                    "install", "export", "set_property",
                                    "set_target_properties", "set_source_files_properties"):
                    if target_mentioned or source_mentioned:
                        failures.append(f"known source/target reuse or modification: {command.name}")
                if command.name == "target_link_libraries":
                    # The test's ordinary outbound backend dependencies remain shared. Only an
                    # inbound link (or a source passed as a link item) contradicts its role.
                    if (target in a[1:] or (a and a[0] != target and target_mentioned)
                            or source_mentioned):
                        failures.append("known inbound target/source reuse: target_link_libraries")
                # Source-valued variable/glob/copy constructions naming this source are refused.
                # Other opaque variables/macros are outside the explicit contract's guard scope.
                if command.name in ("set", "list", "file", "configure_file"):
                    if source_mentioned or any(source.name in v for v in a):
                        failures.append(f"source construction requires shared facts: {command.name}")
                    elif command.name == "file" and a and a[0] in ("GLOB", "GLOB_RECURSE"):
                        for token in a[2:]:
                            for pattern in token.split(";"):
                                if not any(ch in pattern for ch in "*?[") or "$" in pattern:
                                    continue
                                absolute = os.path.abspath(command.path.parent / pattern)
                                prefix = re.split(r"[*?\[]", pattern, 1)[0]
                                base = _identity(root, command.path.parent, prefix) if prefix else root
                                if (fnmatch.fnmatchcase(str(source), absolute)
                                        or (base and (base == source.parent or base in source.parents))):
                                    failures.append("literal glob may collect registered source")
                if (target_mentioned or source_mentioned) and command.name in (
                        "include", "add_subdirectory", "cmake_language"):
                    failures.append(f"opaque role modification: {command.name}")
                safe_metadata = {"target_include_directories", "target_compile_definitions",
                                 "target_compile_features", "target_compile_options",
                                 "target_link_options", "target_link_directories"}
                understood = safe_metadata | {"add_executable", "add_library", "add_test",
                    "target_sources", "target_link_libraries", "install", "export", "set_property",
                    "set_target_properties", "set_source_files_properties", "set", "list", "file",
                    "configure_file", "include", "add_subdirectory", "cmake_language"}
                if (target_mentioned or source_mentioned) and command.name not in understood:
                    failures.append(f"unsupported role-affecting command: {command.name}")
            include_re = re.compile(r'^\s*#\s*include\s*[<"]([^>"\n]+)[>"]', re.M)
            for path in sorted(include_files):
                text = path.read_text(encoding="utf-8")
                # Only literal CPP inclusion is claimed. Macro/preprocessor dependency closure
                # is a build-review obligation, not an invented lexical reachability proof.
                for include in include_re.finditer(text):
                    value = include.group(1)
                    if source in (_identity(root, path.parent, value), _identity(root, root, value)):
                        failures.append(f"literal CPP inclusion: {path.relative_to(root).as_posix()}")
        location = f"{marker.path.relative_to(root).as_posix()}:{marker.line}"
        accepted = not failures
        roles.append(Role(target, source.relative_to(root).as_posix() if source else "", location,
                          accepted, "; ".join(sorted(set(failures))) if failures
                          else "declared nonshipping TU; literal guards passed"))
        if accepted:
            local.add(canonical[source])
    return local, roles
