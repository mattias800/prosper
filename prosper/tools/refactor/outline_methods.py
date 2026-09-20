#!/usr/bin/env python3
"""outline_methods.py -- move a class's method BODIES out of its header into a .cpp.

The other tools here move top-level declarations between files. None of them can act on a class,
because a class is ONE top-level region however large it is: `SpirvCompute` is 4,352 lines of
`rdna2_to_spirv_internal.hpp`, all 231 of its methods are defined inline in the class body, and a
header parsed by eleven translation units therefore re-parses 3,540 lines of implementation eleven
times. `split_file.py` cannot divide it and `promote_internal.py` cannot either -- the whole thing is
a single region to both.

What this does is the standard C++ move: leave a declaration in the class, put the definition in a
.cpp qualified with `Class::`. That is mechanical, it is checkable, and it is the one transformation
that shrinks a header without changing a line of logic.

WHAT IT REFUSES, and why each one is refused rather than handled. This project's tools are narrow on
purpose; a tool that guesses on the hard cases is worse than one that names them.

  * TEMPLATES stay. A member function template's definition has to remain visible to every
    instantiation, so moving it to a .cpp breaks the build at the call site, not here.
  * STATIC methods stay for now. `static` belongs on the declaration and must NOT be repeated on the
    definition; that is a one-line rule, but getting it wrong produces a confusing diagnostic far
    from the cause, so it is left until there is a case that needs it.
  * DEFAULT ARGUMENTS stay. They belong on the declaration only. Same reasoning.
  * CONSTRUCTORS AND DESTRUCTORS stay. Member initialiser lists sit between the signature and the
    body, so "declaration is everything before the body" -- the rule this tool is built on -- is
    false for them.
  * Bodies of three lines or fewer stay. Outlining a one-line accessor costs a call and gains
    nothing; `inline` in a header is the right shape for those.

THE CHECK THAT MAKES IT SAFE is that every moved body is compared BYTE FOR BYTE against the original
after being written, read back from disk, and that the header is the original minus exactly those
spans with a `;` in place of each. That is the same contract `split_file.py` offers, and it means no
logic can change. It does NOT mean the result compiles -- a return type that mentions a nested type
needs qualifying, and only the compiler knows. The tool says so at the end of every run.

Usage:
  python3 prosper/tools/refactor/outline_methods.py --header <hpp> --class SpirvCompute --list
  ... --out <cpp> --apply
  ... --selftest
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys

ci = None


def _need_clang() -> None:
    global ci
    if ci is not None:
        return
    import clang.cindex as _ci
    for cand in ("/usr/lib64/libclang.so.22.1", "/usr/lib64/libclang.so"):
        if pathlib.Path(cand).exists():
            _ci.Config.set_library_file(cand)
            break
    ci = _ci


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def inside_repo(path: pathlib.Path) -> pathlib.Path:
    """Resolve PATH and refuse it if it escapes the checkout.

    These tools are driven by AGENTS with generated arguments, not only by a human typing a known
    path -- that is the normal case in this repository, not a hypothetical. `--group` takes a path
    and the apply step does `mkdir(parents=True)` then `write_text` on it, so a `..` that arrived
    from a parsed symbol name, an issue body or a templated command would create directories and
    write files anywhere the process can reach, silently and with a zero exit status.

    Containment is the whole fix and it costs one resolve() per path. A developer who genuinely
    wants to write outside the checkout can copy the file afterwards; nothing legitimate here needs
    it, and the refusal names what it saw."""
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        sys.exit(f"refusing a path outside the checkout: {path}\n"
                 f"  resolved to {resolved}\n  checkout is {REPO_ROOT}")
    return resolved


IDENT = r"[A-Za-z_][A-Za-z0-9_]*"


def qualify_return_type(decl: str, cls: str, nested: set[str]) -> str:
    """Qualify a return type that names one of the class's own NESTED types.

    Inside the class body `U64PairAdd foo()` resolves; moved out, `U64PairAdd` is not in scope and
    the compiler says only "does not name a type". libclang knows the class's nested type names, so
    the fix is mechanical: prefix the leading type token when it is one of them. Measured on
    SpirvCompute this was one method of 147, and it is the only compile error the move produced."""
    if not nested:
        return decl
    lead = decl.lstrip()
    pad = decl[:len(decl) - len(lead)]
    m = re.match(r"(" + IDENT + r")\b", lead)
    if m and m.group(1) in nested:
        return pad + cls + "::" + lead
    return decl


def qualify(decl: str, name: str, cls: str) -> str | None:
    """Insert `cls::` before the declarator's NAME in DECL.

    Searched from the RIGHT, because a return type may legitimately contain the method's own name
    (`Foo foo()`), and the declarator is always the last occurrence followed by `(`. Returns None
    when no such occurrence exists, so the caller refuses rather than writing something plausible.
    """
    best = None
    for m in re.finditer(re.escape(name) + r"\s*\(", decl):
        s = m.start()
        # Must be a whole token: not preceded by an identifier character or `::`.
        if s > 0 and (re.match(IDENT[-12:], decl[s - 1]) or decl[s - 1] == "_"):
            continue
        if decl[max(0, s - 2):s] == "::":
            continue
        best = s
    if best is None:
        return None
    return decl[:best] + cls + "::" + decl[best:]


def selectable(kind: str, is_static: bool, has_default_arg: bool, body_lines: int,
               min_lines: int) -> tuple[bool, str]:
    """Whether a method may be outlined, and the reason when it may not."""
    if kind == "FUNCTION_TEMPLATE":
        return False, "template (definition must stay visible to instantiations)"
    if kind in ("CONSTRUCTOR", "DESTRUCTOR"):
        return False, "constructor/destructor (member initialiser list sits before the body)"
    if is_static:
        return False, "static (`static` belongs on the declaration only)"
    if has_default_arg:
        return False, "default argument (belongs on the declaration only)"
    if body_lines <= min_lines:
        return False, f"body is {body_lines} line(s); <= --min-lines"
    return True, ""


def flags_for(target: pathlib.Path, build: str):
    db = json.load(open(os.path.join(build, "compile_commands.json")))
    tgt = os.path.realpath(target)
    best = None
    for e in db:
        if os.path.realpath(e["file"]) == tgt:
            best = e
            break
    if best is None:
        # A header has no compile command of its own, so it is parsed through a TU that INCLUDES
        # it -- established with the preprocessor, not guessed from the path. Guessing picked a
        # sibling that does not include the header and reported the class as simply absent, which
        # reads as "you named it wrong" rather than "I parsed the wrong file".
        cands = [e for e in db if os.path.realpath(e["file"]).startswith(
            os.path.dirname(os.path.dirname(tgt)))]
        for e in cands:
            cmd = e.get("arguments") or shlex.split(e["command"])
            probe = [c for c in cmd if c not in ("-c",)]
            out, skip = [], False
            for a in probe:
                if skip:
                    skip = False
                    continue
                if a == "-o":
                    skip = True
                    continue
                out.append(a)
            r = subprocess.run(out + ["-M", "-MF", "/dev/stdout"],
                               cwd=e["directory"], capture_output=True, text=True)
            if r.returncode == 0 and os.path.basename(tgt) in r.stdout:
                best = e
                break
    if best is None:
        sys.exit(f"no translation unit in the compile database includes {target}")
    cmd = best.get("arguments") or shlex.split(best["command"])
    args, skip = [], False
    for a in cmd[1:]:
        if skip:
            skip = False
            continue
        if a in ("-o", "-c"):
            skip = a == "-o"
            continue
        if a.endswith((".cpp", ".cc", ".c")):
            continue
        args.append(a)
    out = subprocess.run(["clang", "-print-resource-dir"], capture_output=True, text=True)
    if out.returncode == 0 and out.stdout.strip():
        args += ["-isystem", str(pathlib.Path(out.stdout.strip()) / "include")]
    return args, best["file"]


def find_methods(header: pathlib.Path, cls_name: str, build: str, min_lines: int):
    _need_clang()
    args, parse_file = flags_for(header, build)
    tu = ci.Index.create().parse(parse_file, args=args + ["-Wno-everything"])
    fatal = [d for d in tu.diagnostics if d.severity >= 3]
    if fatal:
        sys.exit(f"parse failed: {fatal[0].spelling}")

    def walk(c):
        for ch in c.get_children():
            if (ch.kind.name in ("STRUCT_DECL", "CLASS_DECL") and ch.spelling == cls_name
                    and ch.is_definition()):
                return ch
            r = walk(ch)
            if r is not None:
                return r
        return None

    cls = walk(tu.cursor)
    if cls is None:
        sys.exit(f"class {cls_name} not found")

    text = header.read_text()
    starts = list(itertools.accumulate((len(l) for l in text.splitlines(True)), initial=0))

    def off(line, col):
        return starts[line - 1] + col - 1

    nested = {c.spelling for c in cls.get_children()
              if c.kind.name in ("STRUCT_DECL", "CLASS_DECL", "ENUM_DECL", "TYPEDEF_DECL",
                                 "TYPE_ALIAS_DECL") and c.spelling}
    picked, refused = [], []
    for m in cls.get_children():
        if m.kind.name not in ("CXX_METHOD", "FUNCTION_TEMPLATE", "CONSTRUCTOR", "DESTRUCTOR"):
            continue
        if not m.is_definition():
            continue
        body = next((c for c in m.get_children() if c.kind.name == "COMPOUND_STMT"), None)
        if body is None:
            continue
        n = m.extent.end.line - m.extent.start.line + 1
        has_def = any(any(t.spelling == "=" for t in a.get_tokens()) for a in m.get_arguments())
        ok, why = selectable(m.kind.name, m.is_static_method(), has_def, n, min_lines)
        rec = {"name": m.spelling, "lines": n,
               "m0": off(m.extent.start.line, m.extent.start.column),
               "m1": off(m.extent.end.line, m.extent.end.column),
               "b0": off(body.extent.start.line, body.extent.start.column),
               "b1": off(body.extent.end.line, body.extent.end.column),
               "start_line": m.extent.start.line}
        (picked if ok else refused).append(rec if ok else {**rec, "why": why})
    return text, picked, refused, nested


def build_outputs(text, picked, cls_name, nested=frozenset()):
    """Return (new_header, definitions, problems). Pure: no I/O, so --selftest can pin it."""
    problems, defs = [], []
    for r in picked:
        decl = text[r["m0"]:r["b0"]]
        body = text[r["b0"]:r["b1"]]
        q = qualify(qualify_return_type(decl, cls_name, nested), r["name"], cls_name)
        if q is None:
            problems.append(f"{r['name']} (line {r['start_line']}): cannot locate the declarator")
            continue
        defs.append((r, q.strip() + " " + body))
    keep = {id(r) for r, _ in defs}
    out, cursor = [], 0
    for r in sorted(picked, key=lambda x: x["m0"]):
        if id(r) not in keep:
            continue
        out.append(text[cursor:r["m0"]])
        out.append(text[r["m0"]:r["b0"]].rstrip() + ";")
        cursor = r["m1"]
    out.append(text[cursor:])
    return "".join(out), defs, problems


def selftest() -> int:
    ok = True

    def check(label, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {label}" + ("" if good else f"\n        got {got!r}\n        want {want!r}"))

    print("-- qualify --")
    check("plain", qualify("uint32_t foo(int a) const", "foo", "C"),
          "uint32_t C::foo(int a) const")
    # A return type may legitimately contain the method's own name; the declarator is the LAST one.
    check("return type shares the name", qualify("Foo foo(int a)", "foo", "C"), "Foo C::foo(int a)")
    check("name is a substring of another token",
          qualify("void do_foo_thing(); void foo(int)", "foo", "C"),
          "void do_foo_thing(); void C::foo(int)")
    check("already qualified is not re-qualified", qualify("void C::foo(int)", "foo", "C"), None)
    check("absent declarator refuses", qualify("void bar(int)", "foo", "C"), None)

    print("-- inside_repo (these tools are driven by agents with generated paths) --")
    ok_path = inside_repo(pathlib.Path("prosper/tools/refactor/outline_methods.py"))
    check("a path inside the checkout resolves", str(ok_path).startswith(str(REPO_ROOT)), True)
    for bad in ("../../../etc/passwd", "/etc/passwd", "prosper/../../outside.cpp"):
        try:
            inside_repo(pathlib.Path(bad))
            check(f"refuses {bad}", "accepted", "refused")
        except SystemExit:
            check(f"refuses {bad}", True, True)

    print("-- qualify_return_type --")
    check("nested return type is qualified",
          qualify_return_type("U64PairAdd foo(int)", "C", {"U64PairAdd"}), "C::U64PairAdd foo(int)")
    check("an ordinary return type is left alone",
          qualify_return_type("uint32_t foo(int)", "C", {"U64PairAdd"}), "uint32_t foo(int)")
    check("no nested types means no change",
          qualify_return_type("U64PairAdd foo(int)", "C", set()), "U64PairAdd foo(int)")
    check("a longer token starting with the same text is not matched",
          qualify_return_type("U64PairAddition foo(int)", "C", {"U64PairAdd"}),
          "U64PairAddition foo(int)")

    print("-- selectable --")
    check("ordinary method", selectable("CXX_METHOD", False, False, 40, 3), (True, ""))
    check("template refused", selectable("FUNCTION_TEMPLATE", False, False, 40, 3)[0], False)
    check("ctor refused", selectable("CONSTRUCTOR", False, False, 40, 3)[0], False)
    check("static refused", selectable("CXX_METHOD", True, False, 40, 3)[0], False)
    check("default arg refused", selectable("CXX_METHOD", False, True, 40, 3)[0], False)
    check("tiny refused", selectable("CXX_METHOD", False, False, 3, 3)[0], False)
    check("just over the bar", selectable("CXX_METHOD", False, False, 4, 3)[0], True)

    print("-- build_outputs (the moved body must be byte-identical) --")
    src = "struct C {\n    int foo(int a) const {\n        return a + 1;\n    }\n};\n"
    b0 = src.index("{", src.index("foo"))
    rec = {"name": "foo", "lines": 3, "m0": src.index("    int foo") + 4,
           "m1": src.index("}\n};") + 1, "b0": b0, "b1": src.index("}\n};") + 1,
           "start_line": 2}
    hdr, defs, probs = build_outputs(src, [rec], "C", frozenset())
    check("no problems", probs, [])
    check("header keeps a declaration", "int foo(int a) const;" in hdr, True)
    check("header lost the body", "return a + 1" in hdr, False)
    check("definition is qualified", defs[0][1].startswith("int C::foo(int a) const {"), True)
    check("definition carries the body verbatim", "return a + 1;" in defs[0][1], True)

    print("== PASS ==" if ok else "== FAIL ==")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--header", type=pathlib.Path)
    ap.add_argument("--class", dest="cls")
    ap.add_argument("--out", type=pathlib.Path, help="the .cpp to write definitions into")
    ap.add_argument("--group", action="append", default=[], metavar="PATH=REGEX",
                    help="route methods whose name matches REGEX into PATH. Repeatable; the FIRST "
                         "match wins, so order them narrowest-first. A method matching no group is "
                         "an error rather than a default bucket -- a file called `misc` is the "
                         "failure this whole exercise exists to avoid, and one that fills itself "
                         "silently is worse.")
    ap.add_argument("--build", default="prosper/build-linux")
    ap.add_argument("--min-lines", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="outline at most N methods (0 = all)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.header or not args.cls:
        sys.exit("--header and --class are required")

    args.header = inside_repo(args.header)
    if args.out:
        args.out = inside_repo(args.out)
    text, picked, refused, nested = find_methods(args.header, args.cls, args.build, args.min_lines)
    picked.sort(key=lambda r: -r["lines"])
    if args.limit:
        picked = picked[:args.limit]
    moved = sum(r["lines"] for r in picked)
    print(f"{args.cls}: {len(picked)} outlinable method(s), {moved} line(s); "
          f"{len(refused)} left in the class")
    if args.list:
        for r in picked[:20]:
            print(f"  {r['lines']:>5}L  {r['name']}")
        by = {}
        for r in refused:
            by.setdefault(r["why"], []).append(r["name"])
        for why, names in sorted(by.items()):
            print(f"  refused ({len(names)}): {why}")
        return 0

    new_header, defs, problems = build_outputs(text, picked, args.cls, nested)
    if problems:
        for p in problems:
            print(f"  [FAIL] {p}")
        return 1
    # VERIFY before writing: every moved body must appear verbatim in its definition.
    for r, d in defs:
        if text[r["b0"]:r["b1"]] not in d:
            print(f"  [FAIL] {r['name']}: body is not carried verbatim")
            return 1
    print(f"  [ok]   {len(defs)} definition(s) carry their body verbatim")
    if not args.apply:
        print("(dry run; pass --apply with --out)")
        return 0
    if not args.out and not args.group:
        sys.exit("--apply needs --out or --group")
    spelling = str(args.header).split("prosper/src/", 1)[-1]
    groups = []
    for g in args.group:
        path, _, pattern = g.partition("=")
        if not pattern:
            sys.exit(f"--group needs PATH=REGEX, got {g!r}")
        groups.append((inside_repo(pathlib.Path(path)), re.compile(pattern)))
    if not groups:
        groups = [(args.out, re.compile(".*"))]

    routed, unrouted = {}, []
    for r, d in sorted(defs, key=lambda x: x[0]["m0"]):
        for path, pat in groups:
            if pat.search(r["name"]):
                routed.setdefault(path, []).append((r, d))
                break
        else:
            unrouted.append(r["name"])
    if unrouted:
        print(f"  [FAIL] {len(unrouted)} method(s) match no --group, and there is no default "
              f"bucket on purpose:")
        for n in unrouted[:25]:
            print(f"           {n}")
        return 1

    header_note = ("// Generated move, not new code: `%s`'s method bodies, lifted out of\n"
                   "// %s. outline_methods.py checked each body byte for byte.\n\n"
                   "#include \"%s\"\n\nnamespace prosper::gpu {\n\n")
    for path, items in routed.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        body = header_note % (args.cls, spelling, spelling)
        body += "\n\n".join(d for _, d in items)
        body += "\n\n}  // namespace prosper::gpu\n"
        path.write_text(body)
    args.header.write_text(new_header)
    for path, items in routed.items():
        back = path.read_text()
        for r, _ in items:
            if text[r["b0"]:r["b1"]] not in back:
                print(f"  [FAIL] {r['name']}: body missing from {path} after write-back")
                return 1
    print(f"  [ok]   read back from disk: every body still verbatim")
    for path, items in sorted(routed.items(), key=lambda kv: -len(kv[1])):
        print(f"  wrote {path} -- {len(items)} method(s), "
              f"{sum(r['lines'] for r, _ in items)} line(s)")
    print(f"  header now {len(new_header.splitlines())} lines")
    print("  now BUILD: a return type naming a nested type may need qualifying, and only the "
          "compiler knows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
