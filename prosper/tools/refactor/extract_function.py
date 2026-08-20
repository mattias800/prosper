#!/usr/bin/env python3
"""extract_function.py -- shrink a giant function by extracting statement runs, using clangd.

The other tools here move whole declarations between files. They cannot make a FUNCTION smaller, and
several functions in this codebase are the size of a normal source file on their own --
`register_live_renderer` is 8,222 lines, 89% of live_renderer.cpp, so no amount of file-splitting
touches it.

Extract-method is the operation that does, and it is not something to hand-roll: deciding which
locals become parameters, which are read-only (`const&`) and which are mutated (`&`), is a data-flow
analysis, and getting it subtly wrong changes behaviour silently. clangd already implements it
correctly. This drives clangd.

WHY THIS IS NOT A SHELL ONE-LINER. `clang-refactor extract` -- the tool that looks right -- is marked
"(WIP action; use with caution!)" and does NOT compute captures: it emitted `static void
accumulate()` taking no parameters for a loop reading `v` and mutating `total`. clangd's
implementation is a different one and is correct, but it is reachable only over LSP, as a code action
carrying a `clangd.applyTweak` command whose edit arrives as a `workspace/applyEdit` request FROM the
server. So this speaks the protocol.

CANDIDATES COME FROM THE AST, NOT FROM LINE PATTERNS, and that is the other half of the tool. Picking
spans by regex over lines produces ranges that start inside a comment or halfway through a
continuation line; clangd then declines, and the decline looks like a tool limitation rather than a
bad question. Every candidate here is a run of complete sibling statements taken from libclang, with
exact extents.

  python3 prosper/tools/refactor/extract_function.py --file <f> --function <name> --list
  python3 prosper/tools/refactor/extract_function.py --file <f> --function <name> --apply N --name foo
"""

import argparse
import json
import os
import pathlib
import queue
import subprocess
import sys
import threading

import clang.cindex as ci

for cand in ("/usr/lib64/libclang.so.22.1", "/usr/lib64/libclang.so"):
    if pathlib.Path(cand).exists():
        ci.Config.set_library_file(cand)
        break

END_SLACK = int(os.environ.get('EXTRACT_END_SLACK', '2'))

ESCAPING = (ci.CursorKind.RETURN_STMT, ci.CursorKind.BREAK_STMT,
            ci.CursorKind.CONTINUE_STMT, ci.CursorKind.GOTO_STMT)


def repo_root() -> pathlib.Path:
    return pathlib.Path(subprocess.run(["git", "rev-parse", "--show-toplevel"],
                                       capture_output=True, text=True, check=True).stdout.strip())


def flags_for(target: pathlib.Path, db: pathlib.Path) -> list[str]:
    import shlex
    for e in json.loads(db.read_text()):
        if pathlib.Path(e["file"]).resolve() == target.resolve():
            args = shlex.split(e["command"]) if "command" in e else list(e["arguments"])
            out, skip = [], False
            for a in args[1:]:
                if skip:
                    skip = False
                    continue
                if a in ("-c", "-o"):
                    skip = a == "-o"
                    continue
                if a.endswith((".cpp", ".cc", ".c")):
                    continue
                out.append(a)
            res = subprocess.run(["clang", "-print-resource-dir"], capture_output=True, text=True)
            if res.returncode == 0:
                out += ["-isystem", str(pathlib.Path(res.stdout.strip()) / "include")]
            return out
    sys.exit(f"no compile command for {target}")


def has_escape(cursor) -> bool:
    """Does this statement transfer control out of the ENCLOSING function?

    The descent stops at a lambda. A `return` inside `auto f = [](){ return 1; };` returns from the
    lambda, not from the function holding it, so counting it as an escape rejects the statement --
    and in this codebase that rejects almost everything, because the giant functions are built as
    sequences of lambda definitions. `register_live_renderer` has 40 direct statements, 19 of them
    lambda-bearing declarations; treating their bodies as escapes left ZERO candidates.
    """
    stack = [cursor]
    while stack:
        c = stack.pop()
        if c.kind in ESCAPING:
            return True
        if c is not cursor and c.kind == ci.CursorKind.LAMBDA_EXPR:
            continue                      # its returns belong to it, not to us
        stack.extend(c.get_children())
    return False


def names_declared(cursor) -> set[str]:
    out = set()
    stack = [cursor]
    while stack:
        c = stack.pop()
        if c.kind == ci.CursorKind.VAR_DECL and c.spelling:
            out.add(c.spelling)
        stack.extend(c.get_children())
    return out


def names_referenced(cursor) -> set[str]:
    out = set()
    stack = [cursor]
    while stack:
        c = stack.pop()
        if c.kind == ci.CursorKind.DECL_REF_EXPR and c.spelling:
            out.add(c.spelling)
        stack.extend(c.get_children())
    return out


def candidates(path: pathlib.Path, func: str, flags: list[str], min_stmts: int,
               min_lines: int, max_lines: int):
    tu = ci.Index.create().parse(str(path), args=flags)
    fatal = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
    if fatal:
        print(f"  WARNING {len(fatal)} parse error(s); candidates may be wrong")

    def find(c):
        if (c.kind in (ci.CursorKind.FUNCTION_DECL, ci.CursorKind.CXX_METHOD)
                and c.spelling == func and c.is_definition()
                and c.location.file and pathlib.Path(c.location.file.name).resolve() == path.resolve()):
            return c
        for ch in c.get_children():
            r = find(ch)
            if r:
                return r
        return None

    fn = find(tu.cursor)
    if not fn:
        sys.exit(f"function {func} not found in {path}")
    body = next((c for c in fn.get_children() if c.kind == ci.CursorKind.COMPOUND_STMT), None)
    if not body:
        sys.exit(f"{func} has no body")

    # SCAN THE LAMBDA BODIES TOO -- that is where the lines actually are. register_live_renderer has
    # 40 direct statements across 8,217 lines, because 19 of them are `auto x = [](...){ ... };` with
    # hundreds of lines inside. Treating only the top level as extractable finds one candidate: the
    # whole function.
    scopes = [body]
    stack = [body]
    while stack:
        c = stack.pop()
        for ch in c.get_children():
            stack.append(ch)
            if ch.kind == ci.CursorKind.LAMBDA_EXPR:
                lb = next((g for g in ch.get_children()
                           if g.kind == ci.CursorKind.COMPOUND_STMT), None)
                if lb is not None:
                    scopes.append(lb)

    out = []
    for scope in scopes:
        out.extend(_runs_in(scope, min_stmts, min_lines, max_lines))
    return _dedup(out)


def _runs_in(scope, min_stmts, min_lines, max_lines):
    stmts = list(scope.get_children())

    # Everything referenced AFTER a run decides whether its declarations may leave. Precomputed as a
    # suffix so the scan stays linear rather than re-walking the tail for every candidate.
    suffix_refs = [set() for _ in range(len(stmts) + 1)]
    for i in range(len(stmts) - 1, -1, -1):
        suffix_refs[i] = suffix_refs[i + 1] | names_referenced(stmts[i])

    out = []
    i = 0
    while i < len(stmts):
        if has_escape(stmts[i]):
            i += 1
            continue
        j = i
        declared: set[str] = set()
        while j < len(stmts) and not has_escape(stmts[j]):
            declared |= names_declared(stmts[j])
            j += 1
            n = j - i
            span = stmts[j - 1].extent.end.line - stmts[i].extent.start.line + 1
            if (n >= min_stmts and min_lines <= span <= max_lines
                    and not (declared & suffix_refs[j])):
                out.append({
                    "stmts": n, "lines": span,
                    "start_line": stmts[i].extent.start.line,
                    "start_col": stmts[i].extent.start.column,
                    "end_line": stmts[j - 1].extent.end.line,
                    "end_col": stmts[j - 1].extent.end.column,
                    "declares": sorted(declared),
                })
        i += 1
    return out


def _dedup(out):
    """Longest first, dropping overlaps, so the list reads as choices rather than permutations."""
    out.sort(key=lambda c: -c["lines"])
    chosen, used = [], set()
    for c in out:
        rng = set(range(c["start_line"], c["end_line"] + 1))
        if used & rng:
            continue
        used |= rng
        chosen.append(c)
    chosen.sort(key=lambda c: c["start_line"])
    return chosen


# --- the LSP half -------------------------------------------------------------------------------
class Clangd:
    def __init__(self, db_dir: str):
        self.p = subprocess.Popen(["clangd", "--log=error", f"--compile-commands-dir={db_dir}"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)
        self.id = 0
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while True:
            head = b""
            while not head.endswith(b"\r\n\r\n"):
                b = self.p.stdout.read(1)
                if not b:
                    self.q.put(None)
                    return
                head += b
            n = int([h for h in head.decode().split("\r\n")
                     if h.lower().startswith("content-length")][0].split(":")[1])
            self.q.put(json.loads(self.p.stdout.read(n)))

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.p.stdin.write(b"Content-Length: %d\r\n\r\n" % len(data) + data)
        self.p.stdin.flush()

    def request(self, method, params):
        self.id += 1
        self._send({"jsonrpc": "2.0", "id": self.id, "method": method, "params": params})
        return self.id

    def notify(self, method, params):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def wait(self, want_id=None, want_method=None, timeout=900):
        import time
        end = time.time() + timeout
        while time.time() < end:
            try:
                m = self.q.get(timeout=timeout)
            except queue.Empty:
                return None
            if m is None:
                return None
            if want_id is not None and m.get("id") == want_id and "method" not in m:
                return m
            if want_method and m.get("method") == want_method:
                if "id" in m:
                    self._send({"jsonrpc": "2.0", "id": m["id"], "result": {"applied": True}})
                return m
            if "id" in m and "method" in m:
                self._send({"jsonrpc": "2.0", "id": m["id"], "result": None})
        return None

    def open(self, path: pathlib.Path):
        self.uri = "file://" + str(path.resolve())
        rid = self.request("initialize", {
            "processId": os.getpid(), "rootUri": "file://" + str(repo_root()),
            "capabilities": {"workspace": {"applyEdit": True},
                             "textDocument": {"codeAction": {"codeActionLiteralSupport": {
                                 "codeActionKind": {"valueSet": ["refactor"]}}}}}})
        self.wait(want_id=rid)
        self.notify("initialized", {})
        self.notify("textDocument/didOpen", {"textDocument": {
            "uri": self.uri, "languageId": "cpp", "version": 1, "text": path.read_text()}})
        # Wait for diagnostics for THIS uri: clangd publishes for headers too, and acting before the
        # main file has an AST looks exactly like "no refactoring available".
        for _ in range(60):
            m = self.wait(want_method="textDocument/publishDiagnostics")
            if not m or m["params"].get("uri") == self.uri:
                return

    def extract(self, c: dict, new_name: str | None):
        # END PAST THE SEMICOLON. libclang's statement extent stops at the last token, which for an
        # expression statement is BEFORE the `;`. A range ending there is a partial statement, and
        # clangd declines partial selections -- indistinguishable from "this cannot be extracted".
        rng = {"start": {"line": c["start_line"] - 1, "character": c["start_col"] - 1},
               "end": {"line": c["end_line"] - 1, "character": c["end_col"] - 1 + END_SLACK}}
        rid = self.request("textDocument/codeAction", {
            "textDocument": {"uri": self.uri}, "range": rng,
            "context": {"diagnostics": [], "only": ["refactor"]}})
        resp = self.wait(want_id=rid) or {}
        actions = resp.get("result") or []
        act = next((a for a in actions if "function" in (a.get("title") or "").lower()), None)
        if not act:
            return None, [a.get("title") for a in actions]
        cmd = act.get("command") or {}
        rid = self.request("workspace/executeCommand",
                           {"command": cmd.get("command"), "arguments": cmd.get("arguments", [])})
        applied = self.wait(want_method="workspace/applyEdit")
        return applied, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, type=pathlib.Path)
    ap.add_argument("--function", required=True)
    ap.add_argument("--build", default="prosper/build-linux")
    ap.add_argument("--min-stmts", type=int, default=3)
    ap.add_argument("--min-lines", type=int, default=15)
    ap.add_argument("--max-lines", type=int, default=400,
                    help="upper bound; without one, the whole function body is the single longest candidate and swallows every useful chunk")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="ask clangd about EVERY candidate in one session and report "
                         "which it accepts. A candidate this tool considers valid can still be refused, and knowing which is the whole planning question.")
    ap.add_argument("--apply", type=int, help="candidate index to extract")
    ap.add_argument("--name", default="extracted")
    args = ap.parse_args()

    root = repo_root()
    path = root / args.file if not args.file.is_absolute() else args.file
    db = root / args.build / "compile_commands.json"
    flags = flags_for(path, db)

    cands = candidates(path, args.function, flags, args.min_stmts, args.min_lines,
                       args.max_lines)
    print(f"== {args.function} in {args.file}: {len(cands)} extractable candidate(s) ==")
    print("   (complete sibling statements; no return/break/continue; nothing declared escapes)")
    for n, c in enumerate(cands[:25]):
        print(f"  [{n:3d}] lines {c['start_line']:6d}-{c['end_line']:<6d} "
              f"{c['lines']:4d}L {c['stmts']:3d} stmt(s)"
              + (f"  declares {c['declares'][:3]}" if c["declares"] else ""))
    if args.probe:
        cd = Clangd(str(root / args.build))
        cd.open(path)
        ok = 0
        for n, c in enumerate(cands):
            rng = {"start": {"line": c["start_line"] - 1, "character": c["start_col"] - 1},
                   "end": {"line": c["end_line"] - 1, "character": c["end_col"] - 1 + END_SLACK}}
            rid = cd.request("textDocument/codeAction", {
                "textDocument": {"uri": cd.uri}, "range": rng,
                "context": {"diagnostics": [], "only": ["refactor"]}})
            acts = (cd.wait(want_id=rid) or {}).get("result") or []
            titles = [a.get("title") for a in acts]
            good = any("function" in (t or "").lower() for t in titles)
            ok += good
            print(f"  [{n:3d}] lines {c['start_line']:6d}-{c['end_line']:<6d} {c['lines']:4d}L  "
                  f"{'ACCEPTED' if good else 'refused '}  {titles if titles else ''}")
        print(f"\n  clangd accepts {ok} of {len(cands)}")
        return 0

    if args.list or args.apply is None:
        return 0

    c = cands[args.apply]
    print(f"\n  extracting candidate {args.apply}: lines {c['start_line']}-{c['end_line']}")
    cd = Clangd(str((root / args.build)))
    cd.open(path)
    applied, offered = cd.extract(c, args.name)
    if not applied:
        print(f"  clangd declined this range; it offered: {offered or '(nothing)'}")
        return 1
    changes = applied["params"]["edit"].get("changes") or {}
    total = sum(len(v) for v in changes.values())
    print(f"  clangd produced {total} edit(s):")
    for uri, edits in changes.items():
        for e in edits:
            txt = e["newText"]
            if txt.strip():
                print("  " + "\n  ".join(txt.splitlines()[:6]))
                if len(txt.splitlines()) > 6:
                    print(f"  ... ({len(txt.splitlines())} lines)")
    print("\n  NOT written to disk: pass the edits through review, then BUILD and run ctest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
