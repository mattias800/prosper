#!/usr/bin/env python3
"""hle_handler_map — which Sony NIDs are answered by ONE shared prosper HLE handler?

A runtime return-value histogram (`tools/hle_calls --values`) keys on the **prosper handler
symbol**, not on the Sony function. When one handler is registered for several Sony entry points,
all of them collapse into a single row — typically reading `0x0` — and **a mismodelled answer for
one of them is indistinguishable from a correct answer for all of them**. That is the residue the
`registered-but-mismodelled Sony value` frontier (#1905) is left with, and no runtime instrument
can see it, because at runtime the collapse has already happened.

`nid_gate_scan.py --all-nids` (#2056) answers the other axis: which imports a title *branches on*.
Cross the two and you get the only list that matters — **the gated NIDs whose answer is shared with
some other Sony function, and therefore cannot be trusted from a `--values` row alone**:

    nid_gate_scan.py <DUMP_ROOT>/PPSA05325-app0/eboot.bin --all-nids \
        --names ../PS5-3.20_Libs > gated.txt
    hle_handler_map.py --names ../PS5-3.20_Libs --gated gated.txt

This tool is the registration half. It parses prosper's OWN registration tables under `src/hle/`
and reports `NID -> (Sony name, prosper handler, how many distinct Sony names that handler serves)`.

WHY A PARSER AND NOT A GREP (#2070). The measurement was first taken with an ad-hoc regex, and an
ad-hoc regex over this codebase has three failure modes that all look like a smaller answer rather
than like a broken tool:

  * **It recognises the registration forms that existed the day it was written.** prosper registers
    through `Hle::register_fn` / `Hle::register_placeholder`, but almost never *directly*: there are
    file-local `#define R(...)` / `RN(...)` / `RN_SUBMIT(...)` / `RN_SUBMIT_NAMED(...)` macros and
    file-local lambdas, fourteen macro definitions and two lambdas at the time of writing. A new one
    is simply invisible to a fixed pattern, and the census shrinks with nothing looking wrong.
  * **`hle_kernel_mem.cpp` defines `register_kernel_mem_hle()` TWICE**, ~3,200 lines apart, in the
    two arms of one `#if defined(__linux__) || defined(__APPLE__)`. A line-based grep counts both
    arms, which inflates every "distinct names this handler serves" figure that touches libkernel.
    This tool evaluates the conditionals for a chosen `--platform` and reports what it skipped.
  * **It cannot say what it missed.** See the coverage contract below.

So: the shapes are DISCOVERED, not hardcoded. The registration API list comes from parsing the
`class Hle` declaration in `dispatch.hpp`; the wrapper list comes from parsing every `#define` and
every lambda whose body calls one of those APIs, with the argument mapping read out of the wrapper's
own body. A macro nobody has written yet is picked up the day it is written.

COVERAGE CONTRACT — how to read a zero from this tool
-----------------------------------------------------
"No collapses" and "I parsed nothing" must never be the same output (#2149), so every run prints a
coverage block BEFORE any table: files scanned, registration APIs found, wrappers discovered,
sites claimed, NIDs resolved, and — the load-bearing one — **sites UNCLAIMED**.

An unclaimed site is a textual mention of a registration API, or of a discovered wrapper, that the
parser could not turn into a registration. It is the tool's own answer to "did I miss a shape?", and
it is checked by a deliberately over-broad detector that is INDEPENDENT of the shape list: it counts
mentions, while the shape list consumes them. A shape the parser does not know still gets counted as
a mention, so it surfaces as a non-zero residual instead of vanishing.

    exit 0   the scan ran; the printed counts ARE the answer, zero included
    exit 2   refused — nothing was parsed, and no number printed is a result
    exit 3   the scan ran but is INCOMPLETE: at least one registration site was unclaimed.
             Every table printed is a lower bound. Fix the parser before quoting a number.

An *unresolved argument expression* at a site that WAS claimed is a different, milder thing (e.g.
`kUlt[kIdxInitialize].nid`, an array-driven registration whose NID is not a literal). Those are
listed by raw expression, counted, and do NOT change the exit code — the site is accounted for, only
its NID is not a compile-time literal here. They are printed so an unresolved entry stays actionable.

LIBRARY ATTRIBUTION IS A DEFINITION, AND IT IS STATED (#2070)
--------------------------------------------------------------
An earlier reading of this measurement appeared to show two extractions disagreeing (13 vs 37). They
did not: both were the same 41 rows, differing only over whether `libkernel` was folded into "libc".
That is a definition, not a result. This tool therefore reports **per-library counts and never a
rolled-up "libc"/"non-libc" subtotal**, using the library file names of the PS5 3.20 firmware dump
verbatim — so `libkernel` and `libSceLibcInternal` are two different rows and no reader has to guess
which bucket `libkernel` fell into. Attribution comes from `nid_gate_scan.load_nid_names`, i.e. the
first `libSceXxx.c` in sorted order that exports the NID; that rule is shared with the tool this one
is crossed against, so the two cannot disagree about a library.

Usage:
    hle_handler_map.py [--src DIR] [--platform linux|windows|macos] [--names DIR]
                       [--gated FILE] [--registry FILE] [--all] [--min-names N]

    --src        prosper source root holding `src/hle` (default: inferred from this file's path)
    --platform   which arm of the `#if` tree to evaluate (default: the host platform)
    --names      PS5 3.20 firmware genstub dump, for Sony names and library attribution
    --gated      output of `nid_gate_scan.py --all-nids`; restricts the cross to those NIDs
    --registry   an `hle_registry_dump` TSV; reconciles the parse against the compiled binary
    --all        list every registered NID, not only the shared-handler ones
    --min-names  a handler must serve at least N distinct Sony names to be listed (default 2)
"""
import argparse
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nid_gate_scan as G                                            # noqa: E402  (composed, not copied)


# ---------------------------------------------------------------- NID hashing

SONY_B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-"
# Sony's published 16-byte NID salt, the same constant `src/hle/nid.cpp` uses.
NID_SALT = bytes([0x51, 0x8D, 0x64, 0xA6, 0x35, 0xDE, 0xD8, 0xC1,
                  0xE6, 0xB0, 0x39, 0xB1, 0xC3, 0xE5, 0x52, 0x30])
NID_RE = re.compile(r"^[A-Za-z0-9+\-]{11}$")


def nid_hash(name):
    """Sony's symbol NID for `name` — a Python port of `prosper::nid_hash` (src/hle/nid.cpp:66).

    SHA-1 over name||salt, first 8 digest bytes reversed to little-endian, Sony-alphabet base64
    without padding, first 11 characters. Ported rather than shelled out to so the tool needs no
    build; `test_hle_handler_map.py` pins it against the firmware dump, which is an oracle written
    by neither this file nor nid.cpp.
    """
    d = hashlib.sha1(name.encode() + NID_SALT).digest()
    r = d[7::-1]                                    # first 8 bytes, reversed
    out = []
    for i in range(0, 6, 3):                        # two whole 3-byte groups
        v = (r[i] << 16) | (r[i + 1] << 8) | r[i + 2]
        out += [SONY_B64[(v >> 18) & 63], SONY_B64[(v >> 12) & 63],
                SONY_B64[(v >> 6) & 63], SONY_B64[v & 63]]
    v = (r[6] << 16) | (r[7] << 8)                  # 2-byte tail -> 3 characters
    out += [SONY_B64[(v >> 18) & 63], SONY_B64[(v >> 12) & 63], SONY_B64[(v >> 6) & 63]]
    return "".join(out)[:11]


# ---------------------------------------------------------------- lexing helpers

def strip_comments(text):
    """Blank out // and /* */ comments, PRESERVING every character position and line break.

    Positions must survive because line numbers are reported and because wrapper bodies are tracked
    by character span. Comments are replaced with spaces (newlines kept) rather than deleted.

    String and character literals are honoured, so a `//` inside a literal is not treated as a
    comment. That matters less for the argument text than for the CONVERSE case this protects:
    `hle_service.cpp` carries registration lines with a trailing `// ...` comment that itself
    contains parentheses and commas, and a naive strip that ran inside a literal would desynchronise
    the whole file from that point on.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            q = c
            i += 1
            while i < n and text[i] != q:
                i += 2 if text[i] == "\\" else 1
            i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            for j in range(i, min(i + 2, n)):
                out[j] = " "
            i += 2
        else:
            i += 1
    return "".join(out)


def split_args(text, open_idx):
    """Split the argument list whose '(' is at `open_idx`.

    Returns (args, index just past the matching ')'), or (None, None) if the parens never close.
    Splits on TOP-LEVEL commas only, tracking (), [], {} and string/char literals.

    Angle brackets are deliberately NOT tracked: `<` and `>` are ambiguous with the comparison
    operators, and a heuristic that guessed wrong would silently mis-split an argument list. The
    consequence is that a template argument containing a comma would produce the wrong arity — which
    is why arity is CHECKED against the callee's declared parameter count and a mismatch is reported
    as an unclaimed site rather than quietly accepted. No registration in the tree needs it today
    (`(HleFn)&glog_thunk<Offset + Is>` has no comma), and if one ever does, the residual says so.
    """
    depth = 0
    args, cur = [], []
    i, n = open_idx, len(text)
    while i < n:
        c = text[i]
        if c in "\"'":
            q = c
            j = i + 1
            while j < n and text[j] != q:
                j += 2 if text[j] == "\\" else 1
            cur.append(text[i:j + 1])
            i = j + 1
            continue
        if c in "([{":
            depth += 1
            if depth == 1 and c == "(":
                i += 1
                continue                             # the opening paren itself is not argument text
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append("".join(cur))
                if len(args) == 1 and not args[0].strip():
                    args = []                        # a zero-argument call, not one empty argument
                return args, i + 1
        elif c == "," and depth == 1:
            args.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    return None, None


def line_of(text, idx):
    return text.count("\n", 0, idx) + 1


# ---------------------------------------------------------------- preprocessor arms

# The only conditional identifiers that guard a registration site in `src/hle` today. Every one of
# them is KNOWN — either defined for the chosen platform or known to be undefined — so the
# "uncertain" path below is currently unreachable. It exists because the day someone guards a
# registration with a build-system define is the day a silently-included or silently-excluded arm
# would move a count, and an uncertain condition must be visible rather than guessed.
PLATFORM_DEFINES = {
    "linux":   {"__linux__"},
    "windows": {"_WIN32", "_WIN64"},
    "macos":   {"__APPLE__"},
}
KNOWN_COND_IDENTS = {"__linux__", "_WIN32", "_WIN64", "__APPLE__"}

ON, OFF, UNCERTAIN = True, False, None


def eval_condition(expr, defines):
    """Evaluate a `#if`/`#elif` expression to ON / OFF / UNCERTAIN.

    Handles `defined(X)`, `defined X`, bare identifiers, integer literals, `!`, `&&`, `||` and
    parentheses. An identifier outside KNOWN_COND_IDENTS makes the whole condition UNCERTAIN — it
    is NOT silently taken as 0 (which is what C would do for an undefined macro), because a
    build-system define read as 0 would drop a real registration arm and shrink the census with
    nothing looking wrong. UNCERTAIN keeps both arms and says so.
    """
    e = expr.strip()
    if not e:
        return UNCERTAIN
    idents = set(re.findall(r"\b[A-Za-z_]\w*\b", e)) - {"defined"}
    if idents - KNOWN_COND_IDENTS:
        return UNCERTAIN
    e = re.sub(r"defined\s*\(\s*(\w+)\s*\)", lambda m: "1" if m.group(1) in defines else "0", e)
    e = re.sub(r"defined\s+(\w+)", lambda m: "1" if m.group(1) in defines else "0", e)
    e = re.sub(r"\b[A-Za-z_]\w*\b", lambda m: "1" if m.group(0) in defines else "0", e)
    e = e.replace("&&", " and ").replace("||", " or ").replace("!", " not ")
    try:
        return ON if eval(e, {"__builtins__": {}}, {}) else OFF   # noqa: S307 — digits/and/or/not only
    except Exception:                                             # noqa: BLE001
        return UNCERTAIN


def active_lines(text, platform):
    """Per-line activity mask for `platform`, plus the conditions that could not be decided.

    Returns (active[1..nlines], uncertain[(line, directive)]). A line is active when no ENCLOSING
    frame is OFF; an UNCERTAIN frame is treated as active, so both arms of an undecidable condition
    are kept and the caller can report them.
    """
    defines = PLATFORM_DEFINES[platform]
    lines = text.split("\n")
    active = [True] * (len(lines) + 1)
    uncertain = []
    stack = []                                        # [cur_state, taken] per open #if
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        m = re.match(r"#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)", s)
        if m:
            kind, rest = m.group(1), m.group(2).strip()
            if kind in ("ifdef", "ifndef"):
                ident = rest.split()[0] if rest.split() else ""
                if ident not in KNOWN_COND_IDENTS:
                    val = UNCERTAIN
                    uncertain.append((i, s[:100]))
                else:
                    val = ON if (ident in defines) == (kind == "ifdef") else OFF
                stack.append([val, val])
            elif kind == "if":
                val = eval_condition(rest, defines)
                if val is UNCERTAIN:
                    uncertain.append((i, s[:100]))
                stack.append([val, val])
            elif kind == "elif":
                if stack:
                    taken = stack[-1][1]
                    if taken is ON:
                        stack[-1][0] = OFF
                    elif taken is UNCERTAIN:
                        stack[-1][0] = UNCERTAIN
                    else:
                        val = eval_condition(rest, defines)
                        if val is UNCERTAIN:
                            uncertain.append((i, s[:100]))
                        stack[-1][0] = val
                        stack[-1][1] = val if val is not OFF else OFF
            elif kind == "else":
                if stack:
                    taken = stack[-1][1]
                    stack[-1][0] = OFF if taken is ON else (UNCERTAIN if taken is UNCERTAIN else ON)
            elif kind == "endif":
                if stack:
                    stack.pop()
            active[i] = all(f[0] is not OFF for f in stack)
            continue
        active[i] = all(f[0] is not OFF for f in stack)
    return active, uncertain


# ---------------------------------------------------------------- shape discovery

class Wrapper:
    """A macro or lambda that forwards to a registration API.

    `params` are its own parameter names in order; `fwd_args` are the argument EXPRESSIONS it passes
    to `api`, written in terms of those parameters. Expanding a call is textual whole-token
    substitution of actuals for parameters into each forwarded argument, which is exactly what the
    preprocessor does for the macro case and what the compiler does for the lambda case.
    """

    def __init__(self, name, kind, params, api, fwd_args, file, line, body_span):
        self.name, self.kind, self.params = name, kind, params
        self.api, self.fwd_args = api, fwd_args
        self.file, self.line, self.body_span = file, line, body_span
        self.scope = (body_span[1], 1 << 62)      # where calls to it expand; narrowed by #undef

    def signature(self):
        return (self.name, tuple(self.params), self.api, tuple(a.strip() for a in self.fwd_args))

    def expand(self, actuals):
        if len(actuals) != len(self.params):
            return None
        table = dict(zip(self.params, (a.strip() for a in actuals)))
        out = []
        for a in self.fwd_args:
            out.append(re.sub(r"\b[A-Za-z_]\w*\b",
                              lambda m: table.get(m.group(0), m.group(0)), a))
        return out


def discover_apis(dispatch_hpp_text):
    """Registration entry points, read out of the `class Hle` declaration.

    Hardcoding {register_fn, register_placeholder} would be one more thing that silently stops being
    true. Reading the class means a third registration API is picked up the day it is declared.
    """
    body = re.search(r"class\s+Hle\s*\{(.*?)\n\};", dispatch_hpp_text, re.S)
    scope = body.group(1) if body else dispatch_hpp_text
    return sorted(set(re.findall(r"\bstatic\s+[\w:<>]+\s+(register_\w+)\s*\(", scope)))


C_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "sizeof", "do", "else",
              "template", "namespace", "class", "struct", "operator", "decltype", "static_assert"}


def _matching_brace(text, open_idx):
    depth, i, n = 0, open_idx, len(text)
    while i < n:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _forwards_to(bodytxt, targets):
    """(target name, forwarded argument expressions) for the first call in `bodytxt` to any of
    `targets`, or None. `targets` maps a callee name to the expected parameter count (None = any)."""
    for m in re.finditer(r"\b(?:Hle\s*::\s*)?(\w+)\s*\(", bodytxt):
        name = m.group(1)
        if name not in targets:
            continue
        args, _ = split_args(bodytxt, m.end() - 1)
        if args is None:
            continue
        return name, args
    return None


def discover_wrappers(text, apis, path):
    """Every macro, lambda and free function in `text` that FORWARDS to a registration API.

    Recognised structurally, never by name, so `R`, `RN`, `RN_SUBMIT`, `RN_SUBMIT_NAMED`, `reg` and
    anything added later are all found the same way — that is the whole anti-drift property, since a
    fixed pattern list is exactly what made the first version of this measurement a scratch script.

    Two rules do the work:

      * **A candidate must forward one of its OWN PARAMETERS into the NID slot.** Without this,
        `register_service_hle()` — a zero-parameter function whose body is 300 registration calls —
        would itself be classified as a wrapper, its inner calls dismissed as "the wrapper's
        template", and the census would come back near-empty. The parameter test is what separates a
        forwarder from a registration BLOCK, and it is the single most load-bearing line here.
      * **Discovery is a FIXPOINT**, so a macro that wraps another macro is found on the second pass.
        A one-level scan would miss `#define R2(a,b) R(a,b)` entirely — and, worse, would miss it
        silently, because nothing textually mentions a registration API at the `R2(...)` call sites.
    """
    found = []
    targets = {a: None for a in apis}
    for _round in range(8):                                # fixpoint; 8 is far past any real nesting
        before = len(found)
        cands = []
        # --- function-like macros, including `\`-continued bodies
        for m in re.finditer(r"#\s*define\s+(\w+)\s*\(([^)]*)\)[ \t]*(.*(?:\\\n.*)*)", text):
            params = [p.strip() for p in m.group(2).split(",") if p.strip()]
            cands.append((m.group(1), "macro", params, m.group(3).replace("\\\n", " "),
                          (m.start(), m.start() + len(m.group(0))), m.start()))
        # --- `auto NAME = [](params) { ... };`
        for m in re.finditer(r"\bauto\s+(\w+)\s*=\s*\[[^\]]*\]\s*\(([^)]*)\)[^{;]*\{", text):
            close = _matching_brace(text, m.end() - 1)
            if close is None:
                continue
            params = [re.sub(r".*?(\w+)$", r"\1", p.strip())
                      for p in m.group(2).split(",") if p.strip()]
            cands.append((m.group(1), "lambda", params, text[m.end() - 1:close + 1],
                          (m.start(), close + 1), m.start()))
        # --- free functions `<ret> NAME(<typed params>) { ... }`
        for m in re.finditer(r"\b(\w+)\s*\(([^;{)]*)\)\s*\{", text):
            if m.group(1) in C_KEYWORDS or not m.group(2).strip():
                continue
            close = _matching_brace(text, m.end() - 1)
            if close is None:
                continue
            params = []
            for p in m.group(2).split(","):
                p = p.strip()
                mm = re.match(r"^[\w:<>,\s*&]*?(\w+)$", p)
                if mm:
                    params.append(mm.group(1))
            if not params:
                continue
            cands.append((m.group(1), "function", params, text[m.end() - 1:close + 1],
                          (m.start(), close + 1), m.start()))

        # Dedupe by SPAN, not by name. One file legitimately re-`#define`s `R` after `#undef`-ing it
        # — `hle_kernel_mem.cpp` does exactly that, once per `#if` arm — and a name-keyed skip drops
        # the second block's wrapper, silently losing every registration it makes.
        known_spans = {w.body_span for w in found}
        for name, kind, params, bodytxt, span, startpos in cands:
            if span in known_spans:
                continue
            fwd = _forwards_to(bodytxt, targets)
            if fwd is None:
                continue
            callee, args = fwd
            if len(args) < 1:
                continue
            # The discriminator: the NID slot must be built from one of this candidate's own
            # parameters. A registration BLOCK forwards nothing and is not a wrapper.
            if not any(re.search(r"\b%s\b" % re.escape(p), args[0]) for p in params):
                continue
            w = Wrapper(name, kind, params, callee, args, path, line_of(text, startpos), span)
            if kind == "macro":
                u = re.search(r"#\s*undef\s+%s\b" % re.escape(name), text[span[1]:])
                if u:
                    w.scope = (span[1], span[1] + u.start())
            found.append(w)
            targets.setdefault(name, len(params))
        if len(found) == before:
            break
    return found


# ---------------------------------------------------------------- argument resolution

def unwrap(expr):
    """Strip C casts, address-of, and redundant outer parentheses from an expression."""
    e = expr.strip()
    changed = True
    while changed:
        changed = False
        e2 = re.sub(r"^\((?:HleFn|prosper::HleFn|void\s*\*)\)\s*", "", e)
        if e2 != e:
            e, changed = e2, True
        if e.startswith("&"):
            e, changed = e[1:].strip(), True
        if e.startswith("(") and e.endswith(")"):
            args, end = split_args(e, 0)
            if end == len(e) and args is not None and len(args) == 1:
                e, changed = args[0].strip(), True
    return e


STRING_RE = re.compile(r'^"((?:[^"\\]|\\.)*)"$')


def resolve_nid(expr):
    """(nid, how) for a NID argument, or (None, raw expression) when it is not a literal.

    Two literal forms exist: a raw 11-character NID string (`"hVYD7Ou2pCQ"`), and `nid_hash("name")`
    for a documented Sony symbol. Anything else — an array element, a table field — is UNRESOLVED
    and reported by its raw text, never dropped.
    """
    e = unwrap(expr)
    m = STRING_RE.match(e)
    if m:
        s = m.group(1)
        return (s, "raw") if NID_RE.match(s) else (None, e)
    m = re.match(r'^(?:prosper::)?nid_hash\s*\(\s*"((?:[^"\\]|\\.)*)"\s*\)$', e)
    if m:
        return nid_hash(m.group(1)), "nid_hash"
    return None, e


def resolve_string(expr):
    m = STRING_RE.match(unwrap(expr))
    return m.group(1) if m else None


def resolve_handler(expr):
    e = unwrap(expr)
    return e if re.match(r"^[\w:]+(<.*>)?$", e) else e


# ---------------------------------------------------------------- the scan

class Registration:
    def __init__(self, nid, nid_how, name, handler, file, line, shape, api):
        self.nid, self.nid_how, self.name = nid, nid_how, name
        self.handler, self.file, self.line = handler, file, line
        self.shape, self.api = shape, api


class Scan:
    def __init__(self):
        self.regs = []
        self.unresolved = []        # (file, line, shape, raw nid expression)
        self.unclaimed = []         # (file, line, why, text)     <- the load-bearing residual
        self.uncertain_conds = []   # (file, line, directive)
        self.wrappers = []
        self.apis = []
        self.files = []
        self.skipped_lines = 0
        self.out_of_scope = 0       # `NAME(` outside the wrapper's #define..#undef span


def scan_tree(src_hle, platform):
    """Parse every `.cpp` under `src_hle` for the chosen platform arm."""
    sc = Scan()
    hpp = os.path.join(src_hle, "dispatch.hpp")
    if not os.path.isfile(hpp):
        raise SystemExit("refused: %s not found — is --src the prosper source root?" % hpp)
    sc.apis = discover_apis(strip_comments(open(hpp, errors="ignore").read()))
    if not sc.apis:
        raise SystemExit("refused: no `static register_*` in dispatch.hpp's class Hle — "
                         "the registration API moved and every count below would be zero")

    paths = sorted(p for p in os.listdir(src_hle) if p.endswith(".cpp"))
    for fn in paths:
        path = os.path.join(src_hle, fn)
        raw = open(path, errors="ignore").read()
        text = strip_comments(raw)
        active, unc = active_lines(text, platform)
        sc.uncertain_conds += [(fn, ln, d) for ln, d in unc]
        sc.files.append(fn)

        # Blank every inactive line so the scanners below see only the chosen platform arm, while
        # character offsets — and therefore reported line numbers — stay exact.
        chars = list(text)
        pos = 0
        for i, line in enumerate(text.split("\n"), 1):
            if not active[i]:
                sc.skipped_lines += 1
                for j in range(pos, pos + len(line)):
                    chars[j] = " "
            pos += len(line) + 1
        text = "".join(chars)

        wrappers = discover_wrappers(text, sc.apis, fn)

        # Two ACTIVE wrappers of the same name with OVERLAPPING scopes would each claim the other's
        # call sites and double every count they touch. Identical re-definitions are harmless (drop
        # the duplicate); differing ones are genuinely ambiguous and must be loud, never guessed.
        kept = []
        for w in wrappers:
            clash = next((k for k in kept if k.name == w.name
                          and not (w.scope[1] <= k.scope[0] or k.scope[1] <= w.scope[0])), None)
            if clash is None:
                kept.append(w)
            elif clash.signature() != w.signature():
                sc.unclaimed.append((fn, w.line,
                                     "two active definitions of wrapper '%s' with different bodies "
                                     "(also at line %d)" % (w.name, clash.line), w.name))
        wrappers = kept
        sc.wrappers += wrappers
        by_name = {w.name: w for w in wrappers}
        wrapper_spans = [w.body_span for w in wrappers]

        def in_wrapper_body(idx):
            return any(a <= idx < b for a, b in wrapper_spans)

        # ---- direct `Hle::register_*(...)` calls
        for m in re.finditer(r"\bHle\s*::\s*(register_\w+)\s*\(", text):
            api = m.group(1)
            if api not in sc.apis:
                sc.unclaimed.append((fn, line_of(text, m.start()), "unknown registration API",
                                     "Hle::%s" % api))
                continue
            if in_wrapper_body(m.start()):
                continue                                # this is a wrapper's template, not a site
            args, end = split_args(text, m.end() - 1)
            if args is None:
                sc.unclaimed.append((fn, line_of(text, m.start()), "unbalanced argument list",
                                     text[m.start():m.start() + 60]))
                continue
            tail = text[end:end + 40].lstrip()
            if tail.startswith("{"):
                continue                                # a definition (dispatch.cpp), not a call
            _record(sc, args, fn, line_of(text, m.start()), "direct", api)

        # ---- calls through a discovered wrapper, inside that wrapper's own scope
        for w in wrappers:
            for m in re.finditer(r"\b%s\s*\(" % re.escape(w.name), text):
                if in_wrapper_body(m.start()):
                    continue                            # the #define / lambda itself
                actuals, end = split_args(text, m.end() - 1)
                ln = line_of(text, m.start())
                if actuals is None:
                    sc.unclaimed.append((fn, ln, "unbalanced argument list",
                                         text[m.start():m.start() + 60]))
                    continue
                if text[end:end + 40].lstrip().startswith("{"):
                    continue                            # a definition of something else, not a call
                if not (w.scope[0] <= m.start() < w.scope[1]):
                    sc.out_of_scope += 1                # some other `R(` — not a registration
                    continue
                # Follow the chain: a wrapper may forward to another wrapper rather than straight
                # to a registration API (`#define R2(a,b) R(a,b)`). Expand until an API is reached.
                cur, cur_args, chain = w, actuals, [w.name]
                failure = None
                for _hop in range(8):
                    expanded = cur.expand(cur_args)
                    if expanded is None:
                        failure = ("arity mismatch: %s takes %d, call passes %d"
                                   % (cur.name, len(cur.params), len(cur_args)))
                        break
                    if cur.api in sc.apis:
                        break
                    nxt = by_name.get(cur.api)
                    if nxt is None:
                        failure = "forwards to '%s', which is neither an API nor a wrapper" % cur.api
                        break
                    cur, cur_args = nxt, expanded
                    chain.append(cur.name)
                else:
                    failure = "wrapper chain did not reach a registration API in 8 hops"
                if failure:
                    sc.unclaimed.append((fn, ln, failure, text[m.start():m.start() + 60]))
                    continue
                _record(sc, expanded, fn, ln, "%s(%s)" % (w.kind, "->".join(chain)), cur.api)

        # ---- residual: an UNQUALIFIED registration call the qualified scan cannot see
        for api in sc.apis:
            for m in re.finditer(r"(?<![:\w])%s\s*\(" % re.escape(api), text):
                before = text[max(0, m.start() - 12):m.start()]
                if "::" in before or "static" in before:
                    continue
                if in_wrapper_body(m.start()):
                    continue
                sc.unclaimed.append((fn, line_of(text, m.start()),
                                     "unqualified registration call", api))
    return sc


def _record(sc, args, fn, line, shape, api):
    """Turn a resolved argument list into a Registration, or into an unresolved/unclaimed row."""
    if len(args) < 3:
        sc.unclaimed.append((fn, line, "registration with %d args (expected >=3)" % len(args),
                             ", ".join(a.strip() for a in args)[:60]))
        return
    nid, how = resolve_nid(args[0])
    handler = resolve_handler(args[1])
    name = resolve_string(args[2])
    if nid is None:
        sc.unresolved.append((fn, line, shape, how))
        return
    sc.regs.append(Registration(nid, how, name, handler, fn, line, shape, api))


# ---------------------------------------------------------------- aggregation

def handler_index(regs):
    """handler -> {nid: Registration}. Distinct NIDs, so a NID registered twice to the SAME handler
    (e.g. `scePthreadAttrSetaffinity`, registered at hle_kernel.cpp:4539 and again at :4588) counts
    once — the question is how many Sony ENTRY POINTS collapse onto one answer, not how many source
    lines register them."""
    idx = {}
    for r in regs:
        idx.setdefault(r.handler, {})[r.nid] = r
    return idx


def nid_owner(regs):
    """nid -> the LAST registration in file/line order, i.e. last-write-wins within a file.

    `Hle::register_fn` is last-write-wins with no warning (dispatch.cpp:97), and the true runtime
    order is the order `register_builtin_hle()` calls the per-library registrars in — which a static
    read cannot know across translation units. Cross-file collisions with DIFFERENT handlers are
    therefore reported as ambiguous rather than resolved by guessing.
    """
    owner, ambiguous = {}, {}
    for r in sorted(regs, key=lambda r: (r.file, r.line)):
        prev = owner.get(r.nid)
        if prev is not None and prev.handler != r.handler:
            ambiguous.setdefault(r.nid, [prev]).append(r)
        owner[r.nid] = r
    return owner, ambiguous


def read_registry(path):
    """Parse an `hle_registry_dump` TSV into {nid: (kind, handler address, display name)}.

    The dump ends with a `# N registrations` line; N is checked against the rows actually read, so a
    truncated or half-written file is refused rather than reconciled against as if it were short.
    """
    rows, declared = {}, None
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                m = re.match(r"#\s*(\d+)\s+registrations", line)
                if m:
                    declared = int(m.group(1))
                continue
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                raise SystemExit("refused: malformed registry row: %r" % line[:80])
            rows[parts[0]] = (parts[1], parts[2], parts[3])
    if declared is None:
        raise SystemExit("refused: %s has no trailing `# N registrations` line — it may be "
                         "truncated, and a short table would reconcile as a real disagreement" % path)
    if declared != len(rows):
        raise SystemExit("refused: %s declares %d registrations but carries %d rows"
                         % (path, declared, len(rows)))
    return rows


def declared_table_nids(sc, src_hle):
    """NIDs living in the specific arrays the parser NAMED as unresolved — nothing wider.

    This is the budget for "runtime NIDs the parser is allowed not to have". It has to be tied to
    the exact tables the parser pointed at rather than to "appears as a literal somewhere in
    src/hle", because the looser rule passes even when a whole FILE is dropped from the parse —
    that file's NIDs are of course literals in it. (Measured: dropping `hle_font.cpp` leaves 75
    runtime-only NIDs, and the loose rule calls all 75 explained, which is precisely the silent
    shrink this reconciliation exists to catch.)

    So: read the array identifier out of each unresolved expression (`kUlt[kIdxInitialize].nid` ->
    `kUlt`), locate that array's initializer in the file the parser reported it from, and take the
    NID literals inside it. A runtime NID from anywhere else is unexplained, and unexplained means
    a registration shape the parser cannot see.
    """
    out, seen = set(), set()
    for fn, _line, _shape, expr in sc.unresolved:
        for arr in re.findall(r"\b([A-Za-z_]\w*)\s*\[", expr):
            if (fn, arr) in seen:
                continue
            seen.add((fn, arr))
            path = os.path.join(src_hle, fn)
            if not os.path.isfile(path):
                continue
            text = open(path, errors="ignore").read()
            m = re.search(r"\b%s\s*\[\s*\]\s*=\s*\{" % re.escape(arr), text)
            if not m:
                continue
            close = _matching_brace(text, m.end() - 1)
            if close is None:
                continue
            for lit in re.finditer(r'"([A-Za-z0-9+\-]{11})"', text[m.end() - 1:close + 1]):
                out.add(lit.group(1))
    return out


def reconcile(sc, registry, src_hle):
    """Compare the static parse against the registry the compiled binary built.

    Returns (hard_missing, declared_missing, extra, handler_groups_missed).

    `hard_missing` is the load-bearing one: a NID the binary registers that the parser never saw
    AND that does not live in one of the tables the parser openly reported as unresolved. That is,
    by definition, a registration SHAPE the parser cannot read — the failure this whole tool exists
    to make loud. `declared_missing` is the rest: an already-announced limit, not a coverage gap,
    so it must not be reported the same way or exit 3 fires on a healthy tree and stops meaning
    anything.

    The handler-address column is used only as an UPPER bound on collapse. Two distinct handlers
    with identical machine code can be folded to one address by the linker, so equal addresses do
    not prove a shared handler; unequal addresses do prove a distinct one. Everything the static
    parse calls shared must therefore also share an address at runtime, and that direction — the
    sound one — is what `handler_groups_missed` checks.
    """
    static_nids = {r.nid for r in sc.regs}
    missing = sorted(set(registry) - static_nids)
    declared = declared_table_nids(sc, src_hle)
    hard_missing = [n for n in missing if n not in declared]
    declared_missing = [n for n in missing if n in declared]
    extra = sorted(static_nids - set(registry))
    groups_missed = []
    for handler, d in handler_index(sc.regs).items():
        if len(d) < 2:
            continue
        addrs = {registry[n][1] for n in d if n in registry}
        if len(addrs) > 1:
            groups_missed.append((handler, sorted(d), sorted(addrs)))
    return hard_missing, declared_missing, extra, groups_missed


def read_gated(path):
    """NIDs from a `nid_gate_scan.py --all-nids` report (its first column), in file order."""
    out = []
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            tok = line.split()
            if tok and NID_RE.match(tok[0]):
                out.append(tok[0])
    return out


# ---------------------------------------------------------------- reporting

def print_coverage(sc, platform, names_count):
    shapes = {}
    for r in sc.regs:
        shapes[r.shape] = shapes.get(r.shape, 0) + 1
    macros = sorted({w.name for w in sc.wrappers if w.kind == "macro"})
    lambdas = sorted({w.name for w in sc.wrappers if w.kind == "lambda"})
    hows = {}
    for r in sc.regs:
        hows[r.nid_how] = hows.get(r.nid_how, 0) + 1
    print("# hle_handler_map — prosper HLE registration table, platform=%s" % platform)
    print("#")
    print("# COVERAGE — a zero in any table below is readable only because these are not zero")
    print("#   registration APIs (from dispatch.hpp class Hle): %s" % ", ".join(sc.apis))
    print("#   .cpp files scanned under src/hle:                %d" % len(sc.files))
    print("#   lines skipped as an inactive #if arm:            %d" % sc.skipped_lines)
    print("#   wrappers discovered:  %d macro (%s)" % (len(macros), " ".join(macros) or "-"))
    print("#                         %d lambda (%s)" % (len(lambdas), " ".join(lambdas) or "-"))
    print("#   registration sites claimed:                      %d  [%s]"
          % (len(sc.regs), " ".join("%s=%d" % kv for kv in sorted(shapes.items()))))
    print("#   NID literals resolved:                           %d  [%s]"
          % (len(sc.regs), " ".join("%s=%d" % kv for kv in sorted(hows.items()))))
    print("#   distinct NIDs registered:                        %d" % len({r.nid for r in sc.regs}))
    print("#   distinct handlers:                               %d" % len({r.handler for r in sc.regs}))
    print("#   Sony names known from --names:                   %s"
          % (names_count if names_count is not None else "(none: --names not given, labels only)"))
    print("#   wrapper-name calls outside its #define..#undef:  %d  (not registrations)"
          % sc.out_of_scope)
    print("#   sites claimed but NID not a literal:             %d  (listed below; not a coverage gap)"
          % len(sc.unresolved))
    print("#   sites UNCLAIMED (a missed shape):                %d%s"
          % (len(sc.unclaimed), "" if not sc.unclaimed else "   <<< TABLES BELOW ARE A LOWER BOUND"))
    print("#   #if conditions that could not be decided:        %d" % len(sc.uncertain_conds))
    for fn, ln, shape, expr in sc.unresolved:
        print("#     unresolved NID expr  %s:%d  %-22s %s" % (fn, ln, shape, expr[:70]))
    for fn, ln, why, txt in sc.unclaimed:
        print("#     UNCLAIMED            %s:%d  %s: %s" % (fn, ln, why, txt.strip()[:70]))
    for fn, ln, d in sc.uncertain_conds:
        print("#     undecided #if        %s:%d  %s" % (fn, ln, d))
    print("#")


def print_shared(sc, names, min_names, show_all):
    idx = handler_index(sc.regs)
    rows = sorted(((h, d) for h, d in idx.items() if show_all or len(d) >= min_names),
                  key=lambda kv: (-len(kv[1]), kv[0]))
    print("=== handlers serving %s Sony NID%s ==="
          % ("any number of" if show_all else ">= %d distinct" % min_names,
             "" if min_names == 1 else "s"))
    if not rows:
        print("   (none — every handler answers exactly one NID)")
    for handler, d in rows:
        print("%-34s serves %d" % (handler, len(d)))
        for nid, r in sorted(d.items(), key=lambda kv: kv[1].name or kv[0]):
            sony, lib = names.get(nid, ("", ""))
            print("    %-12s %-46s %-28s %s:%d"
                  % (nid, sony or r.name or "?", lib or "?", r.file, r.line))
    print()
    return idx


def print_cross(sc, idx, names, gated, gated_path, min_names):
    owner, ambiguous = nid_owner(sc.regs)
    shared, unregistered, single = [], [], []
    for nid in gated:
        r = owner.get(nid)
        if r is None:
            unregistered.append(nid)
        elif len(idx.get(r.handler, {})) >= min_names:
            shared.append((nid, r))
        else:
            single.append(nid)
    print("=== crossed with %s ===" % gated_path)
    print("   %d gated NIDs read from the report" % len(gated))
    print("   %d are registered in prosper; %d are NOT registered (no handler, no collapse)"
          % (len(gated) - len(unregistered), len(unregistered)))
    print("   %d gated rows sit on a handler serving >= %d distinct Sony names  <-- the collapse"
          % (len(shared), min_names))
    print("   %d gated rows have a handler of their own" % len(single))
    if ambiguous:
        print("   %d NIDs are registered to DIFFERENT handlers in >1 place (runtime order decides; "
              "see #330)" % len(ambiguous))
    print()

    byh = {}
    for nid, r in shared:
        byh.setdefault(r.handler, []).append(nid)
    print("   by handler:")
    for h, nids in sorted(byh.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print("      %-30s %3d gated  (serves %d distinct Sony names in total)"
              % (h, len(nids), len(idx[h])))
    print()

    # Per-library, never a rolled-up "libc"/"non-libc" subtotal: the one place this measurement has
    # been misread is exactly there, and the disagreement was a definition rather than a result.
    bylib = {}
    for nid, r in shared:
        _, lib = names.get(nid, ("", ""))
        bylib.setdefault(lib or "(not in --names)", []).append(nid)
    print("   by library (PS5 3.20 firmware file names, verbatim — `libkernel` is its OWN library")
    print("   and is NOT folded into `libSceLibcInternal`; no subtotal is rolled up here):")
    for lib, nids in sorted(bylib.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print("      %-34s %3d" % (lib, len(nids)))
    print("      %-34s %3d  (total)" % ("", len(shared)))
    print()

    print("   the rows themselves:")
    for nid, r in sorted(shared, key=lambda t: (names.get(t[0], ("", ""))[1],
                                                names.get(t[0], ("", ""))[0] or t[0])):
        sony, lib = names.get(nid, ("", ""))
        print("      %-12s %-46s %-28s -> %-24s %s:%d"
              % (nid, sony or r.name or "?", lib or "?", r.handler, r.file, r.line))
    return shared


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--src", default=os.path.join(os.path.dirname(here), "..", "src", "hle"),
                    help="prosper's src/hle directory (default: relative to this script)")
    ap.add_argument("--platform", choices=sorted(PLATFORM_DEFINES),
                    default=("windows" if sys.platform.startswith("win")
                             else "macos" if sys.platform == "darwin" else "linux"),
                    help="which #if arm to evaluate (default: this host)")
    ap.add_argument("--names", metavar="DIR", help="PS5 firmware genstub dump, for names+libraries")
    ap.add_argument("--gated", metavar="FILE", help="a `nid_gate_scan.py --all-nids` report to cross")
    ap.add_argument("--registry", metavar="FILE",
                    help="an `hle_registry_dump` TSV; reconciles the parse against the binary")
    ap.add_argument("--all", action="store_true", help="list every handler, not only shared ones")
    ap.add_argument("--min-names", type=int, default=2,
                    help="a handler is 'shared' at this many distinct Sony names (default 2)")
    a = ap.parse_args()

    src = os.path.normpath(a.src)
    if not os.path.isdir(src):
        print("refused: --src %s is not a directory" % src, file=sys.stderr)
        return 2
    names = G.load_nid_names(a.names) if a.names else {}

    sc = scan_tree(src, a.platform)
    if not sc.regs:
        print("refused: parsed 0 registrations from %s — nothing below would be a result" % src,
              file=sys.stderr)
        print_coverage(sc, a.platform, len(names) if a.names else None)
        return 2

    print_coverage(sc, a.platform, len(names) if a.names else None)

    reconcile_bad = False
    if a.registry:
        registry = read_registry(a.registry)
        hard, declared, extra, groups = reconcile(sc, registry, src)
        print("=== reconciled against the registry the binary builds (%s) ===" % a.registry)
        print("   %d NIDs registered at runtime; %d recovered by this parser"
              % (len(registry), len({r.nid for r in sc.regs})))
        print("   %d registered at runtime but NOT found by the parser  <-- a MISSED SHAPE"
              % len(hard))
        for n in hard[:40]:
            print("      missing %s" % n)
        print("   %d more are literals in the array-driven tables the parser lists as unresolved "
              "(a declared limit, not a coverage gap)" % len(declared))
        print("   %d found by the parser but not registered at runtime (an inactive arm, or a "
              "registrar this build does not call)" % len(extra))
        for n in extra[:40]:
            print("      extra   %s" % n)
        print("   %d shared-handler groups the runtime addresses CONTRADICT" % len(groups))
        for h, nids, addrs in groups[:20]:
            print("      %s: %s across %d addresses" % (h, " ".join(nids), len(addrs)))
        print()
        reconcile_bad = bool(hard or groups)

    idx = print_shared(sc, names, a.min_names, a.all)
    if a.gated:
        print_cross(sc, idx, names, read_gated(a.gated), a.gated, a.min_names)

    # A missed shape makes every table a lower bound, so it must not exit 0 — a saved transcript
    # has to carry its own validity, the same contract xref.py adopted after #2399.
    return 3 if (sc.unclaimed or reconcile_bad) else 0


if __name__ == "__main__":
    sys.exit(main())
