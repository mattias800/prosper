#!/usr/bin/env python3
"""Refuse a non-ASCII character inside a C/C++ STRING LITERAL.

stdout and stderr carry BYTES. A UTF-8 punctuation mark in a printf literal leaves the process as
its UTF-8 bytes and is decoded by whoever reads them -- UTF-8 on Linux, but the console/locale code
page (cp1252 / cp437) on Windows. One EM DASH (U+2014 = e2 80 94) therefore reaches a Windows user as

    cp1252: [IMPORT SLOTS] 0 import slots a=" the dynamic table declares neither DT_JMPREL
    cp437 : [IMPORT SLOTS] 0 import slots .-o the dynamic table declares neither DT_JMPREL

so a `grep` for the line prosper itself printed does not match it, and a test asserting on the
rendered string fails on Windows only. That is #2588, found because #2579 shipped five em dashes in
`self_dump --import-slots` output and the Windows MinGW job failed on the one assertion that spanned
one of them while Linux CI stayed green.

WHAT THIS CHECKS, and why it is the whole literal rather than "output literals"

Deciding whether a literal reaches stdout needs dataflow, and every cheap approximation of it (the
callee is named printf, the line contains "fprintf") misses the case that actually shipped: a table
of message fragments concatenated three frames away from the call. So the rule here is the blunt
one -- NO string literal in a scanned source file may contain a non-ASCII character -- and it is
defensible beyond output, because a UTF-8 byte in a literal is also compiler-dependent (MSVC's
source charset), diff-hostile, and unnecessary in every case in this tree.

COMMENTS AND MARKDOWN KEEP THEIRS. The rule is about bytes the program emits, not source style, and
sweeping comments would be a huge diff with no benefit. Comments are stripped before literals are
extracted, so an em dash in prose is invisible to this gate by construction.

Escapes are read too, because they put the same bytes on the wire without a single non-ASCII byte
in the source -- a scanner that looked only at raw bytes could be walked straight around. They split
by how they are written: a \\u / \\U escape above 127 FAILS (it denotes a character, and is how the
accident would be spelled to dodge a byte scan), while a \\x / octal escape above 127 is a NOTE, on
the grounds that nobody types "\\xe2\\x80\\x94" by accident -- the two in this tree are UTF-8
fixtures feeding conversion tests, and failing them would mean banning binary test data.

Run standalone against a checkout, or via ctest as ascii_output_literals.
"""
import re
import sys
from pathlib import Path

SCAN_DIRS = ("src", "frontends", "tools", "tests")
SCAN_EXT = (".c", ".cc", ".cpp", ".h", ".hpp", ".inl")

# Escapes that denote a code point without spelling it in bytes. \u/\U are always a code point;
# \x and octal are byte values, and a byte above 0x7F in a narrow literal is exactly the thing
# being banned (it is how you would write the em dash by hand: "\xe2\x80\x94").
ESCAPE_RE = re.compile(r'\\(?:u([0-9a-fA-F]{4})|U([0-9a-fA-F]{8})|x([0-9a-fA-F]+)|([0-7]{1,3}))')


class Literal:
    __slots__ = ("text", "line", "raw", "at")

    def __init__(self, text, line, raw, at):
        self.text = text      # literal body, escapes NOT expanded
        self.line = line      # 1-based line of the opening quote
        self.raw = raw        # True for R"(...)" -- backslash is not an escape there
        self.at = at          # parallel to text: the 1-based source line of each character

    def line_of(self, idx):
        """The source line of text[idx]. A raw string spans lines, so the opening line is not it."""
        return self.at[idx] if 0 <= idx < len(self.at) else self.line


def extract_literals(src: str):
    """Every string literal in `src`, with comments and char literals removed.

    A hand-rolled scanner rather than a regex because the three things that must not be confused --
    a quote inside a comment, a comment marker inside a string, and a backslash inside a raw string
    -- are exactly what a regex gets wrong, and getting one wrong makes this gate report a false
    CLEAN rather than a false alarm.
    """
    out = []
    i, n, line = 0, len(src), 1
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        # comments
        if c == "/" and i + 1 < n:
            if src[i + 1] == "/":
                while i < n and src[i] != "\n":
                    # a backslash-newline continues a line comment onto the next line
                    if src[i] == "\\" and i + 1 < n and src[i + 1] == "\n":
                        line += 1
                        i += 2
                        continue
                    i += 1
                continue
            if src[i + 1] == "*":
                i += 2
                while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
                    if src[i] == "\n":
                        line += 1
                    i += 1
                i = min(i + 2, n)
                continue
        # char literal: skip it, so '"' does not open a string
        if c == "'":
            j = i + 1
            closed = False
            while j < n and src[j] != "\n":
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "'":
                    closed = True
                    break
                j += 1
            # Not closed on this line, so the quote was a digit separator (1'000) or stray prose,
            # NOT a char literal. Consume just the quote. Skipping to the next ' across newlines is
            # what drifted every line number after it -- a finding then names an innocent line and
            # the real one is never looked at, which is worse than missing it.
            i = (j + 1) if closed else (i + 1)
            continue
        # raw string: R"delim( ... )delim", optionally prefixed u8/u/U/L
        if c == "R" and i + 1 < n and src[i + 1] == '"':
            k = i + 2
            delim = ""
            while k < n and src[k] != "(":
                delim += src[k]
                k += 1
            close = ")" + delim + '"'
            end = src.find(close, k)
            if end < 0:
                end = n
            body = src[k + 1:end]
            at, cur = [], line
            for ch in body:
                at.append(cur)
                if ch == "\n":
                    cur += 1
            out.append(Literal(body, line, True, at))
            line = cur
            i = end + len(close)
            continue
        if c == '"':
            j = i + 1
            body, at, cur = [], [], line
            while j < n:
                if src[j] == "\\" and j + 1 < n:
                    body.append(src[j:j + 2])
                    at.extend([cur, cur])
                    if src[j + 1] == "\n":       # backslash-newline continues the literal
                        cur += 1
                    j += 2
                    continue
                if src[j] == '"':
                    break
                if src[j] == "\n":       # unterminated: leave the newline for the outer loop
                    break
                body.append(src[j])
                at.append(cur)
                j += 1
            out.append(Literal("".join(body), line, False, at))
            line = cur
            i = j + 1 if j < n and src[j] == '"' else j
            continue
        i += 1
    return out


def offenders_in(lit: Literal):
    """[(kind, detail, line)] for each non-ASCII thing in one literal. Empty when clean.

    kind is one of:
      "char"    a non-ASCII character, spelled as itself. The accident class: prose punctuation
                typed into a message. Always a failure.
      "unicode" a \\u / \\U escape denoting a code point above 127. The same accident written so a
                byte-level scanner cannot see it. Nothing in this tree needs one, so also a failure.
      "byte"    a \\x or octal escape above 127. BINARY DATA by construction -- nobody types
                "\\xe2\\x80\\x94" by accident, and the two in this tree are UTF-8 fixtures feeding
                conversion tests, not messages. Reported as a note, never failed; see main().
    """
    bad = []
    for idx, ch in enumerate(lit.text):
        if ord(ch) > 127:
            bad.append(("char", "U+%04X %r" % (ord(ch), ch), lit.line_of(idx), ch))
    if not lit.raw:
        for m in ESCAPE_RE.finditer(lit.text):
            u4, u8, hx, oc = m.groups()
            if u4 is not None:
                val, kind = int(u4, 16), "unicode"
            elif u8 is not None:
                val, kind = int(u8, 16), "unicode"
            elif hx is not None:
                val, kind = int(hx, 16), "byte"
            else:
                val, kind = int(oc, 8), "byte"
            if val > 127:
                bad.append((kind, "%s = U+%04X" % (m.group(0), val), lit.line_of(m.start()),
                            m.group(0)))
    # de-duplicate while keeping order, so one repeated dash reports once per literal
    seen, uniq = set(), []
    for kind, detail, line, needle in bad:
        if (detail, line) not in seen:
            seen.add((detail, line))
            uniq.append((kind, detail, line, needle))
    return uniq


def scan(root: Path):
    """Return (findings, files_scanned, literals_examined, misreported).

    findings:    list of (relpath, line, kind, detail, excerpt)
    misreported: findings whose named line does NOT contain the character they claim -- an internal
                 inconsistency in this scanner, surfaced rather than swallowed. The first version
                 drifted its line counter across an unterminated quote, so a finding pointed at an
                 innocent line while the real one went unexamined; a scanner that cannot detect that
                 about itself sends the reader to the wrong place with full confidence.
    """
    findings = []
    misreported = []
    files = 0
    literals = 0
    for d in SCAN_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in SCAN_EXT or not path.is_file():
                continue
            # Generated and vendored trees: a fetched dependency's literals are not ours to police,
            # and CMake puts build dirs inside the source root here (build-linux, build-windows).
            # DIRECTORY components only -- matching the filename too silently dropped the real
            # source file src/build_revision.hpp, which is how an exclusion quietly narrows a gate's
            # domain while its output still reports a confident, unchanged-looking clean.
            dirs = path.parts[:-1]
            if ("_deps" in dirs or "third_party" in dirs
                    or any(p.startswith("build") for p in dirs)):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            files += 1
            rel = path.relative_to(root).as_posix()
            src_lines = text.split("\n")
            for lit in extract_literals(text):
                literals += 1
                for kind, detail, line, needle in offenders_in(lit):
                    f = (rel, line, kind, detail, lit.text[:80])
                    findings.append(f)
                    src = src_lines[line - 1] if 0 < line <= len(src_lines) else ""
                    if needle not in src:
                        misreported.append(f)
    return findings, files, literals, misreported


# --- self-test ----------------------------------------------------------------------------------
# Without these, one broken pattern makes this gate report a clean tree forever -- the single way it
# can fail silently. Half the cases assert on shapes that must NOT be reported, because a gate that
# fires on valid code gets deleted rather than heeded.
DASH = "\u2014"
SELF_TESTS = [
    # (source snippet, exact expected number of findings)
    ('printf("plain ascii\\n");', 0),
    ('printf("em %s dash\\n");' % DASH, 1),
    ('fprintf(stderr, "a %s b %s c");' % (DASH, DASH), 1),       # one literal, de-duplicated
    ('printf("x" "%s" "y");' % DASH, 1),                          # concatenation: the middle piece
    # MUST NOT fire: comments keep their punctuation, and that is the whole point of stripping them.
    ('// an em dash %s in prose' % DASH, 0),
    ('/* block %s comment */' % DASH, 0),
    ('/* %s */ printf("clean");' % DASH, 0),
    ('printf("clean"); // trailing %s' % DASH, 0),
    # A quote inside a comment must not open a literal, or everything after it is misparsed.
    ('// he said "hello %s\nprintf("clean");' % DASH, 0),
    # A comment marker inside a literal must not start a comment.
    ('printf("// not a comment %s");' % DASH, 1),
    ('printf("/* not a comment */");', 0),
    # Escaped quote must not end the literal early.
    ('printf("say \\"%s\\" now");' % DASH, 1),
    # Char literals must not open a string.
    ("c = '\"'; printf(\"clean\");", 0),
    ("c = '\\\\'; printf(\"%s\");" % DASH, 1),
    # Raw strings: body is verbatim, and a backslash there is not an escape.
    ('const char* s = R"(raw %s here)";' % DASH, 1),
    ('const char* s = R"(C:\\path\\u2014not-an-escape)";', 0),
    ('const char* s = R"delim(a "quote" %s)delim";' % DASH, 1),
    # Escapes that DENOTE a non-ASCII code point without spelling it in bytes.
    ('printf("\\u2014");', 1),
    ('printf("\\U0001F600");', 1),
    ('printf("\\xe2\\x80\\x94");', 3),      # three escapes, all > 127, one finding each
    ('printf("\\342\\200\\224");', 3),
    # MUST NOT fire: ASCII escapes are ordinary.
    ('printf("tab\\there\\n");', 0),
    ('printf("\\x41\\x42");', 0),
    ('printf("\\101");', 0),
    # #include and preprocessor lines are ordinary text to the scanner; must stay clean.
    ('#include "header.h"', 0),
]

# The LINE number a finding carries is the whole value of the report -- a finding at the wrong line
# sends the reader to innocent code and the real one is never opened. The first version of this
# scanner drifted, because an apostrophe with no closing quote on its line (a digit separator, or
# prose inside a string) made it skip forward across newlines it never counted. Every case below
# puts the offender on a line AFTER something that could drift, so a regression shows up as a wrong
# number rather than as a missing finding.
LINE_TESTS = [
    # (source, expected [(line, count)])
    ('printf("clean");\nprintf("%s");' % DASH, [(2, 1)]),
    ('int x = 1\'000\'000;\nprintf("%s");' % DASH, [(2, 1)]),           # digit separator
    ('printf("it\'s fine");\nprintf("%s");' % DASH, [(2, 1)]),          # apostrophe inside a string
    ('// don\'t\nprintf("%s");' % DASH, [(2, 1)]),                      # apostrophe inside a comment
    ('/* multi\n   line\n   comment */\nprintf("%s");' % DASH, [(4, 1)]),
    ('printf("a"\n       "%s");' % DASH, [(2, 1)]),                     # continued across lines
    ('printf("a\\\n%s");' % DASH, [(2, 1)]),                            # backslash-newline
    ('const char* s = R"(one\ntwo %s)";\nprintf("clean");' % DASH, [(2, 1)]),   # inside a raw string
    ('#define X "a" \\\n          "%s"\nprintf("clean");' % DASH, [(2, 1)]),
]


def self_test() -> int:
    bad = 0
    for snippet, want in SELF_TESTS:
        got = sum(len(offenders_in(l)) for l in extract_literals(snippet))
        if got != want:
            print("  [FAIL] self-test: %r -> %d finding(s), want %d" % (snippet, got, want))
            bad += 1
    for snippet, want in LINE_TESTS:
        got = {}
        for lit in extract_literals(snippet):
            for _kind, _detail, line, _needle in offenders_in(lit):
                got[line] = got.get(line, 0) + 1
        if sorted(got.items()) != want:
            print("  [FAIL] line self-test: %r -> %s, want %s"
                  % (snippet, sorted(got.items()), want))
            bad += 1
    if bad:
        print("  the scanner's own patterns are broken -- a tree scan would report a false CLEAN")
        print("  or point at the wrong line, and either way the gate stops being worth reading")
    else:
        print("  [ok]   scanner self-test: %d shape case(s), %d line-number case(s)"
              % (len(SELF_TESTS), len(LINE_TESTS)))
    return bad


# --- quarantine ---------------------------------------------------------------------------------
# Files whose non-ASCII literals were NOT swept by #2588, with the count that was there when the
# gate landed. Every one is owned by the live GPU/compute lane, where a one-character edit on 42
# lines is a conflict magnet in files being rewritten; #2608 tracks the sweep.
#
# The entries are a LEDGER, not an exemption list, and the difference is mechanical:
#   * more findings than recorded -> FAIL. The lane may not add new ones.
#   * exactly the recorded count  -> note. Unchanged, and visible in every run's output.
#   * FEWER, including zero       -> FAIL, asking for the number to be lowered or the row deleted.
# The last rule is what stops this from rotting into a permanent allowlist: an entry that has been
# fixed cannot stay, because the gate refuses to be silently over-broad. A stale allowlist is how a
# gate keeps passing a tree it no longer describes.
QUARANTINE = {
    "src/gpu/command_processor.cpp": 17,
    "src/gpu/gpu_capture_bundle.cpp": 1,
    "src/gpu/gpu_execute.hpp": 3,
    "src/gpu/gpu_executor.cpp": 4,
    "src/gpu/gpu_timeline.cpp": 4,
    "src/gpu/mb3_freelist.cpp": 4,
    "src/gpu/render_state.cpp": 1,
    "frontends/shared/live_compute.cpp": 2,
    "frontends/shared/live_renderer.cpp": 6,
    "tests/render_runner.h": 2,
    "tests/test_descriptor_array_emit.cpp": 1,
    "tests/test_descriptor_array_render.cpp": 1,
    "tests/test_rdna2_to_spirv.cpp": 2,
}


def main() -> int:
    here = Path(__file__).resolve()
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else here.parents[2]
    print("== check_ascii_output ==")
    if self_test():
        return 1

    findings, files, literals, misreported = scan(root)

    # Report the SIZE of what was examined every time, pass or fail. A gate that prints only a
    # verdict cannot be distinguished from a gate that scanned nothing -- a moved directory, a
    # renamed extension, a wrong root argument all yield a silent, permanent CLEAN otherwise.
    print("  [ok]   scanned %d file(s) under %s/{%s}: %d string literal(s) examined"
          % (files, root.name, ",".join(SCAN_DIRS), literals))
    if files == 0 or literals == 0:
        print("  [FAIL] nothing was scanned -- the scan is not seeing the tree, so a CLEAN here")
        print("         would mean nothing. Check the root argument and SCAN_DIRS/SCAN_EXT.")
        return 1

    if misreported:
        print("  [FAIL] %d finding(s) name a line that does not contain what they report:"
              % len(misreported))
        for rel, line, kind, detail, _ in misreported[:5]:
            print("    %s:%d: %s %s" % (rel, line, kind, detail))
        print("  This scanner's line counter has drifted. Fix it before reading any result below --")
        print("  the numbers point at innocent code and the real sites go unexamined.")
        return 1

    quarantined = {}
    fails, notes = [], []
    for f in findings:
        rel, _line, kind, _detail, _ex = f
        if rel in QUARANTINE:
            quarantined[rel] = quarantined.get(rel, 0) + 1
        elif kind == "byte":
            notes.append(f)
        else:
            fails.append(f)

    for rel, _line, _kind, detail, excerpt in notes:
        print("  [note] %s: %s -- a byte escape, i.e. binary data, not a typed character" % (rel, detail))
        print("           in literal: %r" % excerpt)

    ledger_bad = []
    for rel, want in sorted(QUARANTINE.items()):
        got = quarantined.get(rel, 0)
        if got == want:
            print("  [note] %s: %d known non-ASCII literal(s), unswept (see #2608)" % (rel, got))
        elif got > want:
            ledger_bad.append("%s: %d non-ASCII literal(s), %d recorded -- %d ADDED since #2588"
                              % (rel, got, want, got - want))
        else:
            ledger_bad.append("%s: %d non-ASCII literal(s), %d recorded -- lower the number%s"
                              % (rel, got, want, " or delete the row" if got == 0 else ""))

    if not fails and not ledger_bad:
        print("  [ok]   every string literal outside the quarantine is ASCII "
              "(%d note(s), %d quarantined file(s))" % (len(notes), len(QUARANTINE)))
        print("== all checks passed ==")
        return 0

    if fails:
        print("  [FAIL] %d non-ASCII character(s) in string literals:" % len(fails))
        for rel, line, kind, detail, excerpt in fails:
            print("    %s:%d: %s %s" % (rel, line, kind, detail))
            print("        in literal: %r" % excerpt)
        print("  stdout/stderr carry bytes; Windows decodes them as cp1252/cp437, so these render")
        print("  as mojibake and break a grep for the line the program printed. Use ASCII: -- for")
        print("  an em dash, -> for an arrow, ... for an ellipsis, \" for smart quotes, x for a")
        print("  multiplication sign. Comments and Markdown keep theirs; this is about the bytes")
        print("  the program emits, not about source style.")
    if ledger_bad:
        print("  [FAIL] %d quarantine ledger entr(y/ies) no longer match the tree:" % len(ledger_bad))
        for msg in ledger_bad:
            print("    " + msg)
        print("  The ledger at the bottom of this file records what #2588 deliberately left in the")
        print("  GPU/compute lane's files. It is not an exemption: it may only shrink.")
    print("== %d failure(s) ==" % (len(fails) + len(ledger_bad)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
