#!/usr/bin/env python3
"""Refuse a non-ASCII character inside a C/C++ STRING LITERAL.

stdout and stderr carry BYTES. A UTF-8 punctuation mark in a printf literal leaves the process as
its UTF-8 bytes and is decoded by whoever reads them -- UTF-8 on Linux, but the console/locale code
page (cp1252 / cp437) on Windows. One EM DASH (U+2014 = e2 80 94) therefore reaches a Windows user
as three mojibake characters -- a-circumflex, euro, right-double-quote under cp1252; capital-gamma,
C-cedilla, o-umlaut under cp437 -- in place of the dash, so

    [IMPORT SLOTS] 0 import slots <3 junk chars> the dynamic table declares neither DT_JMPREL

is what the console shows. A `grep` for the line prosper itself printed does not match it, and a test asserting on the
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
the grounds that a MULTI-BYTE run such as "\\xe2\\x80\\x94" is nobody's accident -- the two in this
tree are UTF-8 fixtures feeding conversion tests, and failing them would mean banning binary test
data. That justification is honestly weaker for a LONE high byte: printf("caf\\xe9") is the same
mojibake defect written in escape form, and this reports it only as a note. The note COUNT is
printed on every run so growth is visible; discriminating a stray latin-1 byte from deliberate
binary needs a UTF-8 well-formedness test that is not written here.

WHAT THIS CANNOT SEE, stated so its silence is not read as coverage:

  * Python, shell and CMake literals -- the same defect class, measured and filed as #2609.
  * Markdown, and comments in any language. Deliberate.
  * Anything assembled at RUNTIME: a data file, argv, or a title's own UTF-8 name through %s.
  * CHAR literals. 'x' spans are skipped so a quote inside one cannot open a string, and their
    contents are never examined -- L'<em dash>' would be invisible. Zero instances today.
  * C++23 delimited escapes: "\\u{2014}", "\\x{2014}", "\\o{24024}", "\\N{EM DASH}". The project is
    C++20, but GCC and Clang accept some of these as extensions already.
  * The files in QUARANTINE, bounded by that ledger rather than unexamined.
  * Whether ASCII output is CORRECT. This bans a class of bytes, nothing more.

Run standalone against a checkout, or via ctest as ascii_output_literals.
"""
import re
import sys
from pathlib import Path

SCAN_DIRS = ("src", "frontends", "tools", "tests")
# .m/.mm are here for a macOS frontend that does not exist yet: adding an extension costs nothing,
# while a future Objective-C file would otherwise be silently out of scope, and silent scope loss is
# this checker's whole failure mode.
SCAN_EXT = (".c", ".cc", ".cpp", ".h", ".hpp", ".inl", ".m", ".mm")

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


CHAR_PREFIXES = ("", "L", "u", "U", "u8")


def opens_char_literal(src: str, i: int) -> bool:
    """Does the quote at src[i] open a char literal, or is it a digit separator / stray prose?

    Decided by the maximal run of identifier characters immediately before it, which is the whole
    of the ambiguity:

        '  x = 'a'   -> run ""    -> yes        '  L'"'      -> run "L"    -> yes (prefix)
        '  5'000     -> run "5"   -> no         '  u8'"'     -> run "u8"   -> yes (prefix)
        '  menu'0    -> run "menu"-> no         '  kL'0      -> run "kL"   -> no

    Both directions have bitten. Treating every `'` as an opener let `5'000` pair with an apostrophe
    inside a later string, swallowing that string's opening quote and hiding its contents -- a false
    clean. Rejecting every `'` preceded by an alphanumeric then broke the ENCODING PREFIXES, since
    L/u/U/u8 are alphanumeric too, so `L'"'` stopped being a char literal and its inner quote opened
    a string instead -- the same false clean, one case narrower, and against a docstring that
    asserts the opposite. Both were found by review, the second in the fix for the first.
    """
    j = i
    while j > 0 and (src[j - 1].isalnum() or src[j - 1] == "_"):
        j -= 1
    return src[j:i] in CHAR_PREFIXES


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
        # char literal: skip it, so '"' does not open a string.
        #
        # A DIGIT SEPARATOR is the hazard, and it fails in the dangerous direction. `5'000` is not a
        # char literal, but if a string later on the same line contains an apostrophe -- `puts("it's
        # here")` -- a scanner that just hunts for the next `'` pairs the separator with that one,
        # swallows the string's opening quote, and every character of that string becomes INVISIBLE
        # to the gate. A false clean, which is the one outcome this checker must never produce.
        # So a quote only opens a char literal when the character before it cannot end an
        # identifier or a number, which is exactly what distinguishes `5'000` and `x'` from `= '` or
        # `('`. Reported by review; no instance existed in the tree, and the tree is not the point.
        if c == "'" and opens_char_literal(src, i):
            j = i + 1
            closed = False
            while j < n and src[j] != "\n":
                if src[j] == "\\":
                    # A char literal does not span lines, so a backslash-newline here means this
                    # quote was never one. Stepping over that newline uncounted drifts every line
                    # number after it (measured: a one-line drift), which points findings at
                    # innocent code.
                    if j + 1 < n and src[j + 1] == "\n":
                        break
                    j += 2
                    continue
                if src[j] == "'":
                    closed = True
                    break
                j += 1
            # Not closed on this line: stray prose or an unbalanced quote, not a char literal.
            # Consume just the quote, so the outer loop counts the newline.
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
            # Backslash PARITY: in "\\u2014" the first backslash escapes the second, so the runtime
            # string is six ASCII characters and nothing non-ASCII is emitted. Counting
            # the run of backslashes before the match is what separates that from a real escape.
            # The tree has three live sites (all JSON escapers) that are safe only because none is
            # followed by four hex digits. Reported by review; false-positive direction, but a gate
            # that cries wolf on correct code gets deleted rather than heeded.
            k = m.start() - 1
            slashes = 0
            while k >= 0 and lit.text[k] == "\\":
                slashes += 1
                k -= 1
            if slashes % 2:                 # this backslash is itself escaped
                continue
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
    # MUST NOT fire: an ESCAPED backslash. "\\u2014" emits six ASCII characters, and the tree has
    # three live JSON escapers written this way -- safe today only because none is followed by four
    # hex digits, which is not a property to rely on.
    ('printf("\\\\u2014");', 0),
    ('printf("\\\\\\\\u2014");', 0),          # four backslashes: still escaped pairs
    # MUST fire: three backslashes is an escaped backslash THEN a real \\u escape.
    ('printf("\\\\\\u2014");', 1),
    # MUST fire: a digit separator must not swallow a later string. If the ' in 1'000 pairs with the
    # one in "it's", the whole of that string -- and this dash -- goes invisible. A FALSE CLEAN.
    ('int n = 1\'000; puts("it\'s %s");' % DASH, 1),
    # MUST fire: the four ENCODING PREFIXES still open a char literal. The first version of the
    # guard above rejected them (L/u/U/u8 are alphanumeric), so the quote inside L'"' opened a
    # STRING and everything after it on the line went invisible -- the same false clean the guard
    # was added to remove, one case narrower, and against a docstring asserting the opposite.
    ('wchar_t c = L\'"\'; puts("%s");' % DASH, 1),
    ('char8_t c = u8\'"\'; puts("%s");' % DASH, 1),
    ('char16_t c = u\'"\'; puts("%s");' % DASH, 1),
    ('char32_t c = U\'"\'; puts("%s");' % DASH, 1),
    ('char c = \'"\'; puts("%s");' % DASH, 1),           # control: unprefixed, always worked
    # MUST fire: an identifier that merely ENDS in a prefix letter is not a prefix, so these stay
    # digit-separator-like and must not open a char literal and swallow the string.
    ('int kL = 0; int n = kL\'0; puts("it\'s %s");' % DASH, 1),
    ('int menu = 0; int n = menu\'0; puts("it\'s %s");' % DASH, 1),
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
    # The three below pin line tracking where the arms above did NOT, each verified to fail on a
    # one-line revert of the logic it covers. Review measured that reverting the apostrophe guard
    # left all 34 previous arms green AND a full tree scan green, while drifting every finding on
    # unswept master by up to 114 lines -- and now that the tree is swept, the runtime misreported
    # cross-check cannot catch it either, because it needs a finding to check. The tree that
    # discovered a bug stops being able to detect its return; that is when a self-test has to carry
    # the weight.
    #   1. an UNCLOSED code-level apostrophe before a later string holding an apostrophe. Without
    #      the `!= "\n"` guard the scan runs forward to that second apostrophe across a newline it
    #      never counts, and every later finding is one line early. NOTE: the digit-separator form
    #      of this arm, which review supplied, is inert against this particular revert -- the
    #      opener guard added alongside it means `1'000` is not treated as a char literal at all,
    #      so the newline guard is never reached. Two guards, two arms; an arm that the OTHER fix
    #      makes unreachable pins nothing, and this one was measured surviving before it was
    #      rewritten to the form below.
    ('x = \' ;\nputs("it\'s fine");\nputs("%s");' % DASH, [(3, 1)]),
    #   2. `line = cur` after a NORMAL string continued with backslash-newline
    ('puts("a\\\n b\\\n c");\nputs("x");\nputs("%s");' % DASH, [(5, 1)]),
    #   3. `line = cur` after a RAW string spanning lines
    ('const char* s = R"(one\ntwo\nthree)";\nputs("x");\nputs("%s");' % DASH, [(5, 1)]),
    #   4. a char literal continued with backslash-newline is NOT a char literal (found in author
    #      testing; it drifted by one line and the misreported check turned it into a hard abort)
    ('char c = \'\\\n n\';\nputs("%s");' % DASH, [(3, 1)]),
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
    for rel, kind, want_bucket in CLASSIFY_TESTS:
        got_bucket = classify(rel, kind)
        if got_bucket != want_bucket:
            print("  [FAIL] classify self-test: (%s, %s) -> %s, want %s"
                  % (rel, kind, got_bucket, want_bucket))
            bad += 1
    if bad:
        print("  the scanner's own patterns are broken -- a tree scan would report a false CLEAN")
        print("  or point at the wrong line, and either way the gate stops being worth reading")
    else:
        print("  [ok]   scanner self-test: %d shape case(s), %d line-number case(s), "
              "%d bucket case(s)" % (len(SELF_TESTS), len(LINE_TESTS), len(CLASSIFY_TESTS)))
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
    "src/gpu/pm4/command_processor.cpp": 17,
    "src/gpu/capture/gpu_capture_bundle.cpp": 1,
    "src/gpu/execute/gpu_execute.hpp": 3,
    "src/gpu/execute/gpu_executor.cpp": 4,
    "src/gpu/timeline/gpu_timeline.cpp": 4,
    "src/gpu/execute/mb3_freelist.cpp": 4,
    "src/gpu/state/render_state.cpp": 1,
    "frontends/shared/live_compute.cpp": 2,
    "frontends/shared/live_renderer.cpp": 6,
    "tests/render_runner.h": 2,
    "tests/test_descriptor_array_emit.cpp": 1,
    "tests/test_descriptor_array_render.cpp": 1,
    "tests/test_rdna2_to_spirv.cpp": 2,
}


def classify(rel: str, kind: str) -> str:
    """Which bucket one finding lands in: "note", "ledger" or "fail".

    KIND is decided before FILE, and the order is the whole content of this function. The other way
    round, a \\x byte escape inside a quarantined file counted into the ledger -- so the identical
    "\\xe2\\x80\\x94" fixture was a silent note anywhere else and a CI failure in a quarantined one,
    reported as "3 ADDED since #2588": the three-tier rule suspended exactly where it would
    misdiagnose legitimate binary test data as the defect class. Two of the thirteen quarantined
    entries are test files, which is where a hex fixture lands.

    It lives out here, rather than inline in main(), only so CLASSIFY_TESTS can pin that order --
    review's point that this was the one fix in the PR whose removal nothing would detect. The tree
    cannot catch it (no quarantined file holds a byte escape) and the literal-level arms never reach
    main(), so without these two assertions it is verified by hand exactly once, in a session that
    ends.
    """
    if kind == "byte":
        return "note"
    if rel in QUARANTINE:
        return "ledger"
    return "fail"


# The full 3x2 truth table: every kind offenders_in() can produce, in and out of the quarantine.
#
# It is exhaustive on purpose, and the reason is worth the four extra lines. The first version had
# only the four `char`/`byte` rows and pinned the ORDER without pinning the FUNCTION: `unicode`
# never appeared, so `if kind in ("byte", "unicode"): return "note"` passed every arm while routing
# the \u tier to a silent note -- a false clean on exactly the tier that exists to stop someone
# spelling an em dash as an escape to walk around a byte scan. Review measured that mutation
# leaving ALL 53 arms green. A set of arms drawn as a sample of a small finite domain is a sample;
# enumerated, it is a proof.
CLASSIFY_TESTS = [
    # (relpath, kind, expected bucket)
    ("src/gpu/state/render_state.cpp", "byte",    "note"),    # quarantined AND a byte escape -> a note
    ("src/hle/other.cpp",        "byte",    "note"),    # ... identical outcome anywhere else
    ("src/gpu/state/render_state.cpp", "char",    "ledger"),  # quarantined, a real character -> ledger
    ("src/hle/other.cpp",        "char",    "fail"),    # the ordinary case
    ("src/hle/other.cpp",        "unicode", "fail"),    # the \u tier is a failure, not a note
    ("src/gpu/state/render_state.cpp", "unicode", "ledger"),  # ... and inside the quarantine, the ledger
]


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
        bucket = classify(rel, kind)
        if bucket == "note":
            notes.append(f)
        elif bucket == "ledger":
            quarantined[rel] = quarantined.get(rel, 0) + 1
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
        print("  If you MOVED one of these out of a quarantined file rather than writing it, the")
        print("  ledger below expects the old file's count to drop -- sweep it while it is in hand.")
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
