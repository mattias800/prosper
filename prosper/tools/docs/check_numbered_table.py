#!/usr/bin/env python3
r"""Validate Markdown tables that other documents cite by row number.

(Raw string: this docstring quotes the `\|` escape it teaches, and in a normal string that is an
invalid escape sequence -- a SyntaxWarning today and an error in a future Python.)

Two defects landed in docs/GAME_COMPAT_ORCHESTRATION.md on 2026-08-01, and no other check
in the pipeline could see either one:

  * A stray blank line between two rows. In Markdown a blank line TERMINATES a table, so the
    rows after it render as a separate table -- or, for a single trailing row, as a one-row
    table of its own. Master carried two at once, rendering 36 rows as three tables. Nothing
    is missing from the file, so the diff looks perfect and only a reader of the rendered
    page would notice.

  * Two branches independently claiming the same row number. Both merge CLEANLY, because they
    are separate lines with no textual conflict, producing a table reading "33, 34, 35, 32"
    at mergeable=CLEAN with every job green. Conflict detection is textual and line-local; it
    has no notion that a column is a unique key. Other documents cite these rows BY NUMBER, so
    a duplicate silently redirects every such citation to an unrelated entry.

HOW A BREAK IS TOLD FROM TWO LEGITIMATE TABLES. A proper Markdown table opens with a header
row followed by a delimiter row (`|---|---|`). A fragment left behind by a stray blank line has
no delimiter -- it is a continuation, not a table. So an all-pipe run WITHOUT a delimiter that
directly follows a proper table, separated only by blank lines, is a break; two properly
delimited tables in one file are not. This also catches a blank line between the delimiter and
the first data row, which orphans the entire body.

Fenced code blocks are skipped: this repository's docs paste tool output containing pipe
characters, and this file's own defect example would otherwise fail the check that documents it.

FOUR CLASSES OF CHECK, deliberately separated:

  STRUCTURE (always on) -- no blank line may split a table. A fact about Markdown, true of
  every table in the repository.

  ARITY (always on) -- every row has exactly as many cells as its header. Also a fact about
  GFM rather than a convention, and it caught a defect that had been rendering wrong for
  months (#2108). GFM splits a row into cells on `|` BEFORE it parses any inline content, so a
  pipe inside an inline CODE SPAN is a cell boundary, not text -- the spec says so explicitly
  ("it's possible to include a pipe in a cell's content by escaping it, including inside other
  inline spans"). A row with more cells than the header has the excess SILENTLY DISCARDED, and
  one with fewer is silently padded. Either way the raw file looks right, the diff looks right,
  and only the rendered page is wrong -- so an editor never sees it. Six rows of this file's own
  instrument-trap table were truncated on GitHub this way, the sharpest being trap 40, whose
  subject is `cmd \| tail` and which was broken by writing `cmd | tail` unescaped: it rendered as
  an 8-column row in a 3-column table, losing most of its text. The fix is one character, `\|`.
  The delimiter row is checked against the header too, because GFM requires those to agree or the
  block is not a table at all and renders as a paragraph of literal pipes.

  ORDER (--ordered, opt-in) -- the numbered column is unique and strictly ascending. GAPS ARE
  LEGAL; see WHY GAPLESSNESS WAS REMOVED below. This is a CONVENTION of the trap table ("append,
  never renumber"), NOT a property of numbered tables in general. RE-MEASURED under this rule
  rather than inherited from --sequential's: over the 101 tracked Markdown files, all 101 satisfy
  structure, while of the 10 documents holding an all-numeric-first-column table only 6 satisfy
  order -- the rest lead with frame indices, draw ordinals and submit numbers, where repeats are
  the correct content. Applying order everywhere would report correct documents as broken, and a
  check that fires on correct data gets deleted rather than heeded.

  The figure had to be re-measured because relaxing gaplessness can only INCREASE it, so
  --sequential's "3 of 10" could not survive the rewrite unchanged (#2610 review). The three
  documents that qualify under --ordered and did not under --sequential are
  ASTROBOT_LINUX_HANDOFF_2026_07_19.md, RENDERER_PERFORMANCE_2026_07.md and
  SONIC_CROSSWORLDS_STATUS.md. The conclusion is unaffected: 4 of 10 still fail, so "do not apply
  it broadly" stands on the new rule as it did on the old one.

  Two selector details, both from real documents here rather than from theory:
    * The order is checked from its own first value, not from 1, because a genuine numbered
      work list in this repo starts at 0 (EVERGATE_PERFORMANCE_HANDOFF_2026_07.md).
    * A table qualifies only if EVERY body row is numbered. That same work list ends with a
      `| Separate | ...` row, which is legitimate content and must not be read as an unnumbered
      row to be silently ignored.

  PERSISTENCE (--baseline PATH, opt-in, requires --ordered) -- every row number present in the
  BASELINE copy of the same table is still present here. This is the check that catches a row
  which was DELETED or RENUMBERED, and it is what replaced gaplessness. It needs a second file
  because "was this row here before?" is not a question one file can answer.

Validates structure, never content, so no class can rot as a table grows.

WHAT THIS CANNOT SEE, and a real incident. It validates STRUCTURE. It does not read content, so it
cannot tell that a row still says what its author meant. On 2026-08-01 a rebase silently reverted an
entry's TEXT to a version asserting a conclusion its author had already retracted, while the
numbering stayed perfectly contiguous -- this check would have passed it without a word. The
boundary is sharp and worth knowing: numbering catches a row that VANISHED (--baseline), and
nothing here catches a row that CHANGED ITS MIND. A green run means the table is well formed, never
that it is true. Diff the region for content too.

What ARITY specifically cannot see, stated so silence is not read as coverage:
  * HTML tables (`<table>`), which this repository does not use and which are not scanned at all.
  * Any pipe run with no delimiter row -- a fragment has no header to be measured against, so its
    rows are unchecked. The structure class reports the fragment itself when it abuts a real table
    above it, but an isolated delimiter-less pipe block is invisible to every class here.
  * Tables whose rows have NO LEADING PIPE (`a | b` / `--- | ---`), and tables inside a BLOCKQUOTE
    (`> | a | b |`). GitHub renders both as real tables and truncates them exactly the same way,
    and `parse_tables` sees neither, because it requires a row to start with `|`. Both are real
    false negatives rather than judgement calls; there are zero instances in the corpus today,
    which is the only reason they are recorded here instead of fixed. Note the shape of this
    entry: it is trap 112's own rule -- a gate's silence is only evidence about the class it
    inspects -- applied to trap 112's own gate (#2116 review).
  * Content INSIDE a cell. A pipe that is correctly escaped is counted as text and nothing asks
    whether `\|` was what the author meant; equally, a cell count that matches the header proves
    nothing about the cells being in the right ORDER or the right columns.
  * Fenced regions, deliberately: a ``` block of shell pipes is not a table, and a checker that
    fires on pasted tool output gets disabled. Nested/mismatched fences (``` inside ~~~) are
    tracked as a single toggle, so a document mixing the two markers could desynchronise. This
    applies to the conflict-marker scan too, and it is a real blind spot rather than a free choice:
    a genuine unresolved conflict that happens to land inside a fenced block is invisible here. The
    alternative -- rejecting every document that quotes a marker as an example -- would reject the
    documentation of this very defect, so the trade is made the same way it is made for tables.
    `git diff --check` has no such blind spot and remains the backstop the charter runs separately.

WHY GAPLESSNESS WAS REMOVED (#2089), and what took over its job. Until 2026-08-17 the opt-in class
was `--sequential`: unique, ascending AND gapless. Gaplessness is the reason this file used to carry
a section headed "a red Docs job that is the gate working" -- two concurrent PRs appending to a
gapless table create a MERGE-ORDER DEPENDENCY. Master ends at 42, both branches append, the earlier
claimant keeps 43 and the later renumbers to 44; the later branch then reads `42, 44` and its `Docs`
job is red BY CONSTRUCTION until the earlier PR lands. Neither PR is broken, and there is nothing
its author can do about it.

That is the property worth naming, because it is what separates gaplessness from the other two
classes: **a violation of uniqueness or of ascent is repairable by the author alone** -- bump the
number by one character, or move the line -- **while a violation of gaplessness is not repairable at
all.** The missing number belongs to somebody else's unmerged branch. The only local "repair" is to
renumber into the gap, which is exactly the forbidden operation ("append, never renumber") and which
reproduces the duplicate the checker warns about. That prohibition is not ceremonial. Measured on
`origin/master` (`7413647a`) on 2026-08-17, excluding the table's self-references:

    118 references on 110 lines in 46 files, naming 63 distinct rows, every one resolving
    -- 66 of them in .md prose, 50 in .cpp/.hpp/.py COMMENTS

so a renumber breaks compiled files as well as documentation. Re-derive rather than trusting this
figure, which goes stale on every append:

    git grep -nE '(instrument[- ])?traps? [0-9]{1,3}' -- ':!prosper/docs/GAME_COMPAT_ORCHESTRATION.md'

(counting `s_trap`/`v_trap` in the shader tests as the false positives they are -- they are RDNA2
mnemonics, not citations). So the gapless rule could only be satisfied by waiting, and the cost was real:
#2089 records four lanes blocked in one session, two of them writing push-window guard scripts
rather than working on titles, and three sets of rows nearly lost because a row that has to wait
gets parked on whatever branch is at hand.

Gaplessness was kept because it catches a DELETED row -- a row that simply vanishes in a bad merge,
a wholesale `git checkout` of the file, or a careless rebase leaves a file that is perfectly well
formed, and uniqueness and structure both pass it. That reasoning is right about the danger and
wrong about the coverage. MEASURED on master (`0c268362`, whose Instrument table is 186 rows, 1..186
-- the "192" an earlier draft carried is the whole-FILE numbered-row count,
`grep -cE '^\s*\|\s*[0-9]+\s*\|'` over all 14 tables in the document, rather than this table's
186. Both figures here are re-derived; the first correction of this sentence attached 248 to that
grep, which is the document's total BODY-row count and a third wrong number in the same place):

  * delete an INTERIOR row (100)                      -> caught, rc=1
  * delete the HIGHEST row (186)                      -> GREEN, rc=0
  * restore the file wholesale from a 40-commit-old
    revision, losing 61 rows                          -> GREEN, rc=0, "contiguous and unbroken"

The last of those is the shape this repository has actually suffered: `git checkout <old-branch> --
<file>` while rebuilding a branch, which cost #1701 ten lines of documentation and is instrument
trap 41. It truncates the TAIL, and a truncated tail is still gapless. So gaplessness was a partial
deleted-row detector, blind to the dominant real-world case, bought at the price of serializing
every lane that appends.

--baseline is the whole detector, and it is free of the coupling: it compares this file's row
numbers against the same table in the PR's own base, so a row that vanished is reported whether it
came from the middle, the end, or a whole-file revert -- and a RENUMBER is reported too, which
nothing here could see before. Nothing another lane does can make it fire. CI passes `HEAD^1`'s copy
of the file, which is the base commit on a `pull_request` merge ref and the previous master on a
push, so both the pre-merge and the post-merge run are covered.

TWO ALTERNATIVES WERE REJECTED, recorded here because the next person to find this painful will
propose one of them, and both are more expensive than they look:

  * BATCHING (#2089's own proposal) -- lanes write rows unnumbered into a staging area, and one
    periodic PR assigns a contiguous block. It does remove the serialization. It also destroys the
    property #2089 correctly insists on keeping: a row staged and never promoted is INVISIBLE, since
    there is no gap to notice, because it never had a number. That trades a check which fires for one
    which cannot, and adds a mandatory orchestrator step to every lane. --baseline gets the lane
    independence without the staging area.

  * STABLE NON-SEQUENTIAL IDS (date- or hash-prefixed) plus a generated index (#1664). Rejected for
    the reason its own filer conceded when they read #1729: the identifier was never the problem, and
    changing it breaks every existing citation to buy a property that a claim-time check gets for
    free.

Both were declined in favour of removing the one requirement that could not be satisfied locally.

WHAT THE GATE CAN NO LONGER CATCH, stated plainly so silence is not read as coverage. A number that
is allocated and then never used -- an abandoned PR, a row dropped in review -- leaves a permanent
gap, and nothing reports it. That is deliberate: such a gap is not a defect, and treating it as one
is what produced the serialization. The success line prints the unused numbers so a reader can see
them, but they are informational. If a lane writes a row, never opens a PR, and tells nobody, no
check here will notice -- the same as today, since that row never reached a numbered table at all.

KNOWN LIMIT, recorded as a decision rather than a defect. The abutment rule keeps the
`blank line, prose, orphaned rows` shape -- a real table split -- and that shape is structurally
identical to a correct one: a pipe-leading line in the middle of a paragraph below an unrelated
table. No rule separates those two, so that half is a choice between two errors rather than a bug to
fix -- abutment keeps the genuine defect, and the shape it misreads has zero instances across the
tracked corpus and is bounded to a single message. If it starts appearing, narrowing to "gap
contains no blank line" trades it the other way at a known cost.

(A fence between a table and further rows was misread the same way. That half was separable and is
now fixed -- #1709: fence line numbers are recorded and a gap containing one is not a split.)
"""

from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache
from pathlib import Path

FENCE = re.compile(r"^\s*(```|~~~)")
DELIMITER = re.compile(r"^\s*\|[\s:|-]+\|?\s*$")
LEADING_NUMBER = re.compile(r"^\s*\|\s*(\d+)\s*\|")

# See N1 in persistence_problems: how far above the baseline's maximum a NEW row number may
# sit before it is treated as a typo rather than as a deliberate step clear of a collision.
MAX_JUMP = 50

# Unresolved merge markers. `=======` is deliberately NOT included: it is a legal setext heading
# underline in Markdown, and a checker that fires on correct documents gets deleted rather than
# heeded. The other two are unambiguous -- measured across the tracked corpus, zero instances of
# either, which is what makes this safe to gate on.
CONFLICT = re.compile(r"^(<<<<<<<|>>>>>>>)")


ESCAPED_PIPE = re.compile(r"(?<!\\)\|")


def delimiter_pipes(line: str) -> list[int]:
    """Indices of every `|` in `line` that GFM treats as a CELL BOUNDARY.

    The rule is exactly "a pipe is a delimiter unless the IMMEDIATELY PRECEDING character is a
    backslash" -- no backslash counting. cmark-gfm splits table rows into cells in a pre-inline
    pass that only looks at the one character before the pipe; CommonMark's parity rule for
    backslash escapes belongs to the *inline* pass, which never runs on a cell boundary.

    This is measured, not reasoned. Rows `b|c` through `b\\\\\\\\|c` (0 to 4 backslashes) were put
    through GitHub's own renderer, with the 0-backslash row as a positive control:

        0 backslashes  ->  splits, trailing cell discarded   (control: the instrument works)
        1 backslash    ->  one cell, renders `b|c`
        2 backslashes  ->  one cell, renders `b|c`      <-- a parity rule predicts a SPLIT here
        3 backslashes  ->  one cell, renders `b\\|c`
        4 backslashes  ->  one cell, renders `b\\|c`

    So every backslash run of length >= 1 escapes the pipe for splitting purposes. Only the
    rendered *text* differs above 2, because the inline pass then applies CommonMark escaping to
    what the split left behind -- and cell ARITY, which is all this measures, is unaffected.

    An earlier revision of this function counted parity and therefore reported `| a | b\\\\|c | d |`
    as a truncated row, which GitHub renders perfectly (#2116 review).

    Nothing else protects a pipe -- not an inline code span, not emphasis, not a link title --
    because GFM performs this split BEFORE any inline parsing. That is the entire defect this
    exists to catch (#2108).
    """
    return [m.start() for m in ESCAPED_PIPE.finditer(line)]


def cell_bounds(line: str) -> list[int]:
    """The delimiter pipes that actually END a cell, after dropping an optional leading pipe.

    GFM treats a leading and a trailing pipe as optional decoration rather than as cell
    boundaries, so `| a | b | c |` is three cells and not five. Cell k (1-based) therefore ends
    at the returned index k-1, which is what lets a caller point at the FIRST pipe that should
    not be there rather than merely reporting a count.
    """
    pipes = delimiter_pipes(line)
    if not pipes:
        return []
    # Leading pipe: everything before the first delimiter is blank (up to 3 spaces of indent).
    if not line[: pipes[0]].strip():
        pipes = pipes[1:]
    return pipes


def cell_count(line: str) -> int:
    """Number of cells GFM renders for this row."""
    bounds = cell_bounds(line)
    if not bounds:
        return 1 if line.strip() else 0
    # Each bound closes a cell; a trailing pipe with only blank text after it closes nothing.
    trailing = not line[bounds[-1] + 1 :].strip()
    return len(bounds) if trailing else len(bounds) + 1


def excerpt(line: str, pos: int, width: int = 46) -> str:
    """`...text|text...` around `pos`, so the message points at the offending character."""
    lo, hi = max(0, pos - width), min(len(line), pos + width)
    return ("..." if lo else "") + line[lo:hi].strip() + ("..." if hi < len(line) else "")


class Table:
    """A maximal run of consecutive table lines, with 1-based source line numbers."""

    def __init__(self, start: int, lines: list[str]) -> None:
        self.start = start
        self.lines = lines

    @property
    def proper(self) -> bool:
        """True if this opens with a header + delimiter, i.e. it is a table rather than a fragment."""
        return len(self.lines) >= 2 and bool(DELIMITER.match(self.lines[1]))

    @property
    def body(self) -> list[tuple[int, str]]:
        """(line_no, text) for rows after the header and delimiter."""
        skip = 2 if self.proper else 0
        return [(self.start + i, l) for i, l in enumerate(self.lines)][skip:]

    def numbered_rows(self) -> list[tuple[int, int]]:
        out = []
        for line_no, text in self.body:
            m = LEADING_NUMBER.match(text)
            if m:
                out.append((line_no, int(m.group(1))))
        return out

    @property
    def all_rows_numbered(self) -> bool:
        rows = self.body
        return bool(rows) and len(self.numbered_rows()) == len(rows)

    @property
    def header(self) -> str:
        return self.lines[0].strip() if self.lines else ""


def parse_tables(lines: list[str]) -> tuple[list[Table], set[int], set[int]]:
    """Split into table runs, ignoring fenced code.

    Returns (tables, blank_line_numbers, fence_line_numbers). The fence lines are needed
    because a fence opening straight after a table looks exactly like the interruption of
    a split table (#1709) -- it is a non-blank line between two pipe runs -- but it is a
    correct document. It is separable by a rule as principled as the blank-line one, which
    is why this is a fix rather than another entry in KNOWN LIMIT.
    """
    tables: list[Table] = []
    blanks: set[int] = set()
    fences: set[int] = set()
    fenced = False
    run: list[str] = []
    run_start = 0

    for i, line in enumerate(lines, start=1):
        if FENCE.match(line):
            fences.add(i)
            fenced = not fenced
            if run:
                tables.append(Table(run_start, run))
                run = []
            continue
        if fenced:
            continue
        # 4+ columns of indentation is a code block in Markdown, not a table row. Tabs count as
        # indentation too, so expand them first: a tab-indented pipe block previously read as a
        # table fragment and drew "Delete the blank line", which would merge a code block into a
        # table. A wrong message that instructs is worse than one that merely fires.
        indent = line.expandtabs(4)
        if line.startswith("|") or (indent[:4].strip() and indent.lstrip().startswith("|")):
            if not run:
                run_start = i
            run.append(line)
            continue
        if not line.strip():
            blanks.add(i)
        if run:
            tables.append(Table(run_start, run))
            run = []
    if run:
        tables.append(Table(run_start, run))
    return tables, blanks, fences


def arity_problems(path: Path, table: Table) -> list[str]:
    """Every row of a proper table must have its header's cell count. See ARITY above.

    Only PROPER tables are checked: a fragment has no header of its own, so there is nothing to
    measure it against, and the structure class is what reports the fragment.
    """
    problems: list[str] = []
    header_line, delimiter_line = table.lines[0], table.lines[1]
    expected = cell_count(header_line)

    # GFM requires the delimiter to match the header, or the block is not a table at all -- it
    # renders as a paragraph of literal pipes. Worth its own message because the symptom (the
    # whole table gone) looks nothing like the symptom of a bad body row (one row truncated).
    delimiter_cells = cell_count(delimiter_line)
    if delimiter_cells != expected:
        problems.append(
            f"{path}:{table.start + 1}: the delimiter row has {delimiter_cells} cell(s) but the "
            f"header at line {table.start} has {expected} -- GFM requires them to match, so this "
            f"block does NOT render as a table at all; it becomes a paragraph of literal pipes. "
            f"Delimiter reads: {delimiter_line.strip()[:60]!r}"
        )
        return problems  # every body row would be measured against a header GFM never accepted

    for line_no, text in table.body:
        actual = cell_count(text)
        if actual == expected:
            continue
        if actual > expected:
            # Point at the pipe that closed the last legitimate cell early: in a correct row that
            # pipe would be the trailing one, so it is exactly the character to escape or remove.
            bounds = cell_bounds(text)
            stray = bounds[expected - 1]
            problems.append(
                f"{path}:{line_no}: table row has {actual} cells but the header at line "
                f"{table.start} has {expected} -- GitHub renders the first {expected} and "
                f"SILENTLY DISCARDS the rest, so this row loses its last column(s) on the "
                f"rendered page while looking correct in the file. GFM splits a row on `|` "
                f"BEFORE parsing inline content, so a pipe inside a `code span` is a cell "
                f"boundary; escape it as `\\|`. First stray boundary at column {stray + 1}: "
                f"{excerpt(text, stray)}"
            )
        else:
            problems.append(
                f"{path}:{line_no}: table row has {actual} cells but the header at line "
                f"{table.start} has {expected} -- GitHub pads the row with empty cells, so the "
                f"missing column renders blank rather than reporting anything. Did a cell get "
                f"dropped? Row reads: {text.strip()[:80]!r}"
            )
    return problems


def read_lines(path: Path) -> tuple[list[str] | None, str | None]:
    """(lines, error). Split out so --baseline reports its own file rather than the subject's."""
    try:
        return path.read_text(encoding="utf-8").split("\n"), None
    except OSError as exc:
        return None, f"{path}: cannot read: {exc}"
    except UnicodeDecodeError as exc:
        return None, f"{path}: is not valid UTF-8: {exc}"


def select_numbered_table(
    path: Path, tables: list[Table], table_header: str | None
) -> tuple[Table | None, list[str]]:
    """The one fully-numbered table --ordered was pointed at, or the reason there isn't one.

    Shared with --baseline so the two files are never measured against DIFFERENT tables: a
    baseline whose header text no longer matches would otherwise silently compare nothing and
    report every row as intact.
    """
    candidates = [t for t in tables if t.proper and t.all_rows_numbered]
    if table_header:
        candidates = [t for t in candidates if table_header in t.header]
    if not candidates:
        return None, [
            f"{path}: --ordered found no numbered table"
            + (f" whose header contains {table_header!r}" if table_header else "")
        ]
    if len(candidates) > 1:
        # Ambiguous rather than guessed: checking the wrong table would report correct data as
        # broken, and a check that fires on correct data gets deleted rather than heeded.
        where = ", ".join(f"line {t.start}" for t in candidates)
        return None, [
            f"{path}: --ordered is ambiguous -- {len(candidates)} numbered tables ({where}). "
            f"Select one with --table-header."
        ]
    return candidates[0], []


@lru_cache(maxsize=None)
def load_baseline(baseline_path: str, table_header: str | None) -> tuple[tuple[int, ...] | None, str]:
    """The baseline table's row numbers, parsed ONCE. Returns (numbers, error-or-empty).

    Cached deliberately, and not merely to save work. The comparison and the count printed on the
    success line are two reads of the same file, and two reads can disagree: `--baseline <(cat f)`
    is a process substitution, which is a single-read FIFO, so the second read saw an empty file and
    the run printed "none of the None rows ... has been deleted" and exited 0 (#2610 review). A
    falsifiability device that can print `None` is not one. One parse, one answer.
    """
    baseline = Path(baseline_path)
    lines, err = read_lines(baseline)
    if err:
        return None, f"--baseline {err}"
    tables, _, _ = parse_tables(lines or [])
    table, problems = select_numbered_table(baseline, tables, table_header)
    if table is None:
        return None, "; ".join(f"--baseline: {p}" for p in problems)
    return tuple(n for _, n in table.numbered_rows()), ""


def persistence_problems(
    path: Path, table: Table, baseline: Path, table_header: str | None
) -> list[str]:
    """Every row number the BASELINE had must still be here. See PERSISTENCE above.

    This is what replaced gaplessness, and it is strictly stronger at the job gaplessness was kept
    for: a gap could only see an INTERIOR deletion, so a truncated tail -- which is what
    `git checkout <old-branch> -- <file>` produces, instrument trap 41 / #1701 -- passed it green.
    Comparing against the base sees all three shapes, plus a renumber, which no single-file rule
    can distinguish from an append plus a deletion.

    Fails closed. A baseline that cannot be read, or whose matching table cannot be found, is an
    ERROR rather than a skip: the caller asked for this check, and a silent skip would leave a
    green run meaning nothing.
    """
    base_numbers, err = load_baseline(str(baseline), table_header)
    if base_numbers is None:
        return [err]

    now = {num for _, num in table.numbered_rows()}
    problems: list[str] = []

    # N1 -- a bound on how far a NEW number may sit above everything the base held. Gaps are legal,
    # but a mistyped digit is not: appending `1189` instead of `189` passes uniqueness and ascent,
    # reports "1000 unused numbers", and permanently poisons the space, because the allocator then
    # hands out 1190 and everything after it. The old gapless rule caught this incidentally; nothing
    # else does. The bound is deliberately local -- it compares against the BASE's own maximum, so
    # no other lane's merge timing can make it fire -- and generous, since no realistic set of
    # concurrent lanes opens a gap this wide.
    if base_numbers:
        ceiling = max(base_numbers) + MAX_JUMP
        for line_no, num in table.numbered_rows():
            if num not in base_numbers and num > ceiling:
                problems.append(
                    f"{path}:{line_no}: row number {num} is more than {MAX_JUMP} above the highest "
                    f"row in the baseline ({max(base_numbers)}). Gaps are legal, but a jump this "
                    f"large is almost always a mistyped digit -- and it is not self-correcting, "
                    f"because every later allocation is taken from the new maximum. If the jump is "
                    f"deliberate, lower it: nothing requires stepping this far clear."
                )

    missing = sorted(n for n in base_numbers if n not in now)
    if not missing:
        return problems
    listed = ", ".join(str(n) for n in missing[:12]) + (" ..." if len(missing) > 12 else "")
    return problems + [
        f"{path}:{table.start}: {len(missing)} row number(s) present in the baseline are GONE "
        f"from this table: {listed}. A row that vanishes leaves a perfectly well-formed file, so "
        f"nothing else here can see it -- the usual causes are a whole-file `git checkout "
        f"<branch> -- <file>` while rebuilding a branch (instrument trap 41), a bad merge "
        f"resolution, or a renumber. Numbers are cited BY NUMBER from other documents, so restore "
        f"the rows rather than closing the hole -- and if a renumber was intended, it is not "
        f"allowed: append, never renumber."
    ]


def check(
    path: Path,
    ordered: bool,
    table_header: str | None,
    baseline: Path | None = None,
) -> list[str]:
    """Return a list of problems; empty means the tables are sound."""
    lines, err = read_lines(path)
    if err:
        return [err]
    assert lines is not None

    # ALWAYS ON, and it caught nothing here until it caught this file's own author. A stray
    # `>>>>>>> <sha>` left by a conflict resolution landed immediately after the LAST row of the
    # instrument table, and every check in this file passed it: the marker is not a table row, so it
    # simply ENDED the table, and with no orphaned rows below it there was no fragment to report.
    # `git diff --check` found it instead -- which is why the charter runs that as a separate gate --
    # but nothing stops a marker reaching a branch where nobody runs it, and the numbering was
    # meanwhile reported "unique and ascending" over a line that was a merge artifact.
    #
    # FENCED REGIONS ARE SKIPPED, like every other class here, and the first draft of this scan
    # forgot it -- running before parse_tables, with no fence tracking, always on, including the
    # repo-wide *.md sweep. It therefore rejected any document that PASTES an example conflict
    # inside a ``` block, which is exactly how this defect gets documented. This file's own header
    # settled that question long ago: "this file's own defect example would otherwise fail the check
    # that documents it." Same self-referential shape as the marker that got past it in the first
    # place (#2610 review).
    conflicts: list[str] = []
    fenced = False
    for i, line in enumerate(lines, start=1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if fenced or not CONFLICT.match(line):
            continue
        conflicts.append(
            f"{path}:{i}: unresolved merge conflict marker: {line.strip()[:60]!r}. A marker directly "
            f"after a table's last row ENDS the table rather than splitting it, so no other check "
            f"here can see it."
        )
    if fenced:
        # An UNCLOSED fence makes every check here go quiet from that point on -- the marker scan
        # above, the table scan below -- so a real conflict marker after it reads CLEAN. That is the
        # skip turning into a blind spot rather than a trade, and unlike the fenced-example case it
        # has no legitimate shape: a document that opens a fence and never closes it is malformed.
        # Safe to gate on by the same measurement that made the marker scan safe: 0 of 101 tracked
        # Markdown files end with an unbalanced fence count (#2610 review).
        conflicts.append(
            f"{path}: a fenced block is opened and never closed, so every check in this file goes "
            f"silent from that point on -- including the conflict-marker scan. Close the fence."
        )
    if conflicts:
        # Return immediately: every other class would be measuring a file that is half two files.
        return conflicts

    tables, blanks, fences = parse_tables(lines)
    if not tables:
        # Under --ordered we were pointed at ONE specific table, so its absence means the path
        # is wrong or the format changed, and failing here is what stops this check passing
        # forever on a file it can no longer see. In a plain structure sweep across many
        # documents, a file with no table is ordinary and must not be an error -- 35 of this
        # repo's 77 Markdown files have none.
        if ordered:
            return [f"{path}: no Markdown tables found -- wrong path, or the table format changed"]
        return []

    problems: list[str] = []

    # `origin` is the proper table the current run of fragments descends from. It must survive a
    # fragment, because a table broken twice yields fragment-after-fragment: comparing each run
    # only against the run immediately above it would report the first break and miss every later
    # one.
    #
    # What separates the runs does NOT decide whether this is a break -- only whether the run
    # below is a proper table does. A blank line and a row that lost its leading pipe both end a
    # table, and the second is the likelier lane accident. Treating a non-blank gap as "these are
    # unrelated" loses that entirely: the rows below leave the sequence check with it, so a
    # duplicate number underneath passes green, and the success line reports the truncated row
    # count as though it were the whole table.
    origin: Table | None = None
    for previous, current in zip(tables, tables[1:]):
        if previous.proper:
            origin = previous
        if current.proper or origin is None:
            continue  # a table of its own, or a fragment with no proper table above it

        # What separates the runs does not decide whether this is a break, but it does decide
        # whether they are ADJACENT -- and without that bound, backward attribution is unbounded:
        # a delimiter-less table 200 lines below an unrelated one produced 202 errors, each
        # accusing a prose line, with the real cause pushed past GitHub's 10-annotation limit.
        # Two of them told the author to delete the blank lines that correctly ended the table,
        # and wrong guidance is worse than a false positive: an agent who follows it damages the
        # file. A fragment continues the table above only if it abuts it -- the whole gap is
        # blank, or the line immediately above the fragment is itself the interruption.
        gap_start = previous.start + len(previous.lines)
        gap = range(gap_start, current.start)
        # #1709: a fence opening immediately after a table is a correct document, not an
        # interruption. It is separable from a genuine split because a genuine one never
        # contains a fence delimiter -- the fenced content is skipped entirely, so the run
        # below it is code, not orphaned rows.
        if any(n in fences for n in gap):
            origin = None
            continue
        all_blank = all(n in blanks for n in gap)
        interrupted_at = current.start - 1
        if not all_blank and interrupted_at in blanks:
            continue  # separated by prose that ends in a blank line: not this table's continuation

        # One problem per fragment, not one per gap line: the fragment is the defect, and a gap of
        # N blank lines is one mistake, not N.
        if all_blank:
            problems.append(
                f"{path}:{gap_start}: blank line splits the table that starts at line "
                f"{origin.start} -- in Markdown a blank line ends a table, so line {current.start} "
                f"onward renders as a separate table. Delete the blank line."
            )
        else:
            problems.append(
                f"{path}:{interrupted_at}: line is not a table row but sits inside the table that "
                f"starts at line {origin.start}, ending it -- so line {current.start} onward "
                f"renders as a separate table and drops out of any sequence check. Did it lose "
                f"its leading '|'? Line reads: {lines[interrupted_at - 1].strip()[:60]!r}"
            )

    # ARITY, always on and independent of the structure loop above: a split table and a truncated
    # row are unrelated defects, and a file can carry either without the other.
    for table in tables:
        if table.proper:
            problems.extend(arity_problems(path, table))

    if not ordered:
        return problems

    table, selection_problems = select_numbered_table(path, tables, table_header)
    if table is None:
        return problems + selection_problems

    rows = table.numbered_rows()
    seen: dict[int, int] = {}
    highest = rows[0][1]
    for line_no, num in rows:
        if num in seen:
            # Repairable by this author alone, and that is the point: bump the LATER row. Under the
            # gapless rule the bump had to be re-derived against whatever master looked like at that
            # moment, and re-derived again if a lane landed in between (#2581). It no longer does --
            # any number above the current highest is valid, so overshooting is safe.
            problems.append(
                f"{path}:{line_no}: duplicate row number {num} (first used at line {seen[num]}). "
                f"Two branches almost certainly appended the same number concurrently -- such a "
                f"merge is textually clean. Renumber the LATER row to {max(seen) + 1} or higher "
                f"(any higher number is fine; gaps are legal), and re-check every by-number "
                f"citation of it elsewhere in the docs."
            )
            continue
        seen[num] = line_no
        if num < highest:
            # Deliberately does not say which row to move: either this one earlier or the higher one
            # later restores the order, and which is correct depends on whose row arrived by a merge.
            # What it must say is that renumbering is NOT the repair -- that is the reflex, and it is
            # the one operation the citation contract forbids.
            problems.append(
                f"{path}:{line_no}: row number {num} is out of ascending order (an earlier row "
                f"already reached {highest}). The table is sorted by this column, so MOVE the row "
                f"to its numeric position. Do NOT renumber it -- other documents cite it by number."
            )
        highest = max(highest, num)

    if baseline:
        problems.extend(persistence_problems(path, table, baseline, table_header))
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", type=Path, help="Markdown files to check")
    ap.add_argument(
        "--ordered",
        action="store_true",
        help="also require the numbered column to be unique and strictly ascending (a convention "
        "of the instrument-trap table, not of numbered tables generally). GAPS ARE LEGAL -- pair "
        "with --baseline to catch a deleted row",
    )
    ap.add_argument(
        "--sequential",
        action="store_true",
        help=argparse.SUPPRESS,  # removed 2026-08-17; kept only to fail loudly. See below.
    )
    ap.add_argument(
        "--baseline",
        metavar="PATH",
        type=Path,
        help="with --ordered, require every row number in PATH's copy of the same table to still "
        "be present (catches a deleted or renumbered row). CI passes `git show HEAD^1:<file>`",
    )
    ap.add_argument(
        "--table-header",
        metavar="TEXT",
        help="with --ordered, select the table whose header row contains TEXT",
    )
    ap.add_argument(
        "--github",
        action="store_true",
        help="emit ::error:: annotations so failures surface on the GitHub Actions summary",
    )
    args = ap.parse_args()
    # An error rather than an alias. --sequential meant unique + ascending + GAPLESS, and silently
    # accepting it under the new semantics would leave every caller believing a check is running
    # that was deliberately removed (#2089). A caller that is told gets to decide.
    if args.sequential:
        ap.error(
            "--sequential was removed on 2026-08-17: gaplessness serialized every lane appending "
            "to the table and could not see a truncated tail anyway (#2089). Use --ordered "
            "(unique + ascending), and --baseline <base copy> for the deleted-row check that "
            "replaced it. See this file's WHY GAPLESSNESS WAS REMOVED."
        )
    if args.table_header and not args.ordered:
        ap.error("--table-header only applies with --ordered")
    if args.baseline and not args.ordered:
        ap.error("--baseline only applies with --ordered")
    if args.baseline and len(args.paths) > 1:
        # One baseline cannot describe several subjects, and picking one silently would compare
        # the wrong pair.
        ap.error("--baseline takes exactly one subject path")

    problems: list[str] = []
    for path in args.paths:
        problems.extend(check(path, args.ordered, args.table_header, args.baseline))

    for problem in problems:
        if args.github:
            # GitHub decodes these three in annotation text; leaving them raw truncates messages.
            escaped = problem.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            print(f"::error::{escaped}")
        else:
            print(f"error: {problem}", file=sys.stderr)

    if problems:
        print(f"\n{len(problems)} problem(s) found.", file=sys.stderr)
        return 1

    for path in args.paths:
        tables, _, _ = parse_tables(path.read_text(encoding="utf-8").split("\n"))
        proper = [t for t in tables if t.proper]
        rows = sum(len(t.body) for t in proper)
        summary = f"{path}: {len(proper)} table(s), {rows} rows, unbroken, every row matching its header"
        if args.ordered:
            # Report the unused numbers rather than staying silent about them. They are LEGAL --
            # an abandoned PR or a row dropped in review retires its number -- but a reader
            # allocating the next one wants to see the shape of the space, and a sudden jump of
            # fifty is worth a glance even though no rule forbids it.
            table, _ = select_numbered_table(path, tables, args.table_header)
            if table is not None:
                nums = sorted(n for _, n in table.numbered_rows())
                unused = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))
                listed = ", ".join(str(n) for n in unused[:12]) + (" ..." if len(unused) > 12 else "")
                gaps = f", {len(unused)} unused number(s): {listed}" if unused else ", no unused numbers"
                summary += (
                    f"\n{path}: numbered column {nums[0]}..{nums[-1]}, "
                    f"{len(nums)} rows, unique and ascending{gaps}"
                )
                if args.baseline:
                    base_numbers, _ = load_baseline(str(args.baseline), args.table_header)
                    n = len(base_numbers) if base_numbers is not None else 0
                    summary += f"; none of the {n} rows in {args.baseline} has been deleted"
        print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
