#!/usr/bin/env python3
"""Self-checking tests for check_trap_citations.py (exit code is truth).

Two halves, and the second is the one that keeps the check alive. A citation checker that fires on
correct prose gets disabled within a day, and this one scans every tracked file in the repository --
including a thousand source files whose comments mention all sorts of things. So the false-positive
cases here (`s_trap 1`, a bare number, a word that merely contains "trap") carry as much weight as
the true positives.

Run directly, or via ctest as trap_citation_checker.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "check_trap_citations.py"

FAILURES: list[str] = []

TABLE = (
    "# doc\n\n| # | Instrument | How it lied |\n|---|---|---|\n"
    + "".join(f"| {n} | thing {n} | evidence |\n" for n in (1, 2, 3, 7))
    + "\n### Next\n"
)


def case(name: str, citing: str, *, want_rc: int, expect_text: str | None = None,
         table: str = TABLE) -> None:
    with tempfile.TemporaryDirectory() as d:
        t = Path(d) / "table.md"
        c = Path(d) / "citing.cpp"
        t.write_text(table, encoding="utf-8")
        c.write_text(citing, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(TOOL), "--table", str(t), str(c), str(t)],
            capture_output=True, text=True,
        )
    out = proc.stdout + proc.stderr
    if proc.returncode != want_rc:
        FAILURES.append(f"{name}: expected rc={want_rc}, got {proc.returncode}: {out[:300]!r}")
        return
    if expect_text and expect_text not in out:
        FAILURES.append(f"{name}: expected {expect_text!r}, got {out[:300]!r}")
        return
    print(f"  ok  {name}")


print("references that do not resolve:")

case("a reference above the table's range",
     "// see instrument trap 99 for why\n",
     want_rc=1, expect_text="cites row 99")

# A gap is legal in the table (#2089), so a number INSIDE the range can still be absent -- and this
# is the case a "is it <= max?" check would miss entirely.
case("a reference INTO A GAP, inside the range but absent",
     "// see instrument trap 5 for why\n",
     want_rc=1, expect_text="cites row 5")

# Every reference on the line must be checked, not just the first. "traps 64, 116 and 121" is a real
# shape from CLAUDE.md, and a checker that took only the first number would pass two thirds of it.
case("a comma-and list is expanded, and the bad member is named",
     "// traps 1, 2 and 99 all apply\n",
     want_rc=1, expect_text="cites row 99")

case("every dangling reference is reported, not just the first",
     "// trap 88 here\n// and trap 99 there\n",
     want_rc=1, expect_text="2 dangling reference(s) found")

print("correct shapes that must NOT fire:")

case("a resolving reference passes", "// instrument trap 7 explains it\n", want_rc=0)
case("a plain list of resolving numbers passes", "// traps 1, 2 and 3\n", want_rc=0)

# THE PREFIX FORMS ARE ASSERTED IN THE FAILING DIRECTION, and that is not stylistic. An arm that
# says "`instrument-trap 3` passes" is VOID: if the parser stopped recognising the form entirely it
# would find no reference at all and still exit 0, so rc=0 cannot distinguish "recognised and
# resolved" from "never seen". The #2621 review caught exactly this — deleting the whole
# `(?:instrument[- ]|orchestration )` group left three such arms green, because the bare `trap N`
# inside each string still matched. Only a citation that must be REPORTED proves it was parsed.
case("the hyphenated form is recognised (asserted by failing)",
     "// see instrument-trap 99\n", want_rc=1, expect_text="cites row 99")
# These two are NOT alternation tests and are named for what they actually establish: deleting the
# `(?:instrument[- ]|orchestration )` group leaves both green, because the bare `trap 99` inside each
# string still matches through the other branch. They pin that a prefixed citation is found at all,
# which is worth having; only the HYPHENATED case above can discriminate the alternation, because a
# hyphen is the one separator the bare branch refuses (#2621 review).
case("a prefixed citation is found (does not discriminate the alternation)",
     "// see instrument trap 99\n", want_rc=1, expect_text="cites row 99")
case("an orchestration-prefixed citation is found (does not discriminate the alternation)",
     "// per orchestration trap 99\n", want_rc=1, expect_text="cites row 99")

# THE ARM THAT WOULD HAVE CAUGHT THE REGRESSION. The obvious trailing boundary, `(?![\w.])`, also
# rejects a full stop -- which is how a sentence ends -- so `See instrument trap 41.` stopped being a
# citation at all. 22 of 118 references vanished from the corpus, 5 of them in .cpp/.hpp/test
# comments, and NOTHING went red: the gate stays green while the success line still claims "all
# resolving". A citation auditor that silently stops seeing sentence-final citations and then reports
# full coverage is precisely the shape it exists to police.
case("a sentence-final citation is still a citation",
     "See instrument trap 99.\n", want_rc=1, expect_text="cites row 99")

# ...and the same at the end of a LIST, which the broken form also truncated: `traps 55 and 56.`
# was yielding only 55, so the loss was not confined to single references.
case("the last member of a sentence-final list is still a citation",
     "See traps 1, 2 and 99.\n", want_rc=1, expect_text="cites row 99")

# The counter-arm that keeps both honest: a DECIMAL must still be rejected, which is the entire
# reason the boundary exists. `(?!\w|\.\d)` excludes only a digit after the stop.
case("a decimal is still not a citation, with the looser boundary",
     "// trap 99.5 ms of overhead\n", want_rc=0)

# `trap #NN` is live in this repository — DRAGON_QUEST_STATUS.md, OREGON_TRAIL_STATUS.md and
# SYBERIA_STATUS.md all use it. Missing it while the success line says "all resolving" would be a
# completeness claim the tool does not have (#2621 review).
case("the hash form is recognised (asserted by failing)",
     "// see instrument trap #99\n", want_rc=1, expect_text="cites row 99")
case("...and a resolving hash reference passes", "// see trap #7\n", want_rc=0)

# ...but a hash after a COMPOUND WORD is not a citation. `[agc] WRITE-TRAP #1[val=…]` is a pasted
# diagnostic line in tools/AGENTS.md, and accepting `#` naively turns it into a reference to row 1.
# This is why the leading boundary is split: an explicitly prefixed citation may follow a hyphen
# ("instrument-trap 43"), a bare one may not.
# NOTE the operand: the real log line reads `WRITE-TRAP #1`, and a fixture using 1 would be the
# THIRD void arm in this file — 1 is a valid row, so the case passes whether the compound word was
# excluded or matched-and-resolved. Every "must not fire" arm here uses a number the fixture table
# does not hold, which is the only thing that makes rc=0 mean "not matched".
case("a hyphenated compound word with a hash is not a citation",
     "    [agc] WRITE-TRAP #99[val=0x5a]\n", want_rc=0)

# B1 (#2621 review) — the number needs a TRAILING boundary or `\d{1,3}` bites a prefix out of
# anything longer. Each of these was a real defect: the first turns a repo-wide required gate RED on
# correct code, the second is worse because it SILENTLY resolves to an unrelated row.
case("a hex vector is not a citation to row 0",
     "// trap 0x40 is the debug vector\n", want_rc=0)
case("a longer number is not truncated into a valid row",
     "// the guest traps 1234 times per frame\n", want_rc=0)
case("a decimal measurement is not a citation",
     "// trap 41.5 ms of overhead\n", want_rc=0)

# THE discriminating false positive. `s_trap` is an RDNA2 instruction mnemonic and appears in the
# shader recompiler's tests; without the identifier-character lookbehind the real corpus reports
# three spurious references (`prosper/tests/gpu/agc/test_rdna2_to_spirv.cpp` among them), and both of these
# cases fail. Mutation-checked: deleting `(?<![A-Za-z0-9_])` makes exactly these two go red.
#
# The operand here is deliberately a number the table does NOT hold. The repository's actual text is
# `// s_trap 1`, and a fixture using 1 would be VOID: 1 is a valid row, so the case passes whether
# the mnemonic was excluded or matched-and-resolved, and it cannot tell the two apart. That was the
# first draft of this case, and the mutation arm is what exposed it.
case("s_trap is an RDNA2 mnemonic, not a citation",
     "        0xBF920063u, // s_trap 99\n", want_rc=0)
case("v_trap is not a citation either", "// v_trap 99 is an encoding\n", want_rc=0)

# Narrowness is deliberate: an unrecognised phrasing must be silently ignored rather than guessed
# at, because a checker that fires on prose gets deleted rather than heeded.
case("a bare number is not a citation", "// see entry 99\n", want_rc=0)
case("a word merely containing 'trap' is not a citation", "// the trapdoor 99 opens\n", want_rc=0)

print("fails closed:")

# If the table cannot be found the run means nothing either way, so it must say so rather than
# report a count. rc=2 separates "the instrument is broken" from "your references are broken".
case("a table that cannot be found is an error, not a clean run",
     "// trap 1\n", want_rc=2, expect_text="no single numbered table",
     table="# doc\n\nno table here at all\n")

# The success line quotes the counts. A scan that matched nothing -- a broken regex, a wrong file
# list -- exits 0 exactly like a clean one, and the counts are the only way to tell them apart.
case("the success line quotes the reference and row counts",
     "// trap 1 and trap 7\n", want_rc=0, expect_text="naming 2 distinct row(s), all resolving")

print()
if FAILURES:
    for f in FAILURES:
        print(f"FAIL: {f}", file=sys.stderr)
    print(f"\n{len(FAILURES)} case(s) failed.", file=sys.stderr)
    sys.exit(1)
print("all cases passed")
