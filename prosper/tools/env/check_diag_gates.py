#!/usr/bin/env python3
"""Find diagnostics whose PRODUCER and PRINTER are armed by different PROSPER_* switches (#2149).

The defect shape, stated once:

    A diagnostic prints a field whose producer is armed by a different switch than the one that
    arms the printer. When the producer is unarmed the field reads as a confident, well-formed
    ZERO -- indistinguishable from a measured negative.

That zero is not merely uninformative. It is affirmatively misleading, because a zero in these
fields is usually exactly what the interesting hypothesis predicts, and it lands in `## Ruled out`
rows and issue comments as a *measurement*. #2111's instrument emitted 183 negatives it had no
capacity to falsify. The failure is silent, survives review, and looks like rigour: the output is
well-formed and the number is specific.

Three signatures are checked. Each was derived from a MEASURED instance, not invented:

  SPLIT-LOCAL  A local declared with a default (`= 0`) whose only real writes sit inside an
               env-gated region, printed OUTSIDE that region. The declaration's default is the
               lie the reader sees.
               Measured instance: `fresh_extent` on #2132 -- `uint32_t fresh = 0;` filled only
               under `udprov_enabled()` (PROSPER_UDPROV), printed by `[udcand]` regardless.

  TWO-GATE     One report statement that requires TWO distinct PROSPER_* variables to be armed.
               Arming only the one whose NAME matches your question yields silence that reads as
               data. Measured instances: `[udtail]` needs PROSPER_UD_TAIL_ALIGN *and* a dyntrace
               log (#2146); the unimplemented-NID call-count table needs PROSPER_PROGRESS_UNIMPL
               *and* PROSPER_PROGRESS (#2149's fourth instance, measured live).

  SPLIT-CALL   A callee reached ONLY under env gates, whose call sites do not agree on which gate.
               Some invocations then need a variable the others do not, so arming one switch
               enables the consumers and leaves a producer off. Measured instance: #2111 --
               `record_guest_write(ColorTarget, ...)` sat inside a PROSPER_PROVENANCE_DIM-gated
               function while its three sibling recorders followed writer_provenance_enabled().

## How to read a finding

A finding is a CANDIDATE, not a verdict. The tool is lexical: it sees which variables must be set
for a line to run, not what the printed number MEANS. Judge each against the two questions that
decide it:

  1. If the producer's switch is unset, does the printed field still print? (If it does not print
     at all there is no lie -- an absent field invites the question, a `0` answers it wrongly.)
  2. Is the value it then carries indistinguishable from a real measurement?

A finding where both are yes is the defect. A finding where the field is structural (an index, a
size, a mode) rather than a *count of things observed* is usually benign -- record that judgement
in the baseline rather than deleting the rule.

## Baseline, and why this fails on NEW findings only

Every finding must appear in `diag_gate_baseline.txt` with a classification and an issue link.
The check FAILS on:

  - a finding not in the baseline  -> a newly introduced split gate, fix it or classify it;
  - a baseline entry that no longer reproduces -> the fix landed, delete the row.

The second half matters as much as the first: an audit nobody is forced to update is one that
silently stops describing the tree, which is the same failure class this tool exists to catch.

## What this CANNOT see -- read before quoting a clean run

  - It is a lexical scanner, not a compiler. It tracks braces, `if` conditions and early-return
    guards; it does not resolve overloads, templates, macros, or values.
  - Cross-translation-unit STATE is invisible. A producer that fills a global in one file, printed
    from another, is only caught if the producer is reached through a call whose gates disagree
    (SPLIT-CALL). A shared counter written and read under different gates in different files is
    NOT detected.
  - A disjunction is treated as ungated: `if (getenv("A") || flag)` imposes no requirement, and
    `if (getenv("A") || getenv("B"))` imposes none either. That understates deliberately -- it is
    the direction that produces no false accusation.
  - A NEGATED env test imposes nothing, for the same reason: `if (!getenv("X"))` and the opt-out
    `getenv("PROSPER_NO_X") == nullptr` both run on a default boot. Corollary, and a real gap:
    `if (getenv("X")) return;` genuinely does make the remainder require X to be UNSET, which is a
    coupling this tool has no way to express and therefore does not report.
  - An argument list is read as one conjunct, so `f(getenv("A") != nullptr, limit_from_B)` yields
    the any-of clause `{A|B}` rather than the two independent parameters it is. That widens a
    clause, which is the understating direction for TWO-GATE.
  - A braceless `if` body is not scoped: `if (a) if (b) x = c;` attributes `x` to the enclosing
    block. LITERAL_RHS_RE exists because of this -- see the defaulted-alias rule in run().
  - `#if`-gated code is scanned as if it were live.
  - A diagnostic gated by something other than an environment variable (a build flag, a member
    field, a runtime setting) is out of scope by construction.

So a clean run means "no NEW instance of three specific lexical shapes", never "no diagnostic in
this tree can print an unmeasured field".

Run standalone against a checkout, or via ctest as `diag_gate_selftest` (--selftest arm).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SCAN_DIRS = ("src", "frontends", "tools", "tests")
SCAN_EXT = (".c", ".cc", ".cpp", ".h", ".hpp")

# The four classifications a baseline row may carry. `unreviewed` is honest debt; the other three
# are judgements, and the JUDGEMENT is the expensive artifact in that file -- each cost a human
# reading a diagnostic and deciding whether its zero can lie. See baseline_integrity().
BASELINE_CLASSES = ("defect", "config-echo", "benign", "unreviewed")

# How many `unreviewed` rows the baseline is allowed to carry -- asserted EXACTLY, not as a ceiling.
#
# Exactly, because the two directions are different events and both want a human:
#   more than this -> debt was ADDED. `--emit-baseline` writes EVERY row as `unreviewed`, so this
#     is precisely what "just regenerate the baseline" looks like after it has laundered a `defect`
#     row into an unjudged one. That is the failure this whole file exists to prevent, and until
#     now nothing in the tool could see it: load_baseline() kept the note only so `--list` could
#     echo it, and the header's class counts were skipped as comments.
#   fewer than this -> debt was PAID. Tightening the number is then a one-line, visible, reviewable
#     edit, and the ratchet is self-arming: it can only be loosened deliberately and in the diff.
# #2572 annotated the remaining rows and this is now 0: no unjudged row can enter the baseline
# again, and a `--emit-baseline` regeneration -- which classifies every row `unreviewed` -- fails on
# the first row it writes rather than after it has laundered 54 judgements.
UNREVIEWED_BUDGET = 0

ENV_NAME = r"PROSPER_[A-Z0-9_]+"
# Every way this tree asks for a PROSPER_* variable. `env_enabled("X")` and friends pass the name
# as an argument, so anchor on the LITERAL rather than on a fixed list of callee names -- the same
# reasoning check_cached_env.py records for its wrapper forms.
ENV_LITERAL_RE = re.compile(r'"(' + ENV_NAME + r')"')

PRINTF_RE = re.compile(r"\b(?:fprintf|printf|snprintf|std::fprintf|std::printf)\s*\(")

# A function definition at file scope: `type name(args) {`. Loose on the return type, strict on
# the tail -- a call ends in `;`, a definition in `{`.
FUNC_DEF_RE = re.compile(
    r"^[A-Za-z_][\w:<>,\*&\s\[\]]*?\b([A-Za-z_]\w*)\s*\([^;]*\)\s*"
    r"(?:const\s*)?(?:noexcept\s*)?(?:->[\w:<>\*&\s]+)?\{\s*$"
)

def is_func_def(line: str):
    """`type name(args) {` with BALANCED parens.

    The balance check is not pedantry: `std::erase_if(resources, [&](const R& r) {` matches the
    shape without it, and a lambda call read as a function definition silently re-scopes every
    local below it.
    """
    m = FUNC_DEF_RE.match(line)
    if not m:
        return None
    head = line[:m.end()]
    if head.count("(") != head.count(")"):
        return None
    return m


NOT_A_CALL = {
    "if", "for", "while", "switch", "return", "sizeof", "catch", "throw", "case",
    "and", "or", "not", "defined", "static_cast", "reinterpret_cast", "const_cast",
    "dynamic_cast", "decltype", "alignof", "noexcept", "typeid", "else", "do",
}

# RHS values that mean "unset". A declaration initialised to one of these is the DEFAULT the
# reader mistakes for a measurement, so it does not count as a write. This exclusion IS the
# SPLIT-LOCAL signature, not a convenience.
DEFAULT_RHS_RE = re.compile(
    r"^\s*(?:0|0u|0U|0ul|0ull|0ULL|0x0+|0\.0f?|false|nullptr|NULL|\{\s*\}|\"\"|-1)\s*$")

IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b")
CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

# A declaration statement of its own: `[qualifiers] type name = rhs;`. The leading character
# class excludes `(`, so a `for (int i = 0; ...)` header cannot match.
DECL_RE = re.compile(
    r"^(?:static\s+|const\s+|constexpr\s+|thread_local\s+|inline\s+|unsigned\s+|signed\s+|auto\s+)*"
    r"(?:[A-Za-z_][\w:<>,\*&\s\[\]]*?\s[\*&\s]*)?([A-Za-z_]\w*)\s*=\s*(.*)$")

ASSIGN_RE = re.compile(
    r"^[\w:<>\*&\s\[\]\.\-]*?\b([A-Za-z_]\w*)\s*(?:=(?!=)|\+=|-=|\|=|&=|\^=)\s*(.*)$")

NEGATIVE_TEST_RE = re.compile(r"(?:^\s*!|!\s*[A-Za-z_(]|==\s*(?:nullptr|NULL|0)\b|<=\s*0\b|"
                              r"==\s*0\b|\.empty\(\)|!\*)")

# A comparison that follows an env read and INVERTS it. `getenv("X") == nullptr` is this tree's
# opt-OUT idiom -- it is TRUE when X is unset -- and `<= 0` is the same thing for a numeric read.
INVERTING_TAIL_RE = re.compile(r"^\s*(?:==\s*(?:nullptr|NULL|0)\b|<=\s*0\b|<\s*1\b)")

# A lambda introducer at the head of an initialiser: `[&](args) {`, `[] {`, `[=]() -> bool {`.
LAMBDA_INTRO_RE = re.compile(r"^\s*\[[^\]\[]*\]\s*(?:\(|\{|mutable\b|constexpr\b|noexcept\b|->)")

# `name =` with the compound and comparison forms excluded: `+=`/`!=`/`<=`/`==` all fail to match
# because the character before `=` is then not part of the identifier and not whitespace.
ASSIGN_TO_NAME_RE = re.compile(r"\b([A-Za-z_]\w*)\s*=(?!=)")

# A COMPILED-IN CONSTANT -- the thing a "supply a default" statement assigns. Deliberately narrower
# than "any value not derived from the environment": a computed RHS (`resdump = strtoull(fa, ...) ==
# code_addr;`, gpu_executor.cpp:6417) sits under a braceless `if` whose condition this scanner does
# not scope, so treating it as a default would retire the alias on a path that is not actually
# reachable by default -- and that row is a live two-gate finding. A literal cannot be conditional
# on anything the scanner missed. (`""` is what strip_comments() leaves of any non-PROSPER_ string.)
LITERAL_RHS_RE = re.compile(
    r"^\s*-?(?:\d+(?:\.\d*)?[uUlLfF]*|0[xX][0-9a-fA-F]+[uUlL]*"
    r"|true|false|nullptr|NULL|\"\"|''|\{\s*\})\s*$")


# --------------------------------------------------------------------------------------------
# Lexing
# --------------------------------------------------------------------------------------------
def strip_comments(text: str) -> list[str]:
    """Per-line code with comments blanked and literals neutralised, PROSPER_* names kept.

    Line count is preserved so every finding cites a real file:line. Braces and parens inside
    string literals are blanked -- otherwise a printed `{` corrupts depth for the rest of a file.
    """
    out = []
    in_block = False
    for line in text.split("\n"):
        buf, i, n = [], 0, len(line)
        while i < n:
            two = line[i:i + 2]
            if in_block:
                if two == "*/":
                    in_block = False
                    i += 2
                else:
                    i += 1
                continue
            if two == "/*":
                in_block = True
                i += 2
                continue
            if two == "//":
                break
            ch = line[i]
            if ch in ('"', "'"):
                quote = ch
                j = i + 1
                while j < n:
                    if line[j] == "\\":
                        j += 2
                        continue
                    if line[j] == quote:
                        j += 1
                        break
                    j += 1
                lit = line[i:j]
                if quote == '"' and "PROSPER_" in lit:
                    buf.append(re.sub(r"[{}()]", " ", lit))
                else:
                    buf.append('""' if quote == '"' else "''")
                i = j
                continue
            buf.append(ch)
            i += 1
        out.append("".join(buf))
    return out


def env_literals(expr: str) -> set[str]:
    return set(ENV_LITERAL_RE.findall(expr))


# --------------------------------------------------------------------------------------------
# Polarity
#
# A gate is "X must be SET". An env test that is NEGATED says the opposite, and this tree writes
# far more of the negated form than the plain one -- `PROSPER_NO_*` is the standard opt-OUT idiom,
# so `getenv("PROSPER_NO_RESOLVE") == nullptr` is TRUE on a default run and imposes nothing.
# Reading those as requirements was the single commonest reason a finding was not a real coupling
# (#2572 measured 47 of 79 reviewed rows, with this as the largest cluster; #2628).
#
# The negation was ALREADY pushed through for the early-return form (guard_gates()), so before this
# the two spellings of one idea disagreed with each other:
#
#     if (!getenv("X")) return;        -- the remainder requires X    (correct)
#     if (!getenv("X")) { ... }        -- the block required X        (backwards)
#
# It is done per LITERAL rather than per operand on purpose. A whole-operand test (the shape
# NEGATIVE_TEST_RE encodes for guard_gates) is applied to text that is sometimes an entire function
# body -- collect_predicates() calls clauses_of() on one -- where a single `.empty()` or `== 0`
# anywhere would retire every gate in it. Per-literal, `getenv("A") && !getenv("B")` correctly
# yields just {A}.
# --------------------------------------------------------------------------------------------
def _matching_close(expr: str, open_pos: int) -> int:
    depth = 0
    for pos in range(open_pos, len(expr)):
        if expr[pos] == "(":
            depth += 1
        elif expr[pos] == ")":
            depth -= 1
            if depth == 0:
                return pos
    return -1


def _enclosing_open(expr: str, idx: int) -> int:
    """Position of the nearest `(` to the left of idx that is still open at idx, or -1."""
    depth, j = 0, idx - 1
    while j >= 0:
        ch = expr[j]
        if ch == ")":
            depth += 1
        elif ch == "(":
            if depth == 0:
                return j
            depth -= 1
        j -= 1
    return -1


_MAX_PAREN_LEVELS = 8


def _is_negated(expr: str, idx: int) -> bool:
    """Does an ODD number of inversions apply to the env literal at `idx`?

    Walks outward through the paren groups that enclose the literal. At each level two inversions
    are recognised, and both must be, because expand() rewrites an alias into a nested paren group:
    `!profile_fold` becomes `!((getenv("PROSPER_STAGE_FOLD_PROFILE")))`, where the `!` is three
    levels out from the literal.

      - a `!` before the callee that opens the group -- `!getenv(`, `!*getenv(`, `!(`;
      - an inverting comparison after the group closes -- `getenv("X") == nullptr`.
    """
    negated, i, levels = False, idx, 0
    while levels < _MAX_PAREN_LEVELS:
        open_pos = _enclosing_open(expr, i)
        if open_pos < 0:
            break
        levels += 1
        k = open_pos - 1
        while k >= 0 and expr[k].isspace():
            k -= 1
        while k >= 0 and (expr[k].isalnum() or expr[k] in "_:"):   # the callee name
            k -= 1
        while k >= 0 and (expr[k].isspace() or expr[k] == "*"):    # `!*getenv(...)`
            k -= 1
        # `expr[k-1] not in "!=<>"` keeps `a != getenv(...)` and `!!x` from reading as one negation.
        if k >= 0 and expr[k] == "!" and (k == 0 or expr[k - 1] not in "!=<>"):
            negated = not negated
        close = _matching_close(expr, open_pos)
        if close >= 0 and INVERTING_TAIL_RE.match(expr[close + 1:]):
            negated = not negated
        i = open_pos
    return negated


def positive_env_literals(expr: str) -> set[str]:
    """The PROSPER_* names this expression requires to be SET for it to be true."""
    return {m.group(1) for m in ENV_LITERAL_RE.finditer(expr)
            if not _is_negated(expr, m.start())}


def split_ternary(expr: str) -> tuple[str, str, str] | None:
    """`cond ? then : else` split at nesting depth 0, or None.

    Brackets and braces count as well as parens, so the `return e ? atol(e) : 0;` inside an
    immediately-invoked lambda is NOT seen as the initialiser's own ternary.
    """
    expr = unwrap(expr)
    depth, q = 0, -1
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and ch == "?":
            if q < 0:
                q = i
        elif depth == 0 and ch == ":" and q >= 0:
            if expr[i:i + 2] == "::" or (i and expr[i - 1] == ":"):
                i += 2
                continue
            return expr[:q], expr[q + 1:i], expr[i + 1:]
        i += 1
    return None


def unwrap(expr: str) -> str:
    """Drop a paren pair that encloses the WHOLE expression.

    `if (...)` hands the condition over with its own parens still on, so without this every
    top-level `||` sits at depth 1 and never splits -- which silently turns `A || B` into
    `A and B` and reports a two-gate requirement that does not exist. That was a real failure of
    this scanner's first revision, caught only by the must-not-match case for disjunction.
    """
    expr = expr.strip()
    while len(expr) >= 2 and expr[0] == "(" and expr[-1] == ")":
        depth = 0
        for pos, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and pos != len(expr) - 1:
                    return expr
        expr = expr[1:-1].strip()
    return expr


def split_top_level(expr: str, op: str) -> list[str]:
    """Split on `op` (`||` or `&&`) at paren depth 0."""
    expr = unwrap(expr)
    parts, buf, depth, i = [], [], 0, 0
    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and expr[i:i + 2] == op:
            parts.append("".join(buf))
            buf = []
            i += 2
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


# --------------------------------------------------------------------------------------------
# Gate aliases
#
# Without these the scanner sees `if (udprov_enabled())` and `if (interval <= 0) return;` as
# ungated, and every measured instance hides behind exactly such a name -- an aliasless version of
# this tool reports a clean tree by construction. But an alias table that is GLOBAL over the tree
# is worse than none: single-letter locals then stand for whatever variable was declared near an
# unrelated getenv, and the first revision of this tool reported 719 findings that way. So:
#
#   global scope -- predicate FUNCTIONS only (`bool udprov_enabled() { ... getenv(...) ... }`),
#                   which genuinely mean the same thing everywhere;
#   block scope  -- declarations, resolved during the walk and popped with their brace.
# --------------------------------------------------------------------------------------------
PRED_MAX_LINES = 12


def clauses_of(expr: str) -> frozenset:
    """The requirement an expression imposes, as a CONJUNCTION OF DISJUNCTIONS.

    A flat set of names cannot express this tree's actual gates and gets both directions wrong:

        getenv(A) && getenv(B)   -- BOTH needed          -> two clauses {A}, {B}
        getenv(A) || getenv(B)   -- EITHER suffices      -> one clause {A, B}
        getenv(A) || flag        -- neither is required  -> no clause
        getenv(A) && !getenv(B)  -- only A is required   -> one clause {A}
        getenv(A) || !getenv(B)  -- neither is required  -> no clause

    Read flat, the second reads as "both required" and manufactures a two-gate finding that does
    not exist -- 133 of the first tree run's findings came from that, including one predicate the
    scanner claimed required all sixteen of its alternatives.

    The last two lines are polarity (#2628): a negated env test is satisfied by the DEFAULT run, so
    it constrains nothing, and a disjunct that is satisfiable with nothing set leaves the whole
    disjunction unconstrained. See positive_env_literals().
    """
    disjuncts = split_top_level(expr, "||")
    per: list[set[str]] = [positive_env_literals(d) for d in disjuncts]
    if len(disjuncts) == 1:
        out = set()
        for operand in split_top_level(disjuncts[0], "&&"):
            names = positive_env_literals(operand)
            if names:
                out.add(frozenset(names))
        return frozenset(out)
    if all(per):
        # Every branch needs SOME variable, but any one of them suffices.
        union: set[str] = set()
        for s in per:
            union |= s
        return frozenset({frozenset(union)})
    # One branch is reachable with nothing set, so nothing is required.
    return frozenset()


def simplify(ctx) -> frozenset:
    """Drop clauses implied by a narrower one (CNF absorption).

    `{UNIMPL} AND {PROGRESS or UNIMPL}` is just `{UNIMPL}` -- satisfying the first satisfies the
    second. Without this, a function that opens `if (a <= 0 && !b) return;` and then narrows to
    `if (b)` looks like a two-gate requirement when the second gate is the same variable seen
    twice, and the tool reports a defect it has itself just watched get fixed.
    """
    clauses = sorted(set(ctx), key=len)
    out = []
    for i, c in enumerate(clauses):
        if any(other < c for other in clauses[:i]):
            continue
        out.append(c)
    return frozenset(out)


def expand_with(aliases: dict[str, frozenset], expr: str) -> str:
    """Splice each alias back in as an explicit `(A || B) && (C)` so clauses_of() can read it."""
    def sub(m):
        ctx = aliases.get(m.group(0))
        if not ctx:
            return m.group(0)
        return "(" + " && ".join(
            "(" + " || ".join(f'getenv("{n}")' for n in sorted(cl)) + ")"
            for cl in sorted(ctx, key=lambda c: sorted(c))) + ")"
    return re.sub(r"\b[A-Za-z_]\w*\b", sub, expr)


def guard_clauses(cond: str, expand) -> frozenset:
    """For `if (COND) return;` -- what the REST requires, i.e. the gates of !COND.

    The negation has to be pushed through, and getting that backwards inverts the answer:

        if (!a || !b) return;      remainder needs a AND b   -> two clauses
        if (a <= 0 && !b) return;  remainder needs a OR b    -> one any-of clause

    The second form is the one that mattered: reading it as "both" made the scanner keep reporting
    the very instance #2149's fix had just repaired, because the repaired function opens with
    exactly `if (interval <= 0 && !dump_unimpl) return;`. A checker that cannot see its own fix
    land is worse than none -- it teaches people to regenerate past it.

    Only NEGATIVE tests contribute. The negation of a positive test (`if (GATE) return;`) says the
    remainder needs GATE unset, which is not a gate; and any operand whose negation is not an env
    fact leaves its whole disjunct unconstrained.
    """
    out: set = set()
    for disjunct in split_top_level(cond, "||"):
        clause: set = set()
        unconstrained = False
        for operand in split_top_level(disjunct, "&&"):
            if not NEGATIVE_TEST_RE.search(operand):
                unconstrained = True
                break
            names = env_literals(expand(operand))
            if not names:
                unconstrained = True
                break
            clause |= names
        if not unconstrained and clause:
            out.add(frozenset(clause))
    return frozenset(out)


def split_statements(text: str) -> list[str]:
    """Split on `;` at nesting depth 0."""
    out, buf, depth = [], [], 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ";" and depth <= 0:
            out.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if "".join(buf).strip():
        out.append("".join(buf))
    return out


IF_RETURN_RE = re.compile(r"^\s*if\s*\((.*)\)\s*\{?\s*return\b(.*)$", re.S)
RETURN_RE = re.compile(r"^\s*return\b(.*)$", re.S)


def predicate_clauses(body_text: str) -> frozenset:
    """The requirement that calling a short bool/int predicate imposes.

    This used to be `clauses_of()` over the whole joined body, which happened to work only because
    polarity was ignored: every `PROSPER_*` literal anywhere in the body became an alternative of
    one any-of clause. Once negation is modelled (#2628) that reading is actively wrong in BOTH
    directions at once, and `dyntrace_failed_shader_enabled` (gpu_executor.cpp:401) shows both:

        if (!std::getenv("PROSPER_DYNTRACE_FAIL")) return false;   // a GUARD -- FAIL is required
        const char* filter = std::getenv("PROSPER_DYNTRACE_FAIL_ADDR");
        if (!filter) return true;                                  // early SUCCESS -- optional

    A flat any-of read said `FAIL|FAIL_ADDR`; a flat polarity read would have said `FAIL_ADDR` and
    dropped the one variable that IS required, silently weakening a `defect` row's key. A predicate
    body is a function body, so it is read as one: guards impose their negation (the same rule
    guard_clauses() already applied to early returns in ordinary code), an early truthy return ends
    the accumulation, and a declaration is an alias rather than a requirement of its own.

    A body in which NO statement is recognised falls back to the flat reading, so a `static int v =
    -1; if (v < 0) { ... }` memo (`audiolog_level`, hle_audio.cpp:327) keeps its gate instead of
    quietly resolving to nothing -- the direction that disables a rule.

    Statements this does not recognise contribute nothing -- the understating direction, which is
    the one that makes no false accusation.
    """
    body = body_text[body_text.find("{") + 1:] if "{" in body_text else body_text
    aliases: dict[str, frozenset] = {}
    ctx: set = set()            # required on every path that can return true
    alts: list[frozenset] = []  # one per `if (COND) return <truthy>;` early exit
    tail: set = set()           # what the path falling through all of them requires
    recognised = False

    def exp(expr: str) -> str:
        return expand_with(aliases, expr)

    for stmt in split_statements(body):
        m = IF_RETURN_RE.match(stmt)
        if m:
            recognised = True
            cond = unwrap(m.group(1))
            value = m.group(2).strip().split(";")[0].strip()
            if value and not DEFAULT_RHS_RE.match(value):
                alt = clauses_of(exp(cond))
                if not alt:
                    # `if (!filter) return true;` -- true with nothing further armed, so the guards
                    # already collected are the whole necessary condition.
                    return simplify(ctx)
                alts.append(alt)
            else:
                (tail if alts else ctx).update(guard_clauses(cond, exp))
            continue
        m = RETURN_RE.match(stmt)
        if m:
            recognised = True
            (tail if alts else ctx).update(clauses_of(exp(m.group(1))))
            continue
        dm = DECL_RE.match(stmt.strip())
        if dm:
            got = clauses_of(exp(dm.group(2)))
            if got:
                aliases[dm.group(1)] = got

    if not recognised:
        return simplify(set(clauses_of(body)))
    if alts:
        # Several exits can return true, so what is NECESSARY is the disjunction of their
        # requirements -- `report_selected_sbuffer_reject` (rdna2_gta5_compute_contracts.cpp:39)
        # returns early under PROSPER_DBG and otherwise needs PROSPER_GTA5_SBUFFER_REJECT, so
        # either one alone reaches the report. Expressible only when every branch is a single
        # clause; otherwise the disjunction is dropped, which understates.
        branches = alts + [frozenset(tail)]
        if all(len(b) == 1 for b in branches):
            names: set[str] = set()
            for b in branches:
                names |= set(next(iter(b)))
            ctx.add(frozenset(names))
    return simplify(ctx)


def collect_predicates(files: dict[Path, list[str]]) -> dict[str, frozenset]:
    """name -> the requirement calling it imposes, for SHORT bool/int predicate functions.

    Three constraints, each earned by a wrong answer from the version without it:
      - the body must CLOSE inside PRED_MAX_LINES; otherwise the window ran past the function and
        collected its neighbours' variables (`writer_provenance_enabled` came out needing three
        `*_DIM` variables that appear nowhere in it);
      - `main` and other long functions are therefore excluded automatically -- the first version
        had `main` standing for 47 variables, unioned across every tool that defines one;
      - a name defined more than once must AGREE everywhere, or it is dropped. Two files with
        different `avp_log()` bodies do not share a meaning, and guessing one is how an alias
        table starts inventing gates.
    """
    seen: dict[str, list[frozenset]] = {}
    for _path, lines in files.items():
        for idx, line in enumerate(lines):
            m = is_func_def(line.strip())
            if not m:
                continue
            if not re.match(r"^\s*(?:static\s+|inline\s+|constexpr\s+)*(?:bool|int)\b", line):
                continue
            body, depth, closed = [], 0, False
            for probe in lines[idx:idx + PRED_MAX_LINES]:
                body.append(probe)
                depth += probe.count("{") - probe.count("}")
                if depth <= 0 and "{" in " ".join(body):
                    closed = True
                    break
            if not closed:
                continue
            text = " ".join(body)
            if not env_literals(text):
                continue
            seen.setdefault(m.group(1), []).append(predicate_clauses(text))
    return {name: variants[0] for name, variants in seen.items()
            if variants[0] and all(v == variants[0] for v in variants)}


class Scope:
    """One open brace: the gates it imposes, the aliases and the default-initialised locals in it.

    SPLIT-LOCAL is evaluated when a scope CLOSES, because a local's lifetime is its brace, not its
    function. Anchoring it on function detection instead made the rule inert everywhere under
    `namespace { }` and, where it did fire, paired a print with a declaration 250 lines BELOW it.
    """

    def __init__(self, gates: set[str]):
        self.gates = set(gates)
        self.aliases: dict[str, frozenset] = {}
        # var -> (decl line, [(line, gates)] writes, [(line, gates)] prints)
        self.decls: dict[str, tuple] = {}


# --------------------------------------------------------------------------------------------
class Finding:
    def __init__(self, kind: str, where: str, subject: str, gates, detail: str):
        self.kind = kind
        self.where = where
        self.subject = subject
        # A set of clauses; each clause is a frozenset of names, any of which satisfies it.
        self.gates = frozenset(gates)
        self.detail = detail

    @staticmethod
    def render(ctx) -> str:
        """`A+B` = both required. `A|B` = either suffices. The distinction is the whole point.

        Long any-of clauses are elided: this tree has a "some diagnostic dump is on" predicate
        with 28 alternatives, and printing all of them buries the conjunction -- which is the part
        a reader has to see.
        """
        out = []
        for cl in ctx:
            names = sorted(cl)
            out.append("|".join(names) if len(names) <= 3
                       else f"{names[0]}|{names[1]}|..{len(names) - 2}more")
        return "+".join(sorted(out))

    @staticmethod
    def render_key(ctx) -> str:
        """The requirement AS IDENTITY. Long alternations collapse to `(any)` -- no membership.

        `render()` is presentation and may elide with a count; identity must not, and the two must
        not share a function. The count IS membership: this tree has a 28-alternative "some
        diagnostic dump is on" predicate, and with `..23more` inside the key, adding ONE
        `PROSPER_DUMP_*` variable to it re-keys every finding that depends on it -- measured on the
        first baseline, 13 rows, all in one file, all of them changing together. That would fail
        the Docs job on a PR that did nothing wrong, pointing at a baseline table rather than at
        the cause, and the fix would look like "just regenerate the baseline" -- which is exactly
        how a `defect` row silently becomes an `unreviewed` one.

        The principle: **the conjunction is identity; a long alternation is presentation.** That is
        the same argument render()'s own docstring makes for eliding it -- printing all of them
        buries the conjunction, which is the part a reader has to see.

        The residual, stated rather than papered over: the threshold is a boundary, so a clause
        crossing 3 -> 4 names DOES re-key. That is a genuine loosening of a small, deliberate
        alternation and is worth re-reading; the hazard this guards against is an unrelated name
        joining a bulk one.
        """
        out = []
        for cl in ctx:
            names = sorted(cl)
            out.append("|".join(names) if len(names) <= 3 else "(any)")
        return "+".join(sorted(out))

    def key(self) -> str:
        """Baseline identity: no line number, and no membership of a long alternation.

        A line-numbered key churns on every unrelated edit above the finding, and each churn shows
        up twice -- once as a new finding, once as a stale baseline row. In a tree several lanes
        edit at once that turns the gate into noise people learn to regenerate past. File +
        subject + requirement is stable under code motion and still distinguishes two different
        couplings in one file; see render_key() for the second churn door and why it is shut.
        """
        return f"{self.kind}|{self.where.rsplit(':', 1)[0]}|{self.subject}|{self.render_key(self.gates)}"

    def line(self) -> str:
        return f"{self.key()}  @{self.where}  {self.detail}"


class FileScanner:
    def __init__(self, rel: str, lines: list[str], preds: dict[str, frozenset],
                 call_sites: dict[str, list[tuple[str, frozenset]]],
                 printing: set[str] | None = None,
                 flags: dict[str, frozenset] | None = None,
                 flag_writes: dict[str, list[frozenset]] | None = None):
        self.rel = rel
        self.lines = lines
        self.preds = preds
        self.call_sites = call_sites
        self.printing = printing or set()
        self.flags = flags or {}
        self.flag_writes = flag_writes
        self.findings: list[Finding] = []
        self.stack: list[Scope] = [Scope(set())]

    # -- gate helpers ------------------------------------------------------------------------
    # A "clause" is a frozenset of variable names, ANY of which satisfies it. A gate context is a
    # set of clauses, ALL of which must be satisfied. Two clauses in one context is the TWO-GATE
    # defect; sixteen names in ONE clause is an ordinary any-of switch and is not.
    def active(self) -> frozenset:
        ctx: set = set()
        for s in self.stack:
            ctx |= s.gates
        return simplify(ctx)

    def expand(self, expr: str) -> str:
        """Substitute the PROSPER_* names an identifier stands for, so clauses_of() sees them.

        A predicate that stands for a disjunction is spliced in as `(A || B)`, keeping its any-of
        shape; substituting a bare list would turn `if (udprov_enabled())` into a demand for both
        of its alternatives.
        """
        def sub(m):
            ident = m.group(0)
            ctx = self.preds.get(ident)
            if ctx is None:
                ctx = self.flags.get(ident)
            if ctx is None:
                for s in reversed(self.stack):
                    if ident in s.aliases:
                        ctx = s.aliases[ident]
                        break
            if not ctx:
                return ident
            return "(" + " && ".join(
                "(" + " || ".join(f'getenv("{n}")' for n in sorted(cl)) + ")"
                for cl in sorted(ctx, key=lambda c: sorted(c))) + ")"
        return re.sub(r"\b[A-Za-z_]\w*\b", sub, expr)

    @staticmethod
    def alias_is_a_gate(rhs: str) -> bool:
        """Is this initialiser's VALUE the thing an env switch decides? Two shapes say no. #2628.

        1. A STORED LAMBDA. `auto readable = [&](uint64_t a, uint32_t n) { ... }` holds a callable,
           not a value, so the `PROSPER_*` names in its body belong to the branch inside it -- but
           the block-scope alias path resolved them anyway and exported the requirement to every
           later `readable(...)` CALL, invisibly at the call site. `readable`
           (gpu_executor.cpp:2854) and `build_bds` (live_renderer.cpp:5602) are the measured
           instances; collect_predicates() already declines to alias a whole function for exactly
           this reason and the lambda path applied no such restriction.
           An IMMEDIATELY-INVOKED lambda is the opposite case and stays: `static const long
           interval = [] { ... getenv("PROSPER_PROGRESS") ...; }();` really does hold a value the
           environment decides, and #2149's fourth measured instance depends on it resolving.
           The two are told apart by the trailing `()`.

        2. A TERNARY WITH A LIVE DEFAULT. `X ? atoi(X) : 60` yields 60 when X is unset, so the
           value exists either way and testing it later is not an env gate. `X ? atol(X) : 0` is
           NOT that: DEFAULT_RHS_RE is the tool's existing statement of which literals mean
           "unset", and a falsy default keeps the alias -- which is what makes `if (interval <= 0)
           return;` still impose PROSPER_PROGRESS.
        """
        body = rhs.strip().rstrip(";").strip()
        if LAMBDA_INTRO_RE.match(body):
            return body.endswith(")")          # invoked -> a value; stored -> a callable
        parts = split_ternary(body)
        if parts is not None:
            fallback = parts[2].strip()
            if fallback and not DEFAULT_RHS_RE.match(fallback) and "PROSPER_" not in fallback:
                return False
        return True

    def required(self, cond: str) -> frozenset:
        """The gate context a condition imposes on the block it opens."""
        return clauses_of(self.expand(cond))

    def guard_gates(self, cond: str) -> frozenset:
        """For `if (COND) return;` -- what the REST of the block requires. See guard_clauses()."""
        return guard_clauses(cond, self.expand)

    # -- the walk ----------------------------------------------------------------------------
    def note_local(self, var: str, line_no: int, gates: frozenset, is_write: bool):
        """Attribute a write or a print to the scope that DECLARED the variable."""
        for s in reversed(self.stack):
            if var in s.decls:
                (s.decls[var][1] if is_write else s.decls[var][2]).append((line_no, gates))
                return

    def close_scope(self, scope: Scope):
        """Evaluate SPLIT-LOCAL for the locals this scope declared, then drop them."""
        for var, (dline, writes, prints) in scope.decls.items():
            if not writes or not prints:
                continue
            common = set(simplify(writes[0][1]))
            for _ln, g in writes[1:]:
                common &= set(simplify(g))
            if not common:
                continue
            for pln, pg in prints:
                # A clause is only MISSING if nothing at the print site implies it. `{PROGRESS}`
                # implies `{PROGRESS or UNIMPL}`, so a plain set difference reports a gap that
                # cannot occur -- the print is unreachable without satisfying the write's gate.
                missing = frozenset(c for c in common
                                    if not any(d <= c for d in simplify(pg)))
                if missing:
                    self.findings.append(Finding(
                        "SPLIT-LOCAL", f"{self.rel}:{pln}", var, missing,
                        f"declared with a default at :{dline}, written only under "
                        f"{Finding.render(common)}, printed without it"))
                    break

    def gather_statement(self, i: int, start: int) -> tuple[str, int]:
        """Accumulate from `start` on line i to the `;` that ENDS the statement.

        Depth-aware, because the initialisers that matter here are immediately-invoked lambdas:

            static const long interval = [] {
                const char* e = getenv("PROSPER_PROGRESS"); return e ? atol(e) : 0; }();

        A fixed line window instead of this swallowed the NEXT declaration too, and the alias for
        `interval` came out standing for `PROSPER_PROGRESS` *or* `PROSPER_PROGRESS_UNIMPL` -- which
        merges the exact two switches whose separation is the finding.
        """
        buf, depth, k = [], 0, i
        pos = start
        while k < len(self.lines) and k - i < 24:
            text = self.lines[k]
            while pos < len(text):
                ch = text[pos]
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                elif ch == ";" and depth <= 0:
                    buf.append(";")
                    return "".join(buf), k - i + 1
                buf.append(ch)
                pos += 1
            buf.append(" ")
            k += 1
            pos = 0
        return "".join(buf), min(k - i + 1, 24)

    def gather(self, i: int, start: int) -> tuple[str, int]:
        """Accumulate a parenthesised clause starting at `start` on line i. -> (text, lines used)."""
        buf = self.lines[i][start:]
        k = i
        while buf.count("(") > buf.count(")") and k + 1 < len(self.lines):
            k += 1
            buf += " " + self.lines[k]
        depth, cut = 0, len(buf)
        for pos, ch in enumerate(buf):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    cut = pos + 1
                    break
        return buf[:cut], k - i + 1

    def run(self) -> list[Finding]:
        n = len(self.lines)
        i = 0
        while i < n:
            raw = self.lines[i]
            line = raw.strip()
            consumed = 1

            # ---- condition of an `if` on this line -----------------------------------------
            cond, tail = None, ""
            if re.search(r"\bif\s*\(", line):
                start = raw.find("(", raw.find("if"))
                if start >= 0:
                    cond, used = self.gather(i, start)
                    consumed = max(consumed, used)
                    joined = " ".join(self.lines[i:i + used])
                    tail = joined[joined.find(cond) + len(cond):] if cond in joined else ""

            here = self.active()
            cond_gates = self.required(cond) if cond else set()
            is_def = is_func_def(line) is not None

            # ---- call sites (SPLIT-CALL input) ---------------------------------------------
            if not is_def:
                for cm in CALL_RE.finditer(line):
                    callee = cm.group(1)
                    if callee in NOT_A_CALL:
                        continue
                    self.call_sites.setdefault(callee, []).append(
                        (f"{self.rel}:{i + 1}", frozenset(here | cond_gates)))

            # ---- TWO-GATE ------------------------------------------------------------------
            pm = PRINTF_RE.search(line)
            reports = bool(pm) or (not is_def and any(
                cm.group(1) in self.printing and cm.group(1) not in NOT_A_CALL
                for cm in CALL_RE.finditer(line)))
            if reports:
                ctx = simplify(set(here) | set(cond_gates))
                if len(ctx) >= 2:
                    self.findings.append(Finding(
                        "TWO-GATE", f"{self.rel}:{i + 1}", "report", ctx,
                        "one report statement requires all of these to be armed"))

            # ---- SPLIT-LOCAL bookkeeping ---------------------------------------------------
            if not is_def:
                dm = DECL_RE.match(line)
                if dm and DEFAULT_RHS_RE.match(dm.group(2).rstrip(";")):
                    self.stack[-1].decls.setdefault(dm.group(1), (i + 1, [], []))
                elif not line.startswith(("if", "for", "while", "return", "}", "else")):
                    am = ASSIGN_RE.match(line)
                    if am and not DEFAULT_RHS_RE.match(am.group(2).rstrip(";")):
                        self.note_local(am.group(1), i + 1, here, True)
                        if self.flag_writes is not None and GLOBAL_FLAG_RE.match(am.group(1)):
                            self.flag_writes.setdefault(am.group(1), []).append(
                                frozenset(here) | frozenset(cond_gates))
                if pm:
                    args, used = self.gather(i, raw.find("(", raw.find(pm.group(0)[:6])))
                    consumed = max(consumed, used)
                    for ident in IDENT_RE.findall(args):
                        self.note_local(ident, i + 1, here, False)

            # ---- a DEFAULTED alias stops being (only) its own gate -------------------------
            # `const char* out = getenv("X"); if (!out || !*out) out = "shader0.bin";` leaves `out`
            # holding a compiled-in default on exactly the path where X is unset, so every later
            # `if (...out...)` is not gated on X at all (hle_graphics.cpp:1249). The repair sits
            # inside an `if`, so the SPLIT-LOCAL write path never saw it. #2628.
            #
            # The alias is WIDENED rather than dropped, because the second value may itself be
            # reachable only under a different switch -- `if (g_dyntrace_force) resdump = true;`
            # (gpu_executor.cpp:6418) means `resdump` is true under PROSPER_RESDUMP *or* under the
            # dyntrace-failure replay, which is a real any-of requirement and not "no requirement".
            # Dropping it there would have retired a live coupling; widening keeps it and still
            # collapses the `out` case, because the default is assigned inside the alias's own gate
            # so simplify() absorbs the widened clause into it.
            for am2 in ASSIGN_TO_NAME_RE.finditer(line):
                name = am2.group(1)
                value = line[am2.end():].split(";")[0]
                if not LITERAL_RHS_RE.match(value):
                    continue
                for s in reversed(self.stack):
                    if name not in s.aliases:
                        continue
                    at = simplify(set(here) | set(cond_gates))
                    have = s.aliases[name]
                    if len(at) == 1 and len(have) == 1:
                        s.aliases[name] = frozenset(
                            {frozenset(next(iter(have)) | next(iter(at)))})
                    else:
                        # Reachable with nothing armed, or a requirement no single clause can
                        # express: the alias no longer stands for an env switch.
                        del s.aliases[name]
                    break

            # ---- scoped alias declaration --------------------------------------------------
            if cond is None:
                dm2 = DECL_RE.match(line)
                if dm2:
                    eq = raw.find("=", raw.find(dm2.group(1)) + len(dm2.group(1)))
                    rhs, used = self.gather_statement(i, eq + 1) if eq >= 0 else (dm2.group(2), 1)
                    consumed = max(consumed, used)
                    ctx = clauses_of(self.expand(rhs))
                    if ctx and self.alias_is_a_gate(rhs):
                        self.stack[-1].aliases[dm2.group(1)] = ctx

            # ---- brace bookkeeping ---------------------------------------------------------
            text = " ".join(self.lines[i:i + consumed])
            is_else = bool(re.match(r"^\}?\s*else\b", line))

            if cond is not None and re.search(r"\b(return|continue|break|goto)\b", tail) \
                    and "{" not in tail:
                # `if (!GATE) return;` -- the remainder of THIS block now needs GATE. (A positive
                # test, `if (GATE) return;`, leaves a remainder that needs GATE unset, which is
                # not a gate; guard_gates() only collects the negated disjuncts.)
                self.stack[-1].gates |= self.guard_gates(cond)

            # Braces are processed in TEXTUAL order. Popping all closers before pushing all
            # openers looks equivalent and is not: a line that opens and closes -- an immediately
            # invoked lambda, `if (x) { y; }` -- then pops the ENCLOSING scope and pushes a fresh
            # empty one, discarding every alias and gate declared in it. That silently unarmed the
            # `interval` alias two lines after it was declared.
            first_open = True
            for ch in text:
                if ch == "{":
                    gates = cond_gates if (first_open and cond is not None and not is_else) \
                        else frozenset()
                    self.stack.append(Scope(gates))
                    first_open = False
                elif ch == "}":
                    if len(self.stack) > 1:
                        self.close_scope(self.stack.pop())

            i += consumed

        while self.stack:
            self.close_scope(self.stack.pop())
        return self.findings


def implies(a: frozenset, b: frozenset) -> bool:
    """Does satisfying gate context `a` also satisfy `b`? (CNF absorption, one direction.)

    `{A}` implies `{A|B}`: arming A satisfies both. simplify() already collapses that pair inside a
    single context; two SEPARATE contexts were compared by clause-set IDENTITY, so an implication
    read as a disagreement -- `hle_audio.cpp` `audio2log`, called under {AUDIO2LOG} at :1698 and
    under {AUDIO2LOG|AUDIO2_PROBE} at :1901, where the first site's arming satisfies the second.
    #2628.
    """
    return all(any(ca <= cb for ca in a) for cb in b)


def split_call_findings(call_sites: dict[str, list[tuple[str, frozenset]]],
                        defined: set[str]) -> list[Finding]:
    """SPLIT-CALL: a callee reached only under gates whose call sites do not agree.

    Restricted to callees DEFINED in this tree and gated at EVERY site. A helper also reachable
    ungated is ordinary code; a helper reachable only in diagnostic mode, under two disjoint
    switches, is a producer that half its consumers cannot arm.
    """
    out = []
    for callee, sites in sorted(call_sites.items()):
        if callee not in defined or len(sites) < 2:
            continue
        if any(not g for _w, g in sites):
            continue
        distinct = sorted({g for _w, g in sites}, key=lambda s: sorted(s))
        if len(distinct) < 2:
            continue
        pair = None
        for a in distinct:
            for b in distinct:
                if a != b and not (a & b) and not implies(a, b) and not implies(b, a):
                    pair = (a, b)
                    break
            if pair:
                break
        if not pair:
            continue
        where = sorted(w for w, g in sites if g == pair[0])[0]
        out.append(Finding(
            "SPLIT-CALL", where, callee, set(pair[0]) | set(pair[1]),
            f"{len(sites)} call sites, gates disagree: "
            f"{Finding.render(pair[0])} vs {Finding.render(pair[1])}"))
    return out


def is_excluded_dir(name: str) -> bool:
    """Is this ONE directory name a build tree or vendored code? Names only -- never a whole path.

    The intent is "skip the build tree and third_party", and it is a statement about a directory
    INSIDE the scan root. Applying it to the absolute path -- which `"build-" in str(path)` did --
    cannot distinguish that from a scan root that merely SITS under such a directory, and the
    difference is not academic: this scanner's own `--selftest` writes its fixtures into `$TMPDIR`,
    so a `TMPDIR` under `prosper/build-linux/` (the build directory the charter names) excluded
    every fixture and all twelve positive arms reported `got=[]` while the must-not-match arms
    passed vacuously. Three lanes hit that within five minutes (#2658, #2660, #2661).

    `build` is matched as a whole HYPHEN-SEPARATED WORD, not as a substring: `build`, `build-linux`,
    `build-windows`, `cmake-build-debug` and `pre-build` are build trees; `builder/`, `build_utils/`
    and `rebuild-cache/` are directories of real sources and are scanned. A bare `"build-" in name`
    would drop `rebuild-cache/`, and dropping real sources is the SILENT direction: the scan then
    reports a clean tree over a smaller one, which is exactly the failure mode
    `prosper/CMakeLists.txt` says this checker must not have.

    Stated exactly, because an earlier draft of this docstring claimed the word rule "keeps every
    spelling the old rule caught" and that was FALSE in both directions (#2688 review). Against
    master's `"build-" in <absolute path>`, per DIRECTORY component and with nothing above the root:

        newly EXCLUDED   `build/`, `pre-build/`      -- no hyphen after `build`, so the old
                                                        substring never matched them
        newly SCANNED    `rebuild-cache/`            -- contains `build-`, so the old rule dropped it
                         `third_party_shim.cpp`      -- a FILE, and files are no longer tested
        unchanged        `build-linux/`, `cmake-build-debug/`, `third_party/` (excluded);
                         `builder/`, `build_utils/`, `buildinfo.cpp` (scanned -- the old rule
                         never matched these either, whatever the first draft said)

    Both new exclusions are intended: a directory literally named `build` is a build tree, and
    `CLAUDE.md`'s own recipe used exactly that name. Measured over the real
    `prosper/{src,frontends,tools,tests}`, **zero** files differ between the two rules.
    """
    return name == "third_party" or "build" in name.split("-")


def collect_files(root: Path) -> dict[Path, list[str]]:
    files = {}
    for d in SCAN_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in SCAN_EXT or not path.is_file():
                continue
            # DIRECTORY components, RELATIVE to the scan root. Relative so nothing above the root
            # participates; directories only so a source file named `third_party_shim.cpp` is
            # scanned -- it is prosper's own code, and dropping it is the silent direction.
            if any(is_excluded_dir(p) for p in path.relative_to(root).parts[:-1]):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            files[path] = strip_comments(text)
    return files


GLOBAL_FLAG_RE = re.compile(r"^g_[a-z]\w*$")


def collect_printing(files: dict[Path, list[str]]) -> set[str]:
    """Functions that PRINT, so a call to one counts as a report statement.

    Needed because the report is often not an `fprintf` at the gated site. The measured instance
    is `if (dump_unimpl && ...) dump_call_log(stderr);` -- one call, no format string, and a
    printf-only rule cannot see it. The reduced fixture written for that instance DID use an
    fprintf, so it passed while the real line went unnoticed: a reduction that removes the very
    property being tested is a positive control for the wrong thing.
    """
    out = set()
    for _path, lines in files.items():
        for idx, line in enumerate(lines):
            m = is_func_def(line.strip())
            if not m:
                continue
            depth, closed = 0, False
            body = []
            for probe in lines[idx:idx + 60]:
                body.append(probe)
                depth += probe.count("{") - probe.count("}")
                if depth <= 0 and "{" in " ".join(body):
                    closed = True
                    break
            if closed and PRINTF_RE.search(" ".join(body[1:])):
                out.add(m.group(1))
    return out


def collect_definitions(files: dict[Path, list[str]]) -> set[str]:
    names = set()
    for _p, lines in files.items():
        for line in lines:
            m = is_func_def(line.strip())
            if m:
                names.add(m.group(1))
    return names


def resolve_flags(flag_writes: dict[str, list[frozenset]]) -> dict[str, frozenset]:
    """`g_*` globals that ONLY an env-gated path can set, as an any-of clause.

    `g_dyntrace_force` is the measured case: declared `= false`, set `= true` only inside
    `dyntrace_failed_shader_enabled(...)` blocks. Without this, `if (log || g_dyntrace_force)`
    reads as "reachable with nothing set", the requirement collapses to nothing, and the one
    report that answers #305 -- which needs PROSPER_UD_TAIL_ALIGN *and* one of those -- is
    invisible. A flag with even ONE ungated assignment is not an alias and is dropped.
    """
    out = {}
    for name, contexts in flag_writes.items():
        if not contexts or any(not c for c in contexts):
            continue
        union: set[str] = set()
        for ctx in contexts:
            for clause in ctx:
                union |= set(clause)
        if union:
            out[name] = frozenset({frozenset(union)})
    return out


def scan_tree(root: Path):
    files = collect_files(root)
    preds = collect_predicates(files)
    printing = collect_printing(files)
    defined = collect_definitions(files)

    # Pass 1 discovers which `g_*` flags only an env gate can set. Its findings are discarded --
    # they are computed without those flags and would understate.
    flag_writes: dict[str, list[frozenset]] = {}
    for path, lines in files.items():
        FileScanner(str(path.relative_to(root)), lines, preds, {}, printing,
                    None, flag_writes).run()
    flags = resolve_flags(flag_writes)

    call_sites: dict[str, list[tuple[str, frozenset]]] = {}
    findings: list[Finding] = []
    for path, lines in files.items():
        rel = str(path.relative_to(root))
        findings += FileScanner(rel, lines, preds, call_sites, printing, flags).run()
    findings += split_call_findings(call_sites, defined)
    return files, preds, findings


def inventory(files: dict[Path, list[str]]) -> dict[str, list[str]]:
    reads: dict[str, list[str]] = {}
    read_re = re.compile(
        r'(?:getenv|PROSPER_ENV_ON|PROSPER_ENV_VALUE|env_enabled|env_value)\s*\(\s*"('
        + ENV_NAME + r')"')
    for path, lines in files.items():
        for idx, line in enumerate(lines):
            for m in read_re.finditer(line):
                reads.setdefault(m.group(1), []).append(f"{path.name}:{idx + 1}")
    return reads


# --------------------------------------------------------------------------------------------
# Paired arms for the four lexical limits closed in #2628.
#
# Every one of those fixes makes the scanner report FEWER findings, which is the direction that can
# silently disable a rule -- this scanner has already done it once (`SPLIT-LOCAL` reporting 0 across
# `src/` while structurally blind, in the audit's `## Ruled out`). So each is written as a PAIR that
# differs only in the property under test, and the must-match half is what stops "fewer findings"
# from becoming "no findings":
#
#   1 polarity        `== nullptr` vs `!= nullptr` -- ONE character apart
#                     `if (!getenv(X)) { ... }` vs `if (!getenv(X)) return;` -- the two spellings
#                     that used to disagree with each other
#   2 defaulted alias the same function with and without its `if (!out) out = "...";` repair line
#                     `? atoi(e) : 60` vs `? atoi(e) : 0` -- one character, and the difference
#                     between an optional parameter and an off-by-default switch
#   3 lambda alias    the same body STORED (`auto g = [&]{...};`, called later) vs IMMEDIATELY
#                     INVOKED (`const bool g = [&]{...}();`). The use site differs -- `g(...)` vs
#                     `g` -- because it must: that IS the property under test, and no C++ spells
#                     both the same way. The assertion is on the exact rendered requirement, which
#                     only the alias branch can produce.
#   4 split-call      a second call site whose gate is IMPLIED by the first (`{A}` vs `{A|B}`)
#                     versus one that is DISJOINT from it (`{A}` vs `{B}`)
#
# Each must-not-match half was checked to be a FINDING on the pre-#2628 scanner, so none of them is
# passing because it was never in the rule's reach.
#
# Two further arms cover predicate_clauses(), which polarity forced to be rewritten: a predicate
# whose body is `guard; optional filter; early success` must impose the GUARD's variable and not
# the filter's, and one with two truthy exits must impose their disjunction. Both assert the exact
# requirement string, so dropping a variable and adding one fail differently.
# --------------------------------------------------------------------------------------------
_OPT_OUT = """
void report(const State& st) {
    static const bool resolve_on = std::getenv("PROSPER_ZZ_NO_RESOLVE") %s nullptr;
    if (resolve_on) {
        if (getenv("PROSPER_ZZ_RESOLVE_LOG")) fprintf(stderr, "[zz] %%u\\n", st.id);
    }
}
"""

_DEFAULTED_ALIAS = """
void dump_shader(const char* nid) {
    if (getenv("PROSPER_ZZ_AGCSHADER")) {
        const char* out = getenv("PROSPER_ZZ_AGCSHADER_OUT");
%s        if (FILE* f = fopen(out, "wb")) { fprintf(stderr, "  [wrote %%s]\\n", out); }
    }
}
"""

_TERNARY_DEFAULT = """
void dump_pass(uint64_t frame) {
    if (getenv("PROSPER_ZZ_DUMP_PASS")) {
        const char* e = getenv("PROSPER_ZZ_DUMP_PASS_EVERY");
        const int every = e ? atoi(e) : %s;
        if (every) { fprintf(stderr, "  [pass every=%%d]\\n", every); }
    }
}
"""

_LAMBDA_ALIAS = """
void fold(const Stage& s) {
    %s
    if (getenv("PROSPER_ZZ_DYNTRACE")) {
        if (%s) fprintf(stderr, "[zz] %%llu\\n", s.addr);
    }
}
"""

_SPLIT_CALL_GATES = """
bool audio2log() {
    static const bool on = std::getenv("PROSPER_ZZ_AUDIO2LOG") != nullptr;
    return on;
}
void probe_dump(uint64_t p) {
    g_history.insert(p);
}
void trace_call(uint64_t p) {
    if (audio2log()) {
        probe_dump(p);
    }
}
void trace_probe(uint64_t p) {
    if (%s) {
        probe_dump(p);
    }
}
"""


# --------------------------------------------------------------------------------------------
# Self-test. Every case is a shape MEASURED in this tree, reduced to the smallest form that keeps
# the signature. More than half assert on what must NOT match: a checker that fires on sound code
# gets skipped, not heeded.
#
# This is also the positive control the charter demands, and it is deliberately NOT drawn from the
# tree scan: a control built by the same machinery as the null inherits the null's blind spots and
# so tests the discriminator rather than the domain.
# --------------------------------------------------------------------------------------------
SELF_TESTS: list[tuple[str, str, list[str]]] = [
    ("fresh_extent (#2132): filled only under PROSPER_UDPROV, printed regardless", """
bool udprov_enabled() {
    static const bool on = std::getenv("PROSPER_UDPROV") != nullptr;
    return on;
}
void report_block(const State& st) {
    uint32_t fresh = 0;
    if (udprov_enabled()) {
        for (uint32_t i = 0; i < kUserSgprs; i++) {
            fresh = i + 1;
        }
    }
    fprintf(stderr, "[udcand]   block: fresh_extent=%u\\n", fresh);
}
""", ["SPLIT-LOCAL:fresh"]),
    ("same gate on producer and printer is sound", """
bool udprov_enabled() {
    static const bool on = std::getenv("PROSPER_UDPROV") != nullptr;
    return on;
}
void report_block(const State& st) {
    uint32_t fresh = 0;
    if (udprov_enabled()) {
        fresh = compute_extent(st);
        fprintf(stderr, "[udcand] fresh=%u\\n", fresh);
    }
}
""", []),
    ("an ungated local is not a finding", """
void report_block(const State& st) {
    uint32_t fresh = 0;
    if (st.ready) {
        fresh = st.extent;
    }
    fprintf(stderr, "[udcand] fresh=%u\\n", fresh);
}
""", []),

    # NOT reduced to an fprintf. hle_agc.cpp:1774 is `dump_call_log(stderr)` -- one call, no
    # format string. The earlier version of this fixture DID use an fprintf, passed, and the real
    # line went unnoticed in the tree scan: a reduction that removes the property under test is a
    # positive control for a different question. Keep the call form.
    ("progress_unimpl (#2149): the table needs PROSPER_PROGRESS too", """
void dump_call_log(FILE* f) {
    fprintf(f, "=== unimplemented Sony calls ===\\n");
}
void progress_heartbeat(size_t draws_last) {
    static const long interval = [] {
        const char* e = getenv("PROSPER_PROGRESS"); return e ? atol(e) : 0; }();
    if (interval <= 0) return;
    static const bool dump_unimpl = getenv("PROSPER_PROGRESS_UNIMPL") != nullptr;
    if (dump_unimpl && (hb++ % 12) == 0) dump_call_log(stderr);
}
""", ["TWO-GATE:PROSPER_PROGRESS+PROSPER_PROGRESS_UNIMPL"]),
    # Also NOT reduced. The real guard is `if (log || g_dyntrace_force)`, and the disjunct that
    # matters is a GLOBAL FLAG, not a getenv. Reduced to `if (log)` this fixture passed while the
    # live `[udtail]` stayed invisible, because a disjunction with a non-env alternative correctly
    # imposes no requirement -- unless the alternative is a flag only an env gate can set.
    ("udtail (#2146): the report is inside the mutation's own gate", """
bool dyntrace_failed_shader_enabled(uint64_t addr) {
    static const bool on = std::getenv("PROSPER_DYNTRACE_FAIL") != nullptr;
    return on;
}
bool g_dyntrace_force = false;
void replay_failed_stage(Draw& d) {
    if (dyntrace_failed_shader_enabled(d.vs)) {
        g_dyntrace_force = true;
        build_stage_table(d);
        g_dyntrace_force = false;
    }
}
void seed_user_data(Stage& s) {
    const bool log = getenv("PROSPER_GFXLOG") != nullptr;
    static const char* const tail_mode = std::getenv("PROSPER_UD_TAIL_ALIGN");
    if (tail_mode) {
        if (log || g_dyntrace_force) {
            fprintf(stderr, "[udtail] rsrc2=0x%08x\\n", rsrc2);
        }
        s.range_start = prefix;
    }
}
""", ["TWO-GATE:PROSPER_DYNTRACE_FAIL|PROSPER_GFXLOG+PROSPER_UD_TAIL_ALIGN"]),
    ("a single gate, nested, is sound", """
void report(const State& st) {
    if (getenv("PROSPER_GFXLOG")) {
        for (auto& d : st.draws) {
            if (d.valid) fprintf(stderr, "[gfx] %u\\n", d.id);
        }
    }
}
""", []),
    ("the same variable twice is one switch", """
void report(const State& st) {
    if (getenv("PROSPER_GFXLOG")) {
        if (getenv("PROSPER_GFXLOG")) fprintf(stderr, "[gfx] %u\\n", st.id);
    }
}
""", []),
    ("a disjunction is not a requirement", """
void report(const State& st) {
    if (getenv("PROSPER_GFXLOG") || getenv("PROSPER_DBG")) {
        fprintf(stderr, "[gfx] %u\\n", st.id);
    }
}
""", []),

    ("record_guest_write (#2111): one recorder behind a different switch", """
bool writer_provenance_enabled() {
    static const bool on = std::getenv("PROSPER_WRITER_PROVENANCE") != nullptr;
    return on;
}
void record_guest_write(Kind k, uint64_t addr, size_t bytes) {
    g_history.insert(k, addr, bytes);
}
void handle_dma(const Packet& p) {
    if (writer_provenance_enabled()) {
        record_guest_write(Kind::DmaData, p.dst, p.bytes);
    }
}
void handle_write_data(const Packet& p) {
    if (writer_provenance_enabled()) {
        record_guest_write(Kind::WriteData, p.dst, p.bytes);
    }
}
void diagnose_resource_provenance(const State& st) {
    const char* dim_env = getenv("PROSPER_PROVENANCE_DIM");
    if (!dim_env) return;
    record_guest_write(Kind::ColorTarget, st.color0_base, st.bytes);
}
""", ["SPLIT-CALL:record_guest_write"]),
    ("agreeing call sites are sound -- #2111 AFTER #2143", """
bool writer_provenance_enabled() {
    static const bool on = std::getenv("PROSPER_WRITER_PROVENANCE") != nullptr;
    return on;
}
void record_guest_write(Kind k, uint64_t addr, size_t bytes) {
    g_history.insert(k, addr, bytes);
}
void handle_dma(const Packet& p) {
    if (writer_provenance_enabled()) {
        record_guest_write(Kind::DmaData, p.dst, p.bytes);
    }
}
void diagnose_resource_provenance(const State& st) {
    if (writer_provenance_enabled()) {
        record_guest_write(Kind::ColorTarget, st.color0_base, st.bytes);
    }
}
""", []),

    # ---- #2628 limit 1: polarity ------------------------------------------------------------
    # The opt-OUT idiom is this tree's standard default-ON switch, so PROSPER_ZZ_NO_RESOLVE being
    # in the requirement was backwards: the block runs when it is UNSET. Its partner differs by
    # one character and must still be a two-gate report.
    ("opt-out (`== nullptr`) is satisfied by a DEFAULT run, so it gates nothing",
     _OPT_OUT % "==", []),
    ("opt-in (`!= nullptr`) is a real gate -- one character from the arm above",
     _OPT_OUT % "!=", ["TWO-GATE:PROSPER_ZZ_NO_RESOLVE+PROSPER_ZZ_RESOLVE_LOG"]),
    # These two spellings of one idea used to DISAGREE: the negation was pushed through for the
    # early-return form and not for the block form, so the same code flagged or not depending on
    # how the author wrote it. Now the block imposes nothing and the remainder still imposes ALPHA.
    ("`if (!getenv(X)) { ... }` runs when X is UNSET -- the block requires nothing", """
void report(const State& st) {
    if (!getenv("PROSPER_ZZ_ALPHA")) {
        if (getenv("PROSPER_ZZ_BETA")) fprintf(stderr, "[zz] %u\\n", st.id);
    }
}
""", []),
    ("`if (!getenv(X)) return;` still makes the REMAINDER require X", """
void report(const State& st) {
    if (!getenv("PROSPER_ZZ_ALPHA")) return;
    if (getenv("PROSPER_ZZ_BETA")) fprintf(stderr, "[zz] %u\\n", st.id);
}
""", ["TWO-GATE:PROSPER_ZZ_ALPHA+PROSPER_ZZ_BETA"]),

    # ---- #2628 limit 2: a defaulted alias -----------------------------------------------------
    # hle_graphics.cpp:1249 reduced. The mutation arm is the SAME function with the one repair line
    # deleted, so nothing else in the fixture can account for the difference.
    ("a defaulted alias is not a gate (`if (!out) out = \"shader0.bin\";`)",
     _DEFAULTED_ALIAS % '        if (!out || !*out) out = "shader0.bin";\n', []),
    ("...and WITHOUT that one repair line the coupling is real",
     _DEFAULTED_ALIAS % "", ["TWO-GATE:PROSPER_ZZ_AGCSHADER+PROSPER_ZZ_AGCSHADER_OUT"]),
    # `: 60` is an optional parameter; `: 0` is a switch that is off by default. One character, and
    # DEFAULT_RHS_RE -- the tool's existing statement of which literals mean "unset" -- decides.
    ("a ternary with a LIVE default (`: 60`) is a parameter, not a gate",
     _TERNARY_DEFAULT % "60", []),
    ("a ternary with a FALSY default (`: 0`) is a gate",
     _TERNARY_DEFAULT % "0", ["TWO-GATE:PROSPER_ZZ_DUMP_PASS+PROSPER_ZZ_DUMP_PASS_EVERY"]),

    # ---- #2628 limit 3: a lambda is not an alias for its body ---------------------------------
    # `readable` (gpu_executor.cpp:2854) reduced: a helper whose body carries an env branch used to
    # export that requirement to every CALL of it, invisibly at the call site. An immediately
    # invoked lambda is the opposite case -- it yields a value the environment decides -- and
    # #2149's fourth measured instance depends on that one still resolving.
    ("a STORED lambda does not export its body's switches to its call sites",
     _LAMBDA_ALIAS % ('auto readable = [&](uint64_t a) { const bool gfxlog = '
                      'getenv("PROSPER_ZZ_GFXLOG") != nullptr; return gfxlog && guest_readable(a); };',
                      "readable(s.addr)"), []),
    ("an IMMEDIATELY INVOKED lambda holds a value the environment decides, and still aliases",
     _LAMBDA_ALIAS % ('const bool readable = [&] { const bool gfxlog = '
                      'getenv("PROSPER_ZZ_GFXLOG") != nullptr; return gfxlog && guest_readable(0); }();',
                      "readable"), ["TWO-GATE:PROSPER_ZZ_DYNTRACE+PROSPER_ZZ_GFXLOG"]),

    # ---- #2628 limit 4: SPLIT-CALL compared clause sets by identity ---------------------------
    # hle_audio.cpp reduced: {AUDIO2LOG} at one site and {AUDIO2LOG|AUDIO2_PROBE} at another are
    # not a disagreement -- arming the first satisfies both. A DISJOINT second gate still is.
    ("call sites whose gates IMPLY one another do not disagree",
     _SPLIT_CALL_GATES % 'audio2log() || getenv("PROSPER_ZZ_AUDIO2_PROBE")', []),
    ("call sites whose gates are DISJOINT still disagree",
     _SPLIT_CALL_GATES % 'getenv("PROSPER_ZZ_AUDIO2_PROBE")', ["SPLIT-CALL:probe_dump"]),

    # ---- predicate bodies, which polarity forced to be modelled ------------------------------
    # dyntrace_failed_shader_enabled (gpu_executor.cpp:401) reduced. The flat reading joined both
    # variables into one any-of clause; a flat POLARITY reading would have kept only the optional
    # one and dropped the required one, quietly weakening a `defect` row's key. The assertion is
    # the exact requirement, so either error fails it.
    ("a predicate's guard is required and its optional filter is not", """
bool dynfail_enabled(uint64_t code_addr) {
    if (!std::getenv("PROSPER_ZZ_DYNFAIL")) return false;
    const char* filter = std::getenv("PROSPER_ZZ_DYNFAIL_ADDR");
    if (!filter) return true;
    return strtoull(filter, nullptr, 16) == code_addr;
}
void replay(const Draw& d) {
    if (dynfail_enabled(d.vs)) {
        if (getenv("PROSPER_ZZ_DYNFAIL_LOG")) fprintf(stderr, "[zz] 0x%llx\\n", d.vs);
    }
}
""", ["TWO-GATE:PROSPER_ZZ_DYNFAIL+PROSPER_ZZ_DYNFAIL_LOG"]),
    # Two exits can return true, so EITHER variable reaches the report -- one any-of clause, not
    # two requirements and not nothing. report_selected_sbuffer_reject
    # (rdna2_gta5_compute_contracts.cpp:39) reduced.
    ("a predicate with two truthy exits requires their DISJUNCTION", """
bool report_reject(const uint32_t* code) {
    if (std::getenv("PROSPER_ZZ_DBG")) return true;
    if (!std::getenv("PROSPER_ZZ_REJECT")) return false;
    return code != nullptr;
}
void reject(const uint32_t* code, const State& st) {
    if (report_reject(code)) {
        if (getenv("PROSPER_ZZ_REJECT_LOG")) fprintf(stderr, "[zz] %u\\n", st.id);
    }
}
""", ["TWO-GATE:PROSPER_ZZ_DBG|PROSPER_ZZ_REJECT+PROSPER_ZZ_REJECT_LOG"]),
]


# --------------------------------------------------------------------------------------------
# Key-stability arms. These test the BASELINE KEY rather than the signatures, because the key is
# what the multi-lane gate stands on: if an unrelated edit re-keys a row, the Docs job fails on a
# PR that did nothing wrong, and the obvious repair -- regenerate the baseline -- silently
# downgrades every classification it carries.
#
# Two arms, and the second is the one that makes the first mean anything. Insensitivity alone is
# satisfiable by a key that never changes at all, which would be useless; the discrimination arm
# is what shows the key still separates two different requirements.
# --------------------------------------------------------------------------------------------
_LONG_PREDICATE = """
bool any_dump_enabled() {
    static const bool on = std::getenv("PROSPER_DUMP_ATLAS") != nullptr ||
        std::getenv("PROSPER_DUMP_DRAWSTEPS") != nullptr ||
        std::getenv("PROSPER_DUMP_RAWTEX") != nullptr ||
        std::getenv("PROSPER_DUMP_RTGROUPS") != nullptr ||
        std::getenv("PROSPER_DUMP_TEX") != nullptr%s;
    return on;
}
void report_block(const State& st) {
    uint32_t hits = 0;
    if (any_dump_enabled()) {
        hits = st.count;
    }
    fprintf(stderr, "[dump] hits=%%u\\n", hits);
}
"""

# The discrimination arm changes the CONJUNCTION: the producer now needs a SECOND, independent
# variable that the printer still does not. That is a real change in what a reader must arm for
# the field to carry a measurement, so the key must move.
#
# Note where the gate had to go. Wrapping the *fprintf* instead leaves the finding's requirement
# untouched -- SPLIT-LOCAL keys on the clauses the writes need and the print does not -- so that
# version of this arm passed while proving nothing. The arm has to change the thing the key is
# made of, which is the same mistake as a reduced fixture that drops the property under test.
_LONG_PREDICATE_TWO_GATE = _LONG_PREDICATE.replace(
    "        hits = st.count;",
    '        if (getenv("PROSPER_RTT")) {\n            hits = st.count;\n        }')


def _scan_fixture(snippet: str) -> tuple[int, list]:
    """Scan one fixture in a temp tree. Returns (files collected, findings).

    The file COUNT is returned so every caller can separate an apparatus failure from a
    substantive one. A fixture the scanner never opened produces no findings, and "no findings"
    is exactly what a broken signature produces too -- #2658 is the whole argument: twelve arms
    printed `got=[]` under the banner "the scanner's own signatures are broken" when the real
    cause was that `collect_files` had discarded the fixture before any signature ran.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "src").mkdir()
        (root / "src" / "fixture.cpp").write_text(snippet, encoding="utf-8")
        files, _preds, findings = scan_tree(root)
    return len(files), findings


def _apparatus_failure(where: str) -> None:
    import tempfile
    print(f"  [FAIL] APPARATUS: {where}'s own fixture collected 0 file(s) -- the scanner never")
    print("         opened the tree it had just written, so every arm below would report an")
    print("         empty result for a reason that has nothing to do with what it tests.")
    print(f"         TMPDIR={tempfile.gettempdir()}")
    print("         (see #2658: a scan-root exclusion that reads the path ABOVE the root)")


def _keys_for(snippet: str) -> set:
    _n, findings = _scan_fixture(snippet)
    return {f.key() for f in findings}


def run_key_stability_test(verbose: bool = False) -> int:
    bad = 0
    base = _keys_for(_LONG_PREDICATE % "")
    if not base:
        print("  [FAIL] key-stability fixture produced NO findings -- it cannot test anything")
        return 1

    # Arm 1: one more name joins the 5-way alternation. Nothing about the requirement a reader
    # must satisfy has changed in kind, so no key may move.
    grown = _keys_for(_LONG_PREDICATE % '||\n        std::getenv("PROSPER_DUMP_NEWCOMER") != nullptr')
    if grown != base:
        bad += 1
        print("  [FAIL] key-stability: adding a name to a long any-of clause re-keyed a finding")
        for k in sorted(base ^ grown):
            print(f"           {k}")
    elif verbose:
        print(f"  [ok]   key stability: +1 name in a long alternation moves no key ({len(base)} key(s))")

    # Arm 2: the CONJUNCTION changes -- a second independent gate on the same report. Without
    # this arm, arm 1 is also satisfied by a key that ignores the requirement entirely.
    changed = _keys_for(_LONG_PREDICATE_TWO_GATE % "")
    if changed == base:
        bad += 1
        print("  [FAIL] key-discrimination: adding an INDEPENDENT gate left every key unchanged --")
        print("         the key is insensitive rather than stable, and cannot separate two")
        print("         different requirements in one file")
    elif verbose:
        print("  [ok]   key discrimination: a changed conjunction moves the key")

    if not bad:
        print("  [ok]   key stability: 2 arms (membership-insensitive, conjunction-sensitive)")
    return bad


def run_exclusion_test(verbose: bool = False) -> int:
    """What `collect_files` skips, as a PAIR of arms. #2658.

    The hostile root is constructed BY HAND rather than by pointing `TMPDIR` at a build directory
    and hoping. `tempfile.gettempdir()` silently falls back when `TMPDIR` names a directory that
    does not exist, so an arm built out of the environment reports a confident pass on a root that
    was never hostile -- the same class of vacuum as the defect it is guarding.

    Both directions matter and they pull opposite ways, which is why neither arm stands alone:
    arm 1 fails on the absolute-path substring this replaced, arm 2 fails on simply deleting the
    exclusion.
    """
    import tempfile

    bad = 0
    fixture = 'void f(void) { if (getenv("PROSPER_ZZ_EXCL")) fprintf(stderr, "x"); }\n'

    with tempfile.TemporaryDirectory() as td:
        # Arm 1: the scan root sits UNDER directories that look like a build tree. Every component
        # above the root is the caller's business, not ours, and must not filter anything.
        hostile = Path(td) / "build-linux" / "tmpdir" / "third_party" / "tree"
        (hostile / "src").mkdir(parents=True)
        (hostile / "src" / "fixture.cpp").write_text(fixture, encoding="utf-8")
        got = collect_files(hostile)
        if len(got) != 1:
            bad += 1
            print("  [FAIL] exclusion: a scan root beneath 'build-linux/tmpdir/third_party'")
            print(f"         collected {len(got)} file(s), want 1 -- the exclusion is reading the")
            print("         path ABOVE the root, so $TMPDIR decides what the scanner can see")
        elif verbose:
            print("  [ok]   exclusion: components above the scan root do not filter")

        # Arm 2: inside the root the exclusion must still bite, or arm 1 is satisfied by deleting
        # it. Both halves are named explicitly, because the dangerous direction here is the quiet
        # one: a build tree kept produces noisy extra findings somebody will investigate, while a
        # source directory dropped produces a CLEAN report over a smaller tree.
        inner = Path(td) / "tree2"
        excluded_dirs = ("src/build", "src/build-linux", "src/cmake-build-debug",
                         "src/pre-build", "src/third_party")
        scanned_dirs = ("src", "src/builder", "src/build_utils", "src/rebuild-cache")
        for sub in excluded_dirs + scanned_dirs:
            (inner / sub).mkdir(parents=True, exist_ok=True)
        scanned_files = ["src/ok.cpp", "src/third_party_shim.cpp", "src/buildinfo.cpp"] + [
            f"{d}/real.cpp" for d in scanned_dirs[1:]]
        for rel in scanned_files + [f"{d}/generated.cpp" for d in excluded_dirs]:
            (inner / rel).write_text(fixture, encoding="utf-8")
        kept = {str(p.relative_to(inner)).replace("\\", "/") for p in collect_files(inner)}
        want = set(scanned_files)
        if kept != want:
            bad += 1
            print("  [FAIL] exclusion: the wrong files were kept inside the scan root")
            print(f"         want={sorted(want)}")
            print(f"         got ={sorted(kept)}")
        elif verbose:
            print("  [ok]   exclusion: build-linux/ and third_party/ inside the root are skipped")

    if not bad:
        print("  [ok]   scan-root exclusion: 2 arms (above the root ignored, inside it applied)")
    return bad


def run_self_test(verbose: bool = False) -> int:
    bad = 0
    for label, snippet, want in SELF_TESTS:
        nfiles, findings = _scan_fixture(snippet)
        if not nfiles:
            _apparatus_failure("the scanner self-test")
            return 1
        got = set()
        for f in findings:
            if f.kind == "TWO-GATE":
                got.add(f"TWO-GATE:{Finding.render(f.gates)}")
            else:
                got.add(f"{f.kind}:{f.subject}")
        missing = [w for w in want if w not in got]
        extra = [g for g in got if g not in want]
        if missing or extra:
            bad += 1
            print(f"  [FAIL] {label}")
            print(f"         want={sorted(want)}")
            print(f"         got ={sorted(got)}")
        elif verbose:
            print(f"  [ok]   {label}")
    if bad:
        print("  the scanner's own signatures are broken -- a tree scan would report a false CLEAN")
    else:
        print(f"  [ok]   scanner self-test: {len(SELF_TESTS)} cases "
              f"({sum(1 for _l, _s, w in SELF_TESTS if w)} positive, "
              f"{sum(1 for _l, _s, w in SELF_TESTS if not w)} must-not-match)")
    return bad


def load_baseline(path: Path) -> dict[str, str]:
    entries = {}
    if not path.is_file():
        return entries
    for line in path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _sep, note = line.partition("  #")
        entries[key.strip()] = note.strip()
    return entries


def _header_class_counts(text: str) -> dict[str, int]:
    """The `#   benign: 2` block. Only the four known class names are read, so a typo in the header
    surfaces as "no count declared for <class>" rather than as a silently accepted new class."""
    declared: dict[str, int] = {}
    for line in text.split("\n"):
        m = re.match(r"^#\s+([a-z][a-z-]*):\s*(\d+)\s*$", line)
        if m and m.group(1) in BASELINE_CLASSES:
            declared[m.group(1)] = int(m.group(2))
    return declared


# Rule 4's two demands on a `benign` note. A `file:line` the next reader can open, and the issue the
# judgement was made under. Deliberately `benign`-ONLY: the general form (every non-`unreviewed` row)
# fails 13 of the 14 such rows the #2628 baseline shipped, and fails them where the demand is wrong --
# ten identical `config-echo` notes that are CLASS STATEMENTS rather than arguments, and two `defect`
# rows that carry an issue number and rightly no file:line. Forcing a citation into those manufactures
# evidence for a regex. `benign` is the one class whose entire content IS an argument, so it is the one
# class where "the note was replaced by a stub" is a loss the tool can define.
BENIGN_CITATION_RE = re.compile(r"[A-Za-z0-9_]+\.(?:c|cc|cpp|h|hpp|py):\d+")
BENIGN_ISSUE_RE = re.compile(r"#\d+")


def baseline_integrity(text: str, budget: int | None = None) -> list[str]:
    """Check the baseline against ITSELF. Returns one message per failure; empty means sound.

    The gate above this one gets half the job. It fails on a stale KEY and on a missing KEY, and it
    is completely indifferent to a NOTE -- so a row can keep its key and lose the judgement that is
    the only reason the row is worth having, with nothing in CI able to tell. That asymmetry is not
    hypothetical: two branches rewrote this file at once (#2628 and #2572), and the resolution
    everyone reaches for -- "keep my side of the conflict" -- passes every check while discarding
    the other lane's 79 notes, with no conflict left over and a diff that reads as your own edit
    (the trap-41 / #1701 shape).

    Four rules, each cheap and each falsifiable:

      1. every row carries a class from BASELINE_CLASSES;
      2. the header's own class counts match the rows, for all four classes -- a summary nobody
         checks is the same silent-drift door one level up, and this file's header is quoted
         verbatim in docs/DIAGNOSTIC_GATE_AUDIT.md;
      3. the `unreviewed` count equals UNREVIEWED_BUDGET exactly (see that constant);
      4. every `benign` note names a file:line and an issue (see BENIGN_CITATION_RE) -- the class
         whose whole content is an argument is the class where a note can be replaced by `ok` and
         still pass rules 1-3.

    Rule 3 is the one with teeth, because `unreviewed` is what a regeneration turns every judgement
    into; rule 4 closes the one hole rule 3 leaves open in the same direction -- keeping the CLASS
    and stubbing the NOTE. What none of the four can see, stated so nobody quotes this as more than
    it is: a baseline copied WHOLE from one branch over another is internally consistent by
    construction and passes all four. Only merge ORDER protects against that -- land the branch that
    rebuilds the keys first, so the second merger's "keep mine" resolution is the one that fails
    loudly.
    """
    if budget is None:
        budget = UNREVIEWED_BUDGET
    rows: list[tuple[str, str]] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _sep, note = line.partition("  #")
        rows.append((key.strip(), note.strip()))

    failures: list[str] = []
    counts = {cls: 0 for cls in BASELINE_CLASSES}
    for key, note in rows:
        cls = note.split(":", 1)[0].strip()
        if cls in counts:
            counts[cls] += 1
        else:
            failures.append(
                f"row carries no recognised classification (want one of "
                f"{', '.join(BASELINE_CLASSES)}): {key}")

    for key, note in rows:
        if note.split(":", 1)[0].strip() != "benign":
            continue
        if not BENIGN_CITATION_RE.search(note):
            failures.append(
                f"`benign` note names no file:line -- the mechanism has to be at somewhere the next "
                f"reader can open, or the row records a verdict with its argument deleted: {key}")
        if not BENIGN_ISSUE_RE.search(note):
            failures.append(
                f"`benign` note names no issue -- say which review judged it, so the reasoning is "
                f"recoverable when the code moves under the citation: {key}")

    declared = _header_class_counts(text)
    for cls in BASELINE_CLASSES:
        if cls not in declared:
            failures.append(f"the header declares no count for `{cls}` -- add `#   {cls}: "
                            f"{counts[cls]}` to the class-count block")
        elif declared[cls] != counts[cls]:
            failures.append(f"the header says `{cls}: {declared[cls]}` and the rows say "
                            f"{counts[cls]} -- one of the two was edited without the other")

    if counts["unreviewed"] > budget:
        failures.append(
            f"{counts['unreviewed']} `unreviewed` row(s) against a budget of {budget} -- unjudged "
            f"debt was ADDED. If this came from `--emit-baseline`, it has overwritten judgements: "
            f"that command classifies every row `unreviewed`, so a `defect` row is laundered into "
            f"an unjudged one by a command that looks like housekeeping. Re-apply the notes.")
    elif counts["unreviewed"] < budget:
        failures.append(
            f"{counts['unreviewed']} `unreviewed` row(s) against a budget of {budget} -- debt was "
            f"PAID. Tighten UNREVIEWED_BUDGET in check_diag_gates.py to {counts['unreviewed']} so "
            f"the ratchet holds; leaving it slack lets the same number of rows silently return.")
    return failures


# Arms for baseline_integrity(). Each pair differs ONLY in the property under test, and every
# must-fail half is a mutation of the must-pass one -- a rule that cannot be shown to fire is not a
# rule, and this file's own `## Ruled out` records a check of exactly that kind sitting inert.
_SOUND_BASELINE = """\
# A baseline.
#   defect: 1
#   config-echo: 0
#   benign: 1
#   unreviewed: 2
#
TWO-GATE|src/a.cpp|report|PROSPER_A+PROSPER_B  # defect: the printer needs both. #2149
SPLIT-LOCAL|src/b.cpp|hits|PROSPER_C  # benign: structural (b.cpp:12), not a count of things observed. #2572
TWO-GATE|src/c.cpp|report|PROSPER_D+PROSPER_E  # unreviewed: not judged
TWO-GATE|src/d.cpp|report|PROSPER_F+PROSPER_G  # unreviewed: not judged
"""
_SOUND_BENIGN = "# benign: structural (b.cpp:12), not a count of things observed. #2572"

BASELINE_INTEGRITY_TESTS = (
    ("a sound baseline passes", _SOUND_BASELINE, 2, ""),
    ("an unclassified row is caught",
     _SOUND_BASELINE.replace("# defect: the printer needs both. #2149", "# ok, looks fine"),
     2, "no recognised classification"),
    ("a header count that disagrees with the rows is caught",
     _SOUND_BASELINE.replace("#   benign: 1", "#   benign: 2"), 2, "was edited without the other"),
    ("a class missing from the header is caught",
     _SOUND_BASELINE.replace("#   config-echo: 0\n", ""), 2, "declares no count for `config-echo`"),
    # The two rule-3 arms. Both mutate the SAME file in the same place; only the direction differs,
    # and the messages have to differ too -- "debt was added" and "debt was paid" call for opposite
    # actions from whoever reads them.
    ("a judgement downgraded to `unreviewed` is caught",
     _SOUND_BASELINE.replace(_SOUND_BENIGN, "# unreviewed: not judged")
                    .replace("#   benign: 1", "#   benign: 0")
                    .replace("#   unreviewed: 2", "#   unreviewed: 3"),
     2, "debt was ADDED"),
    ("debt paid without tightening the ratchet is caught", _SOUND_BASELINE, 3, "debt was PAID"),
    # The two rule-4 arms. Both keep the CLASS -- which is what makes this hole invisible to rules
    # 1-3 -- and delete one half of what makes the class mean anything.
    ("a `benign` note stubbed of its file:line is caught",
     _SOUND_BASELINE.replace(_SOUND_BENIGN, "# benign: ok, structural rather than measured. #2572"),
     2, "names no file:line"),
    ("a `benign` note stubbed of its issue reference is caught",
     _SOUND_BASELINE.replace(_SOUND_BENIGN, "# benign: structural (b.cpp:12), not a count."),
     2, "names no issue"),
)


def run_baseline_integrity_test(verbose: bool = False) -> int:
    bad = 0
    for label, text, budget, want in BASELINE_INTEGRITY_TESTS:
        got = baseline_integrity(text, budget)
        ok = (not got) if not want else any(want in g for g in got)
        if not ok:
            bad += 1
            print(f"  [FAIL] baseline-integrity: {label}")
            print(f"         want={want or '<no failures>'}")
            print(f"         got ={got or '<no failures>'}")
        elif verbose:
            print(f"  [ok]   baseline-integrity: {label}")
    if not bad:
        print(f"  [ok]   baseline integrity: {len(BASELINE_INTEGRITY_TESTS)} arms "
              f"({sum(1 for _l, _t, _b, w in BASELINE_INTEGRITY_TESTS if not w)} positive, "
              f"{sum(1 for _l, _t, _b, w in BASELINE_INTEGRITY_TESTS if w)} must-fail)")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", nargs="?", help="checkout root containing src/ frontends/ tools/ tests/")
    ap.add_argument("--selftest", action="store_true", help="run only the scanner's own cases")
    ap.add_argument("--list", action="store_true", help="print every finding, baselined or not")
    ap.add_argument("--inventory", action="store_true", help="print the PROSPER_* read inventory")
    ap.add_argument("--emit-baseline", action="store_true",
                    help="print a baseline file for the current tree (classify the notes by hand)")
    ap.add_argument("--require-hostile-tmpdir", action="store_true",
                    help="fail unless $TMPDIR itself looks like a build tree -- so the ctest case "
                         "that pins #2658 cannot pass on a TMPDIR that quietly fell back to /tmp")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("== check_diag_gates ==")
    if args.require_hostile_tmpdir:
        import tempfile
        tmp = Path(tempfile.gettempdir())
        if not any(is_excluded_dir(p) for p in tmp.parts):
            print("  [FAIL] --require-hostile-tmpdir: no component of the temp root looks like a")
            print("         build tree. tempfile falls back silently when $TMPDIR names a")
            print("         directory that does not exist, so this run would have pinned #2658")
            print(f"         against an environment it never had. temp root: {tmp}")
            return 1
        print(f"  [ok]   temp root {tmp} is a build-tree path -- the #2658 arm is live")
    if (run_exclusion_test(args.verbose) or run_self_test(args.verbose)
            or run_key_stability_test(args.verbose)
            or run_baseline_integrity_test(args.verbose)):
        return 1
    if args.selftest:
        print("== all checks passed ==")
        return 0

    here = Path(__file__).resolve()
    root = Path(args.root).resolve() if args.root else here.parents[2]
    files, preds, findings = scan_tree(root)
    if not files:
        print(f"  [FAIL] no sources found under {root} -- the scan is not seeing the tree")
        return 1
    if not preds:
        print("  [FAIL] no env predicates resolved -- every measured instance hides behind one, so")
        print("         a scan without them reports a clean tree by construction")
        return 1

    reads = inventory(files)
    multi = {k: v for k, v in reads.items() if len(v) > 1}
    print(f"  [ok]   scanned {len(files)} file(s) under {root.name}: "
          f"{len(reads)} PROSPER_* variable(s) read at {sum(len(v) for v in reads.values())} "
          f"site(s), {len(multi)} read at more than one, {len(preds)} env predicate(s)")

    if args.inventory:
        for name in sorted(reads):
            print(f"    {name}  ({len(reads[name])} read site(s))")

    if args.emit_baseline:
        for key in sorted({f.key() for f in findings}):
            print(f"{key}  # unreviewed")
        return 0

    baseline_path = here.parent / "diag_gate_baseline.txt"
    baseline = load_baseline(baseline_path)
    integrity = (baseline_integrity(baseline_path.read_text(encoding="utf-8"))
                 if baseline_path.is_file() else [])
    by_key = {f.key(): f for f in findings}
    new = [f for k, f in sorted(by_key.items()) if k not in baseline]
    stale = [k for k in sorted(baseline) if k not in by_key]

    counts: dict[str, int] = {}
    for f in findings:
        counts[f.kind] = counts.get(f.kind, 0) + 1
    print("  [ok]   findings: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
          + f" (total {len(findings)}, baselined {len(findings) - len(new)})")

    if args.list:
        for k, f in sorted(by_key.items()):
            print(f"   {' ' if k in baseline else '*'}{f.line()}")
            if k in baseline and baseline[k]:
                print(f"      {baseline[k]}")

    if not integrity:
        print(f"  [ok]   baseline notes: {len(baseline)} row(s) classified, header counts agree, "
              f"{UNREVIEWED_BUDGET} unreviewed as budgeted")

    if not new and not stale and not integrity:
        print("== all checks passed ==")
        return 0

    if integrity:
        print(f"  [FAIL] {len(integrity)} baseline integrity problem(s) -- the KEYS may be current "
              f"while the JUDGEMENTS are not:")
        for msg in integrity:
            print(f"    {msg}")
    if new:
        print(f"  [FAIL] {len(new)} finding(s) not in the baseline:")
        for f in new:
            print(f"    {f.line()}")
        print("  Each is a diagnostic that can print a field it did not measure. Fix it, or add")
        print("  the key to tools/env/diag_gate_baseline.txt with a classification and an issue.")
    if stale:
        print(f"  [FAIL] {len(stale)} baseline entry(ies) no longer reproduce -- delete the row:")
        for k in stale:
            print(f"    {k}")
    print(f"== {len(new) + len(stale) + len(integrity)} failure(s) ==")
    return 1


if __name__ == "__main__":
    sys.exit(main())
