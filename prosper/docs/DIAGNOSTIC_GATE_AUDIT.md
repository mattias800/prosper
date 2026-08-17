# Diagnostic gate audit — when a `0` means "nobody was counting"

Status: **live**. The numbers below are re-derivable, not quoted — regenerate them with the command
in [Re-running the audit](#re-running-the-audit). Tracking issue: #2149.

## The defect this document is about

> A diagnostic prints a field whose **producer** is armed by a different switch than the one that
> arms the **printer**. When the producer is unarmed the field reads as a confident, well-formed
> **zero** — indistinguishable from a measured negative.

The zero is not merely uninformative. It is *affirmatively misleading*, because a zero in these
fields is usually exactly what the interesting hypothesis predicts. #2111's instrument emitted
**183 negatives it had no capacity to falsify**, and the reader wrote the conclusion down before
checking the gate. The cost is not the wasted arm: these zeros land in `## Ruled out` rows and issue
comments as *measurements*, and the charter is explicit that those rows are the most expensive
knowledge in the repository.

The failure is silent, survives review, and **looks like rigour** — the output is well formed and
the number is specific.

## The convention

Three rules, in priority order. The first two are the durable fix; the third is what makes the first
two possible.

1. **A diagnostic that can print a field it did not measure prints its armed state alongside it.**
   #2143 is the worked example: every "no recorded GPU writer" negative now carries the history's
   own per-kind counts, so `color=0` is visible as *"that recorder never fired"* rather than as
   *"no draw wrote here"*. The banner states **scope**, not completeness — trying to make it
   exhaustive is how it becomes wrong again.
2. **Prefer refusing to print over printing a zero.** An absent field invites the question; a `0`
   answers it wrongly. Where armed-ness cannot be determined, print `unarmed` or `n/a`, not a
   numeral.
3. **Keep a report separate from any behaviour change it evaluates.** A diagnostic welded to an
   intervention can only be read on runs whose results are already disqualified. #2146 is the acute
   case and is still open — see the table below.

A fourth rule earned by the instance fixed in this document's own PR:

4. **A switch named for a question answers it alone.** `PROSPER_PROGRESS_UNIMPL` reads as a complete
   switch for *"show me unimplemented-call activity"*. It used to need `PROSPER_PROGRESS` as well,
   so arming only the variable whose name matched the question produced the deduped first-seen log
   and no call-count table — silence that reads as *"8 calls in 214 s, no spin"*. The coupling was
   even documented in the code; what failed is that **the name promised what only the pair
   delivered**.

## The audit

`prosper/tools/env/check_diag_gates.py` is a re-runnable scanner, registered as the ctest
`diag_gate_selftest` and run over the tree in CI's `Docs` job. It checks three lexical signatures,
each derived from a **measured** instance rather than invented:

| Signature | What it means | Measured instance |
| --- | --- | --- |
| `SPLIT-LOCAL` | a local declared with a default (`= 0`) whose only real writes sit inside an env-gated region, printed **outside** it | `fresh_extent` on #2132 |
| `TWO-GATE` | one report statement requiring **two** distinct `PROSPER_*` variables | `[udtail]` (#2146); the unimplemented-NID table (#2149) |
| `SPLIT-CALL` | a callee reached only under env gates whose call sites **disagree** on which gate | `record_guest_write(ColorTarget, …)` (#2111) |

### Headline numbers (master, 2026-08-17)

| | count |
| --- | --- |
| `PROSPER_*` variables read anywhere in the tree | **651** |
| read sites for them | **1,055** |
| variables read at more than one site | **155** |
| env predicate functions resolved (`udprov_enabled()` and friends) | **61** |
| **findings** (`SPLIT-CALL` 4, `SPLIT-LOCAL` 22, `TWO-GATE` 151) | **177** |
| distinct baseline keys for those findings | **93** |

Classification of the 93 keys, from `tools/env/diag_gate_baseline.txt`:

| class | keys | meaning |
| --- | --- | --- |
| `defect` | 2 | a confirmed instance; the note names the issue |
| `config-echo` | 10 | the field echoes **configuration**, so `0` truthfully means "unconstrained" — the shape rule 1 recommends |
| `family` | 28 | **triage, not a clearance** — every variable in the requirement shares a name family, so a reader who arms one can find the other from the names |
| `unreviewed` | 53 | found by the sweep and **not judged**; honest debt |

**`family` is not a clean bill of health, and this is the one line in this document most likely to
be misread.** #2149's fourth instance was `PROSPER_PROGRESS` + `PROSPER_PROGRESS_UNIMPL` — which *is*
a name family, and the family is exactly what made the coupling invisible. Read a `family` row before
quoting its output as a measurement.

### Confirmed instances

| Where | Field / report | Requirement | State |
| --- | --- | --- | --- |
| `src/gpu/gpu_executor.cpp` (`[udtail]`) | the whole report | `PROSPER_UD_TAIL_ALIGN` **and** one of `PROSPER_GFXLOG` / `PROSPER_DYNTRACE_FAIL` / `PROSPER_DYNTRACE_FAIL_ADDR` | **open, #2146** — the charter requires `PROSPER_UD_TAIL_ALIGN` stay off, so the one report answering the #305 question is available only on runs the project disqualifies |
| `src/gpu/gpu_executor.cpp` (`[udcand]`) | `fresh_extent` | filled only under `PROSPER_UDPROV` / `PROSPER_GPU_CAPTURE_RESOURCE_PROVENANCE`, printed regardless | **open, #2149 inst. 3** — `fresh_extent=0` reads as *"the bind programmed nothing"* |
| `src/hle/hle_agc.cpp` | unimplemented-NID call-count table | needed `PROSPER_PROGRESS` as well as `PROSPER_PROGRESS_UNIMPL` | **fixed** — the dump now has its own cadence |
| `src/gpu/gpu_executor.cpp` (`record_guest_write`) | `color=…` | the `ColorTarget` recorder sat behind `PROSPER_PROVENANCE_DIM` | **fixed by #2143** — the scanner agrees: the post-fix shape produces no finding |

## Re-running the audit

```bash
python3 prosper/tools/env/check_diag_gates.py --selftest      # signatures only, any directory
python3 prosper/tools/env/check_diag_gates.py prosper         # scan the tree against the baseline
python3 prosper/tools/env/check_diag_gates.py prosper --list  # every finding, baselined or not
python3 prosper/tools/env/check_diag_gates.py prosper --emit-baseline   # regenerate the key list
```

A finding not in `tools/env/diag_gate_baseline.txt` fails the check: fix it, or add the key with a
classification and an issue link. A baseline row that no longer reproduces **also** fails: the fix
landed, so delete the row. The second half matters as much as the first — a baseline nobody is
forced to update stops describing the tree, which is this document's whole subject.

## What the scanner cannot see

Stated here rather than left implied, because a clean run of an instrument that cannot observe the
thing is the exact failure this document exists to prevent.

- It is a **lexical scanner**, not a compiler: braces, `if` conditions and early-return guards, with
  no overload resolution, template instantiation, macro expansion or value analysis.
- **Cross-translation-unit state is invisible.** A producer filling a global in one file and printed
  from another is caught only if the producer is reached through a call whose gates disagree
  (`SPLIT-CALL`). A shared counter written and read under different gates in different files is not
  detected.
- A **disjunction imposes no requirement**: `if (getenv("A") || flag)` and
  `if (getenv("A") || getenv("B"))` both yield nothing. That understates deliberately — the
  direction that makes no false accusation.
- `#if`-gated code is scanned as if it were live.
- A diagnostic gated by something that is **not an environment variable** — a build flag, a member
  field, a runtime setting — is out of scope by construction. The most consequential example is in
  the next section.

So a clean run means *"no new instance of three specific lexical shapes"*, never *"no diagnostic in
this tree can print an unmeasured field"*.

## Ruled out

- **"A flat set of variable names is enough to express a gate."** It is not, and reading gates flat
  gets both directions wrong: `getenv(A) && getenv(B)` needs both while `getenv(A) || getenv(B)`
  needs either, and a flat reading collapses them. Measured: the flat revision of the scanner
  reported **133** two-gate findings on master, including one predicate it claimed required all
  sixteen of its alternatives. Gates are a conjunction of disjunctions
  (`check_diag_gates.py:clauses_of`). #2149.
- **"`SPLIT-LOCAL` returning zero on `src/` means the emulator is clean."** It meant the rule was
  anchored on file-scope function detection, and nearly everything under `src/` lives inside
  `namespace { }`. The rule was inert across the whole emulator while still firing in `tools/` —
  nine findings, all in `tools/`, and a confident zero everywhere that mattered. Local lifetime is a
  **brace**, not a function. #2149.
- **"A reduced fixture reproducing a bug is a positive control for it."** Both reductions written
  for this scanner passed while their live originals went unnoticed. `hle_agc.cpp:1774` is
  `dump_call_log(stderr)` — a call with no format string — and the fixture used an `fprintf`;
  `gpu_executor.cpp:6402`'s real guard is `if (log || g_dyntrace_force)`, where the alternative is a
  **global flag** rather than a `getenv`, and the fixture used `if (log)`. Each reduction removed
  the very property under test. The fixtures now carry the real shapes. #2149.
- **"`PROSPER_UD_TAIL_ALIGN=off` gives you the `[udtail]` report without the mutation."** It does
  not: the gate is `getenv() != nullptr`, so `0`, `off` and the empty string all enter the block and
  apply `range_start = prefix`. #2146.

## See also

- `prosper/tools/env/check_diag_gates.py` — the scanner, its signatures, and its self-tests.
- `prosper/tools/env/check_cached_env.py` — the neighbouring gate: a `PROSPER_ENV_ON`-cached read
  never observes a later `setenv`, so a test that arms a diagnostic goes vacuous while printing
  `[ok]` (#2214).
- `prosper/docs/GAME_COMPAT_ORCHESTRATION.md` — the instrument-not-the-subject list. Traps 116 and
  121 are this defect's discipline-based predecessor, and 121 records that discipline decaying: the
  agent who hit the third instance had read 116.
