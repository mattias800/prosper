# tools/refactor — mechanical restructuring, and what it cannot do

These tools move code between files and prove they did not change it. They were written for a
codebase where several files had grown past ten thousand lines, and they are deliberately narrow:
each one does a transformation whose correctness can be *checked*, not merely reviewed.

| tool | what it does |
| --- | --- |
| `move_module.py` | relocates modules into folders; rewrites every `#include` to one canonical form and every path citation repo-wide |
| `map_symbols.py` | tiles a translation unit into top-level regions and computes the reference graph between them |
| `promote_internal.py` | lifts shared internals out of anonymous namespaces into an internal header |
| `split_file.py` | splits one translation unit into several along a region partition |
| `classify_tests.py` | proposes a folder for each test from its own includes |
| `check_include_paths.py` | finds targets that reach a project include they cannot resolve |
| `survey_sizes.py` | ranks every large file by WHICH of these tools can act on it |

Run every tool's `--selftest` first; `split_file.py` and `promote_internal.py` run theirs
automatically before doing anything.

## The verification each one offers

`split_file.py` reads its outputs **back from disk** and requires that they rebuild the original byte
for byte, and separately that the region map tiles the original exactly. Together those mean no bytes
were lost. **They do not mean the result compiles** — brace balance depends on which regions went
where, and only the compiler establishes that.

`promote_internal.py` checks that the header holds every promoted region's bytes (verbatim, or with
exactly one `inline` adjustment) and that the remaining source is the original minus precisely those
spans plus one include.

## Ruled out

* **`clang-refactor extract` cannot do extract-method — but `clangd` can, and that distinction is
  the whole point.** They are different implementations and only one is usable.

  `clang-refactor extract`'s own help marks it *"(WIP action; use with caution!)"*, and the caution
  is the whole story: it does not compute captured variables. Measured on a four-line function --
  extracting a loop that reads `v` and accumulates into `total` produced `static void accumulate()`
  taking **no parameters**, referencing both names undeclared, and discarding the mutation. It moves
  text.

  **clangd's `ExtractFunction` tweak does it correctly.** Same input, driven over LSP:

  ```cpp
  void extracted(const std::vector<int> &v, int &total) { for (int x : v) { ... } }
  extracted(v, total);
  ```

  `v` by const reference because it is only read, `total` by reference because it is mutated. That is
  the analysis an IDE performs, and clangd has it. It is **not** reachable from a command line:
  clangd exposes refactorings only as LSP code actions carrying a `clangd.applyTweak` command, so a
  client has to speak the protocol and catch the `workspace/applyEdit` the server sends back.
  Confirmed working against this project's own `compile_commands.json` on
  `src/gpu/execute/gpu_dependency_graph.cpp`.

* **`clang-move` is class-oriented**, so it cannot move free functions or anything in an anonymous
  namespace — which is most of what these files are.

## What these tools do NOT address, and it is the larger half

They move *structure*. They cannot shrink a **function**, and this codebase's real problem is partly
function size. Measured after the recompiler split landed:

| function | lines | file | share of its file |
| --- | --- | --- | --- |
| `register_live_renderer` | 8,221 | `frontends/shared/live/live_renderer.cpp` | 89% |
| `emit_alu` | 7,544 | `src/gpu/recompiler/rdna2_emit_alu.cpp` | 97% |
| `execute_item` | 4,536 | `frontends/shared/live/live_compute.cpp` | 47% |
| `emit_cfg_state_machine` | 4,419 | `src/gpu/recompiler/rdna2_emit_cfg.cpp` | 64% |
| `main` | ~12,400 | `tests/gpu/recompiler/test_rdna2_to_spirv.cpp` | 98% |

For `live_renderer.cpp` a structural split can achieve **nothing**: the file is one function. Moving
it elsewhere renames the problem. Those files need extraction — and per the ruled-out section above,
clangd **does** perform the required analysis, so this is mechanisable rather than hand work. Two
caveats found by trying it:

* **clangd legitimately refuses many spans, and the refusals are correct.** It will not extract a
  span containing `return`/`break`/`continue` (an early exit cannot be expressed as a call), nor one
  declaring a variable used after the span (that needs an out-parameter or a returned struct). Giant
  functions here are dense with both, so a candidate-finder must filter for what clangd accepts
  before offering it a span.
* **clangd refuses to extract from inside a lambda, and that rules out this codebase's giant
  functions.** From `clang-tools-extra/clangd/refactor/tweaks/ExtractFunction.cpp`, the tweak
  returns `nullptr` when a `LambdaExpr` is found — the comment is literally *"Don't extract from
  lambdas"*. `register_live_renderer` spans 8,221 lines of which **98.5%** is inside lambdas (8,101 of 8,221, from the AST — 89% is the
  file-share figure in the table above, a different number) passed to registration calls, so every candidate worth extracting is in refused territory. (Those counts are from an earlier commit and the file has grown since; `survey_sizes.py` measures the same function at 10,244 lines and 98% lambda on 2026-09-10. The conclusion is unchanged, and the two figures are kept distinct rather than reconciled because they were taken at different heads.)
  `extract_function.py --probe` reports 0 of 10 accepted at its default bounds, and that is correct
  behaviour, not a bug. **That figure alone does not establish it**: a hand-built lambda-free
  25-line function also probes 0 of 1, while the same clangd session returns
  `['Extract to function']` for a sub-range of it — so the zero is partly produced by the
  candidate geometry this tool generates. The conclusion rests on clangd's own source and on the
  hand-built lambda control, not on the count.

  Its other documented refusals matter for the same reason: **`requiresHoisting`** (*"cannot extract
  declarations that will be needed in the original function after extraction"*), an unmatched
  `break`/`continue`, a conditional `return`, a **templated** enclosing function, and extracting the
  whole function body. Between them they exclude most spans in code shaped like this.

  So clangd extract-method is genuinely useful — for normal-sized functions, where it does the
  capture analysis correctly and safely. It is **not** the route to shrinking an 8,000-line lambda
  nest. That needs either restructuring the lambdas into named functions first (by hand, since that
  is the very operation being asked for), or an IDE plugin driving a refactoring engine without the
  lambda restriction. JetBrains' shipped MCP server does not offer one: across the IDE family its
  entire refactoring surface is `rename_refactoring`, with AST-based refactoring an open request
  (YouTrack LLM-25880, state Open). That issue's examples name rename, extract *variable*, extract
  *parameter* and change signature; they are introduced by *"such as"*, so they are illustrative
  rather than a scope boundary, and extract-method is neither promised nor excluded by them.
  A community PyCharm plugin advertises Extract Method over HTTP, which shows the shape of that
  route — but it is a STUB: its `PyExtractMethodUtil.extractMethod` call is commented out and it
  returns success with an empty parameter list, which is where its advertised "automatic parameter
  detection" comes from. (`extractVariable` in the same file is inert too.) No IDE-driven example
  of this route was found working during this survey — which is a statement about what was checked,
  not a proof that none exists — so the route's cost has to be treated as unmeasured rather than low.

It stays incremental work regardless: a few extractions at a time, verified, by whoever is already
changing that code.

## Which files are in which state -- measured, 2026-09-10

The table above was written from a handful of files looked at by hand, and it left the impression
that the giant functions are mostly beyond tooling. **They are not.** `survey_sizes.py` measures the
lambda share of every dominant function in the tree, which is the datum that decides it, and across
28 files at or above 2,500 lines the split is:

| verdict | files | meaning |
| --- | --- | --- |
| `EXTRACT` | 13 | one dominant region, but lambda-light -- clangd's ExtractFunction can act |
| `SPLIT` | 9 | no dominant region -- `split_file.py` applies today |
| `UNPARSED` | 4 | a large span the AST never saw -- see below; no AST tool can act until it is split |
| `HAND` | **1** | dominant AND mostly lambda -- clangd refuses, no tool can help |
| `PARSE-FAIL` | 1 | not scored: `prosper-app/main.cpp` needs `-DPROSPER_APP=ON` to have a command |

Exactly one file is in the untouchable state: `frontends/shared/live/live_renderer.cpp`, whose
`register_live_renderer` is 10,244 lines at **98% lambda**. Everything else has a route.

The measurement moved three specific beliefs:

* **`rdna2_emit_alu.cpp`'s 8,186-line `emit_alu` is only 11% lambda**, so it is `EXTRACT`, not hand
  work. So are the giant test `main`s -- `test_rdna2_to_spirv.cpp` at **3%**,
  `test_dynfetch_fold.cpp` at 6%, `test_recompile_coverage.cpp` at 2%. The per-block assertion-delta
  invariant recorded at the end of this file is exactly the tool for those.
* **The largest source file is plain `SPLIT` work** -- `gpu_executor.cpp`, 12,058 lines, biggest
  region 23%.
* **A file of many small functions is the EASIEST split, and the first version of this tool called it
  `OK`.** It required two regions of >=200 lines before offering `SPLIT`, which ranked
  `hle_kernel_mem.cpp` as nothing-to-do. Region COUNT is what a seam needs; region SIZE is not.
  Fixed, and pinned by a selftest case carrying that file's real shape.

### Three ways this survey measured the wrong thing, and all of them failed silently

None of them produced an error or an odd-looking row. Each produced a confident verdict about a file
other than the one asked about, which is why they are recorded here rather than in a commit message.
All three are now pinned by fixtures under `testdata/`, and each fixture is built so the defective
measurement yields a **different verdict**, not merely a different number -- verified by re-running
each pre-fix version against all of them, where every mutation reddens its own fixture and leaves
the others green (`dominant_no_lambda` -> `HAND` at a 0.729 share for a function with no lambda;
`midfile_inactive_arm` -> `EXTRACT` with 0 unparsed, naming `posix_only_1` -- the first declaration
after the skipped arm, whose region had absorbed it; `namespace_without_children` -> `NO-CURSOR`).
The same absorption in the tree names an `#include` rather than a function (`guest_write_watch.cpp`'s
`sched.h`); which declaration a swallowed arm ends up attributed to is simply whichever cursor
follows it.

**The lambda share was measured over the enclosing namespace.** Nearly every file here is one
`namespace prosper { ... }`, so a file's only *top-level* cursor is the namespace -- and a namespace
answers `is_definition()` with True. Walking top-level cursors to find the dominant region's cursor
therefore always found the namespace, and unioned every lambda in the whole file into the dominant
function's share. Measured across the tree: **6 of the 14 files that have a lambda share at all carried an inflated
one**, worst `rdna2_emit_cfg.cpp` at 0.428 against a true **0.258**. Fourteen is the honest
denominator, and it is not the number of scored files: the share is computed only for a file a single
region dominates, so the 9 `SPLIT` files could never have been inflated, and `live_renderer.cpp` --
which was -- is neither `SPLIT` nor `EXTRACT`. So the error rate among files where the question is
asked is 43%, not the 27% a count against all scored files suggests. **No verdict actually changed** -- in a
dominated file the namespace is nearly the dominant function, so the error stayed under the 0.50
threshold everywhere. That is luck, not design: 0.428 is one small edit from flipping a file to
"no tool can act on this". The fix descends to the region's own cursor (`region_cursor`).

**`region_cursor` did not mirror `regions_of`'s descent rule.** `emit()` descends into a namespace
only when it HAS in-target children (`map_symbols.py:174`); the first version of `region_cursor`
descended on `depth < max_depth` alone. A namespace with no in-target children is emitted as a plain
`body` region by the kids-empty fallthrough, so it can BE a file's dominant region -- and the lookup
walked straight past it and reported `NO-CURSOR`, refusing to score a file whose answer was not in
doubt. Reachable with nothing exotic: a namespace whose body is all comments
(`testdata/namespace_without_children.cpp`, whose `commentary` region is 93 of 110 lines, 84.5%). No file in the tree currently hits
it, so this cost nothing -- it was found by constructing the case rather than by observing one, and
it is recorded because `NO-CURSOR` firing on a false alarm would have read as a tool limitation
rather than a tool bug.

**Unparsed lines were inferred from region kinds, which cannot see a mid-file `#if`.** `map_symbols`
tiles a file by cursor, so an inactive arm is never a region of its own -- it is folded into whichever
region follows it and wears that region's kind. Keying on `TEXT`/`TRAILER` therefore saw an arm only
when it ran to the END of the file. `src/host/memory/guest_write_watch.cpp` is the tree's own
counter-instance: a 481-line `#ifdef _WIN32` arm at lines 71-551, 19% of the file, reported as **0**
unparsed lines with a dominant "symbol" of `sched.h` -- the include directive whose region had
absorbed the arm.

The fix asks the preprocessor instead, through `clang_getSkippedRanges` (reached by ctypes; the
Python bindings do not wrap it), and **sums every skipped arm** rather than counting only ones
individually above the threshold. That matters twice over: it found three more files than the
trailing-span heuristic did, taking `UNPARSED` from 1 to 4 --

| file | lines | unparsed | was |
| --- | --- | --- | --- |
| `src/hle/memory/hle_kernel_mem.cpp` | 7,637 | **52%** (7 spans) | UNPARSED, at 51% from one span |
| `src/hle/kernel/hle_kernel.cpp` | 5,395 | **30%** | SPLIT |
| `src/hle/fs/hle_file.cpp` | 4,560 | **21%** | SPLIT |
| `src/host/memory/guest_write_watch.cpp` | 2,513 | **20%** | SPLIT |

-- and it fails **closed** everywhere the question cannot be answered, because "no unparsed lines"
and "I could not ask" must never be the same output. A missing `clang_getSkippedRanges` exits; a file
the translation unit never read, or a NULL range list, scores `NO-PREPROC` rather than 0.

That second guard was itself dead when first written, which is worth recording because it is this
PR's own defect class appearing inside the fix for it. The check was `if File.from_name(...) is
None` -- and it never is. `clang_getFile` resolves through the file manager, so it answers for any
file that exists on disk whether or not the TU read it, and cindex asserts the handle is non-NULL
besides. A foreign file therefore came back with a valid handle, an empty skipped-range list, and a
confident **0**, behind a guard that could not fire. Membership is tested against the TU's own
inclusion record now (`_file_in_tu`), which also turns an existing assumption into a check: a header
has no compile command, so `flags_for` parses the first TU whose *text* mentions it, without
confirming the include sits on an active branch.

The subtlest of those was a path comparison. The clip that keeps other files' skipped include guards
out of the count originally compared `resolve()`d paths -- and this repository is reachable under two
spellings that resolve **differently** and name one file, the container's `/home/<user>/...` bind
mount and the host's `/var/home/<user>/...`. Under the alias spelling every span was discarded and
`hle_file.cpp` reported **0** skipped lines instead of 963, which would have scored it `SPLIT`. The
comparison is by **inode** now (`os.path.samefile`), and where even that errors it counts the span
rather than dropping it: an over-count raises `UNPARSED`, which someone sees, and an under-count is
the silent wrong answer. Measured both spellings: 963 and 963.

Span line numbers include the `#if` and `#endif` directives themselves, because that is what libclang
reports and they are lines the AST produced nothing for. Anyone reconstructing these figures as
"lines of skipped CODE" will get about two fewer per span.

`UNPARSED_SHARE` (0.15) is a **convention, not a measurement**. The tree has four files above it, the
lowest at 20%, and nothing here pins where in the gap below that the line belongs.

### Why `UNPARSED` is its own verdict rather than a footnote

**Every AST-driven statement about such a file is answering about the other part.** The first version
of this tool scored `hle_kernel_mem.cpp` as `SPLIT` from the 48% it could see -- and worse, discarded
the trailing arm as scaffolding first, because `map_symbols` gives the outermost namespace's trailing
text the `close` role and this tool filtered `open`/`close` out. 3,909 lines left the accounting
silently. Caught by cross-checking the survey's own numbers against `map_symbols.py` on the same
file; nothing in the survey's output looked wrong on its own, which is the point.

The practical consequence for #3503: **that split cannot be driven by `split_file.py`**, because the
tool is blind to the arm being moved. A platform-arm split is a textual line-range move at the
`#if`/`#else`/`#endif` boundaries -- provable by concatenation, and needing no AST at all.

`survey_sizes.py` reports a file whose parse emitted errors as `PARSE-FAIL` and gives it **no
verdict**, because a libclang parse driven from a g++ database fails softly -- a partial AST looks
like a small, uninteresting file. In a survey that is the worst place for it to hide, since the file
would be ranked as needing no work. One file is currently in that state:
`frontends/prosper-app/main.cpp`, which has no compile command unless the app is configured
(`-DPROSPER_APP=ON`); configure with it to bring that file into the survey.

### Choosing where to cut

`map_symbols.py --clusters [N]` groups the file's regions by what they actually reference, and
reports the cost of the seam it proposes:

```
== proposed seams: 9 group(s), target ~2149 lines each == -- you asked for 3, and the reference
   structure plus that size cap do not permit fewer
   252 of 2927 references cross a boundary (9%) -- that is the promote-to-header list
   83 merge(s) the references wanted were refused by the size cap; a group AT the cap is the cap
   talking, not a seam
```

The cross-boundary count is the number that matters, because those are the internal-linkage edges
the REFERENCES section warns about -- a split that separates a `static` helper from its callers
*"does not fail at review, it fails at link"*. Having the list up front turns `promote_internal.py`
into a decision instead of a compile-error hunt.

**Two things it says about itself, and both are load-bearing.** A group sitting exactly AT the size
cap is the cap talking, not a structural seam -- it stopped there because it was told to. And the
part count is what the references and the cap allow, not what you asked for; the tool says so rather
than forcing the number. Grouping by NAME finds none of this: on `gpu_executor.cpp` a name-prefix
pass put 183 of 273 regions in "other".

**A 0% cut is a tell, not a success.** An early draft let attachment ignore the cap, which funnelled
every group into the largest connected component -- one 6,374-line "part" out of 6,515 lines,
reported as a perfect zero-cut split. That is pinned by a selftest arm now, with the fixture built so
the defect changes the group COUNT rather than merely a number.

### Running it

```bash
python3 prosper/tools/refactor/survey_sizes.py --min-lines 2500     # needs libclang + a build dir
python3 prosper/tools/refactor/survey_sizes.py --selftest           # no libclang needed
```

`--selftest` runs the classifier's arithmetic cases always, and the four measurement checks **only
where libclang is importable** -- three of them score committed fixtures under `testdata/`, while the
fourth builds a hard link in a temporary directory, because the property it needs (two paths naming
one file that `resolve()` reports as different) cannot be committed as a fixture. `--selftest` says
so explicitly, and names them, when it skips them. Those are the checks covering how the
classifier's INPUTS are measured, and every defect this tool shipped lived there rather than in the
arithmetic. The registered ctest case
(`refactor_survey_classifier`) therefore gates the arithmetic everywhere and the measurement locally.

**The invariant that makes incremental extraction safe is worth copying.** When a block was extracted
out of `test_rdna2_to_spirv.cpp`'s `main`, the thing that made it checkable was a count of assertions
*executed*, asserted as a per-block delta:

```cpp
const int before_bvh = checks;
run_bvh_checks();
if (checks - before_bvh != 8) { /* it did not run */ return 1; }
```

A global floor was tried first and was wrong in both directions at once — calibrated on a machine
with subgroup size 32, it rejected CI's subgroup-8 host, and it could not detect the very block it
was added for (1068 − 8 = 1060, and `1060 < 1060` is false). Per-block deltas have neither problem.
