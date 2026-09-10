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
  file-share figure in the table above, a different number) passed to registration calls, so every candidate worth extracting is in refused territory.
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
| `SPLIT` | 12 | no dominant region -- `split_file.py` applies today |
| `EXTRACT` | 13 | one dominant region, but lambda-light -- clangd's ExtractFunction can act |
| `UNPARSED` | 1 | a large span the AST never saw -- see below; no AST tool can act until it is split |
| `HAND` | **1** | dominant AND mostly lambda -- clangd refuses, no tool can help |

Exactly one file is in the untouchable state: `frontends/shared/live/live_renderer.cpp`, whose
`register_live_renderer` is 10,244 lines at **99% lambda**. Everything else has a route.

The measurement moved three specific beliefs:

* **`rdna2_emit_alu.cpp`'s 8,186-line `emit_alu` is only 11% lambda**, so it is `EXTRACT`, not hand
  work. So are the giant test `main`s -- `test_rdna2_to_spirv.cpp` at **3%**,
  `test_dynfetch_fold.cpp` at 6%, `test_recompile_coverage.cpp` at 2%. The per-block assertion-delta
  invariant recorded at the end of this file is exactly the tool for those.
* **The two largest source files are plain `SPLIT` work** -- `gpu_executor.cpp` (12,058 lines,
  biggest region 23%) and `hle_kernel_mem.cpp` (7,637 lines, biggest region **2%**, #3503).
* **A file of many small functions is the EASIEST split, and the first version of this tool called it
  `OK`.** It required two regions of >=200 lines before offering `SPLIT`, which ranked
  `hle_kernel_mem.cpp` as nothing-to-do. Region COUNT is what a seam needs; region SIZE is not.
  Fixed, and pinned by a selftest case carrying that file's real shape.

### The AST cannot see the inactive side of an `#if`, and that is its own state

`hle_kernel_mem.cpp` is 7,637 lines of which the Windows arm is **3,909 -- 51% of the file** -- behind
an `#if defined(__linux__) || defined(__APPLE__)`. Parsed on Linux, clang never sees it: the region
map runs out at line 3720 and the whole Windows arm arrives as a single `TRAILER`.

This is worth its own verdict rather than a footnote, because **every AST-driven statement about such
a file is answering about the other half**. The first version of this tool scored the file `SPLIT`
from the 49% it could see -- and worse, it discarded the trailer as scaffolding first, because
`map_symbols` gives the outermost namespace's trailing text the `close` role and this tool filtered
`open`/`close` out. 3,909 lines vanished from the accounting silently. Caught by cross-checking the
survey's own numbers against `map_symbols.py` on the same file; nothing in the survey's output looked
wrong on its own, which is the point.

The practical consequence for #3503: **that split cannot be driven by `split_file.py`**, because the
tool is blind to the arm being moved. A platform-arm split is a textual line-range move at the
`#if`/`#else`/`#endif` boundaries -- provable by concatenation, and needing no AST at all.

`survey_sizes.py` reports a file whose parse emitted errors as `PARSE-FAIL` and gives it **no
verdict**, because a libclang parse driven from a g++ database fails softly -- a partial AST looks
like a small, uninteresting file. In a survey that is the worst place for it to hide, since the file
would be ranked as needing no work. One file is currently in that state:
`frontends/prosper-app/main.cpp`, which has no compile command unless the app is configured
(`-DPROSPER_APP=ON`); configure with it to bring that file into the survey.
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
