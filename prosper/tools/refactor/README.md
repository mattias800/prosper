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
| `register_live_renderer` | 8,222 | `frontends/shared/live/live_renderer.cpp` | 89% |
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
  lambdas"*. `register_live_renderer` is 8,222 lines of which **98.5%** is inside lambdas (8,101 of 8,221 lines, from the AST — 89% is the
  file-share figure in the table above, a different number) passed to registration calls, so every candidate worth extracting is in refused territory.
  `extract_function.py --probe` reports 0 of 10 accepted at its default bounds, and that is correct behaviour, not a bug.

  Its other documented refusals matter for the same reason: **`requiresHoisting`** (*"cannot extract
  declarations that will be needed in the original function after extraction"*), an unmatched
  `break`/`continue`, a conditional `return`, a **templated** enclosing function, and extracting the
  whole function body. Between them they exclude most spans in code shaped like this.

  So clangd extract-method is genuinely useful — for normal-sized functions, where it does the
  capture analysis correctly and safely. It is **not** the route to shrinking an 8,000-line lambda
  nest. That needs either restructuring the lambdas into named functions first (by hand, since that
  is the very operation being asked for), or an IDE plugin driving a refactoring engine without the
  lambda restriction. JetBrains' shipped MCP server does not offer one: across the IDE family its
  entire refactoring surface is `rename_refactoring`, with extract-method an open request
  (YouTrack LLM-25880 — which names extract *variable* and extract *parameter*, NOT extract
  method, so it is not even a request for this). A community PyCharm plugin advertises Extract
  Method over HTTP, which shows
  the shape of that route — but it is a STUB: its `PyExtractMethodUtil.extractMethod` call is
  commented out and it returns success with an empty parameter list. There is currently no
  WORKING example of this route in any IDE, which raises its cost rather than lowering it.

It stays incremental work regardless: a few extractions at a time, verified, by whoever is already
changing that code.
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
