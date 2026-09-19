# Refactoring plan — structure, size, and what a tool can actually do about it (2026-09-19)

Measured on `origin/main` at `e7436666276f`, from the build's own compile database, with
`prosper/tools/refactor/{survey_sizes,census_bodies,map_symbols}.py` and the preprocessor. Every
figure here is re-derivable by the command that produced it; re-run rather than quoting, because
the three biggest files take 50–75 commits a month between them and these numbers move weekly.

This is a plan, not a status report. It says what is worst, why each thing is worst, which tool can
act on it, and in what order — and it is explicit about the two places where **no tool in this repo
or in clang can help**, because pretending otherwise is how a refactor stalls three files in.

## How the measurements were taken

```bash
cmake -S prosper -B prosper/build-linux -DGAME_DUMP=<DUMP_ROOT>/PPSA24651-app0 -DPROSPER_APP=ON -G Ninja
python3 prosper/tools/refactor/survey_sizes.py  --min-lines 700 --roots src frontends tools tests
python3 prosper/tools/refactor/census_bodies.py --min-lines 150 --top 500 --json census.json
python3 prosper/tools/refactor/map_symbols.py   --file <f> --clusters N --cap-lines M
```

`survey_sizes` ranks **files**; `census_bodies` ranks **bodies**. Both must be run from a configured
worktree, because both parse with the real compile commands. The census covered **507 TUs with 0
parse failures**, 12,153 bodies over 742 files — worth stating, because a libclang parse driven from
a g++ database fails *softly*, and a census that silently parsed nothing looks exactly like a tidy
tree. (The first run of it did exactly that: 506 of 507 TUs failed on `stddef.h` and it reported
zero large bodies. It said so loudly, which is the only reason it was caught in a minute.)

## The three findings

### 1. The shipping Vulkan backend lives under `tests/`, and it is the single largest file

`prosper/tests/fixtures/render_runner.h` is **12,702 lines**, 54 struct/class definitions and 187
namespace-scope function definitions, all inside `namespace prosper::test`. (Counted from the AST,
not by grep: a `grep -c '^inline '` gives 186 and 52, missing templates and multi-line signatures.) It is not a test fixture. It is the offscreen
Vulkan rendering backend, and the shipping frontend names `prosper::test::` **185 times over 182
lines**, none of them a comment — 181 lines in `frontends/shared/live/live_renderer.cpp` and one in
`frontends/shared/present/present_blit.cpp`:

```cpp
const VkFormat format = prosper::test::backend_color_format(surface.format);   // live_renderer.cpp:145
```

Checked with the preprocessor rather than with grep (`-M` over each production TU's own compile
command, 0 failures): exactly **2 production TUs** compile it, against ~45 test TUs. A grep for the
filename returns 84 files and most of those are comments and docs — the two numbers answer different
questions and only the preprocessor answers this one.

`tests/fixtures/AGENTS.md` already opens with *"`render_runner.h` is NOT test-only, and the directory
name is the trap"*, and `docs/ARCHITECTURE.md` describes a graphics pipeline that ends at
`vk_translate` and never names where the Vulkan backend lives. So the project has documented the lie
rather than fixed it, and #3210 had to say "this is not a test-only change" in its own PR body. That
is the exact failure the user-facing requirement names: you cannot find the renderer by looking at
folders.

It is also, separately, the most expensive header in the build. Total in-repo header parse cost,
measured as *lines × TUs that compile it* over the whole compile database:

| parsed lines | lines | TUs | header | share of in-repo parsing |
| ---: | ---: | ---: | --- | ---: |
| 596,994 | 12,702 | 47 | `tests/fixtures/render_runner.h` | 21.6% |
| 368,016 | 2,788 | 132 | `src/gpu/execute/gpu_execute.hpp` | 13.3% |
| 201,912 | 1,128 | 179 | `src/gpu/resources/shader_resources.hpp` | 7.3% |
| 144,144 | 819 | 176 | `src/gpu/recompiler/rdna2_to_spirv.hpp` | 5.2% |
| 139,356 | 948 | 147 | `src/gpu/pm4/pm4_registers.hpp` | 5.0% |

Two headers are **35% of everything this repository parses of its own source** (2,765,765 lines
total). 31 of 809 TUs could not be preprocessed, so 47 is a floor, not an exact count.

### 2. Three functions are 12% of the shipping codebase

`src/` and `frontends/` are 218,627 lines across **5,684 named function definitions**. The
distribution is not a long tail, it is a cliff:

| named functions | count | lines held | share of shipping code |
| --- | ---: | ---: | ---: |
| ≥ 5,000 lines | 3 | 25,827 | 11.8% |
| ≥ 2,000 lines | 5 | 33,465 | 15.3% |
| ≥ 1,000 lines | 10 | 41,093 | 18.8% |
| ≥ 200 lines | 82 | 68,000 | 31.1% |

Eighty-two functions — 1.4% of the functions in the emulator — hold just under a third of it. The
figure *excludes* `render_draw_pass_rgba` (6,400 lines), because that ships from under `tests/` and
so falls outside the denominator; finding 1 and finding 2 are the same problem seen twice.

### 3. Classes are **not** the problem, and the plan should not spend effort there

Two classes are large, and after them the distribution collapses:

| lines | class | location |
| ---: | --- | --- |
| 4,338 | `SpirvCompute` | `src/gpu/recompiler/rdna2_to_spirv_internal.hpp:385` — 229 methods, 212 fields |
| 2,181 | `VulkanComputeContext` | `frontends/shared/live/live_compute.cpp:1703` |
| 630 | `Fixture` | `tests/gpu/execute/test_storage_output_conflicts.cpp:85` |
| 601 | `Sdl3AudioSink` | `frontends/audio_sdl3/audio_sink_sdl3.cpp:245` |

Third place is 630 lines. This is worth writing down because "large files, large classes, large
functions" is the natural way to state the goal and only two thirds of it is true here: budget for
functions and files, and treat classes as two specific jobs rather than a category.

## The worst offenders

### By body — what actually has to shrink

`verdict` is `census_bodies.py`'s, and it is a statement about which tool applies:
**EXTRACT** clangd's `ExtractFunction` can act; **HAND** it cannot, because it refuses to extract
from a lambda; **SEAMLESS** there are too few top-level statements to cut — the body is one switch,
and its seams are *inside* the statements.

| lines | lam% | stmts | nest | verdict | body | location |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 10,748 | 98% | 45 | 15 | HAND | `register_live_renderer` | `frontends/shared/live/live_renderer.cpp:1356` |
| 10,137 | 61% | 179 | 14 | HAND | *lambda in* `register_live_renderer` | `live_renderer.cpp:1964` |
| 8,648 | 11% | **7** | 10 | SEAMLESS | `emit_alu` | `src/gpu/recompiler/rdna2_emit_alu.cpp:241` |
| 6,431 | 10% | 148 | 10 | EXTRACT | `execute_item` | `frontends/shared/live/live_compute.cpp:6468` |
| 6,400 | 15% | 524 | 11 | EXTRACT | `render_draw_pass_rgba` | `tests/fixtures/render_runner.h:5926` |
| 4,977 | 100% | 5 | 13 | HAND | *lambda in* `register_live_renderer` | `live_renderer.cpp:2951` |
| 4,939 | 25% | 338 | 7 | EXTRACT | `emit_cfg_state_machine` | `src/gpu/recompiler/rdna2_emit_cfg.cpp:1347` |
| 2,699 | 12% | 117 | 9 | EXTRACT | `resolve_dynamic_fetch` | `src/gpu/execute/gpu_executor.cpp:3332` |
| 1,974 | 21% | 183 | 9 | EXTRACT | `main` | `frontends/prosper-app/main.cpp:1367` |
| 1,635 | 5% | 50 | 8 | EXTRACT | `fault_handler` | `src/host/image/exec_image_linux.cpp:1344` |
| 1,496 | 42% | 22 | 7 | EXTRACT | `emit_body` | `src/gpu/recompiler/rdna2_emit_cfg.cpp:6292` |
| 1,425 | 26% | 83 | 6 | EXTRACT | `deserialize_gpu_capture` | `src/gpu/capture/serialize/capture_deserialize.cpp:53` |
| 1,098 | 1% | **3** | 6 | SEAMLESS | `apply` | `src/gpu/pm4/command_processor.cpp:4273` |
| 958 | 5% | 8 | 8 | EXTRACT | `realize_compute_dispatches` | `src/gpu/execute/gpu_executor.cpp:8174` |
| 941 | 5% | 36 | 8 | EXTRACT | `build_stage_table` | `src/gpu/execute/gpu_executor.cpp:6968` |

**Read the `stmts` column before planning any extraction.** `emit_alu` is 8,648 lines with seven
top-level statements and 378 `case` labels across 37 switches; there is no run of siblings to pull
out, so statement-run extraction has nothing to offer it and a tool that says otherwise is wrong.
`apply` in the command processor is the same shape at 1,098 lines and three statements.

**`register_live_renderer` is 98% lambda and nests fifteen braces deep.** clangd's `ExtractFunction`
returns `nullptr` when it finds a `LambdaExpr` — *"Don't extract from lambdas"*,
`ExtractFunction.cpp` — so the entire 10,748 lines is refused territory, and so is the 10,137-line
lambda inside it. This is the one file where no tool in this repository or in clang helps at all.

The test suite is worse, and easier: 127 test bodies over 200 lines hold 109,112 lines, and
`tests/gpu/recompiler/test_rdna2_to_spirv.cpp`'s `main` is 13,060 lines of **4,106 top-level
statements** — the most mechanisable body in the tree.

### By file — where `split_file.py` applies today

`survey_sizes.py` classified 103 files over 700 lines: **SPLIT 35, EXTRACT 55, HAND 1, UNPARSED 10,
PARSE-FAIL 2.** The SPLIT set is the part that can be done now, with a tool that proves it lost no
bytes:

| lines | regions | biggest | share | file |
| ---: | ---: | ---: | ---: | --- |
| 12,399 | 329 | 2,700 | 22% | `src/gpu/execute/gpu_executor.cpp` |
| 6,394 | 510 | 390 | 6% | `src/hle/service/hle_service.cpp` |
| 5,486 | 274 | 1,099 | 20% | `src/gpu/pm4/command_processor.cpp` |
| 4,414 | 316 | 369 | 8% | `src/hle/audio/hle_audio.cpp` |
| 4,189 | 266 | 1,643 | 39% | `src/host/image/exec_image_linux.cpp` |
| 3,936 | 296 | 165 | 4% | `src/hle/graphics/hle_agc.cpp` |
| 3,881 | 180 | 596 | 15% | `src/gpu/timeline/gpu_timeline.cpp` |
| 2,788 | 200 | 893 | 32% | `src/gpu/execute/gpu_execute.hpp` |

Ten more files are **UNPARSED**: over a fifth of each sits behind an inactive `#if`, so no
AST-driven verdict about them is honest. `src/hle/memory/hle_kernel_mem.cpp` is 8,083 lines of which
**4,196 (52%) is Windows-only code** the Linux parse never sees; `src/host/image/exec_image_win.cpp`
is 99% inactive on Linux. Those are a platform-split job (`*_linux.cpp` / `*_win.cpp` beside a
shared header), not a size job, and doing that first is what makes them measurable at all.

### By folder — the tree does not answer "where does this live?"

- **45 of 96 directories holding three or more code files have no `AGENTS.md`**, including
  `src/hle/kernel` (8,611 lines), `src/hle/fs` (5,161), `tests/gpu` (20,814) and `tests/hle`
  (12,824). The charter requires one per folder holding real content.
- **`src/hle/service/hle_service.cpp` is fifteen Sony libraries in one file**, each with its own
  banner comment: Videodec2, Vdecsw, AvPlayer, Psml, NP, Ime, ImeDialog, AppContent,
  SaveDataDialog, NpTrophy2, PlayGo, SaveData, save-data memory, UserService, mouse. To find
  `sceAvPlayerStart` today you grep. After, you open `src/hle/video/avplayer.cpp`.
- **`tests/` is half-classified**: the subsystem subdirectories exist, and 57 files still sit loose
  in `tests/hle`, 26 in `tests/gpu`, 16 in `tests/` itself. `tests/misc/` holds boot, loader,
  present and trap tests that all have homes. `classify_tests.py` proposes a destination for 339 of
  355 files today (26 ambiguous, 16 with no project include at all).
- **`tests/` is larger than the emulator**: 167,299 lines against `src/`'s 161,282.
- One directory is a naming claim that is false in the other direction: `src/gpu/execute/mb3_freelist`
  has no caller in its own folder, as its own `AGENTS.md` records.

## Performance: what the plan must not break, and the one fact that decides it

**There is no LTO.** The build is `-O3 -DNDEBUG -std=gnu++20` with no `-flto` anywhere in
`CMakeLists.txt` or the generated compile commands (checked twice, by reading a compile command and
by searching the build system). So:

> **Moving a hot helper into a different translation unit turns an inlined call into a real one, and
> nothing will inline it back.** Extraction *within* a TU stays inlinable; splitting a TU does not.

That single fact orders the whole plan, because it separates two jobs that look identical in a diff:

- **Cold by construction.** Shader recompilation is cached per compile key
  (`recompile_graphics_shader_cached_shared` → `shader_cache()`), so `emit_alu`,
  `emit_cfg_state_machine`, `decode_operands` and `SpirvCompute` run **once per unique program**,
  not per frame. Restructure these freely.
- **Per-draw and per-submit.** `execute_item`, `render_draw_pass_rgba`, `realize_draw_item`,
  `build_stage_table`, `execute_ordered_gpustate`, `resolve_dynamic_fetch`. Every split here needs a
  measurement.

### The guard, in order of cost

1. **Symbol-level A/B, no GPU and no game.** Build `prosper_core` before and after and compare `nm`
   output. A symbol that was previously *absent* (inlined away) and is now present and global is a
   new cross-TU call boundary — exactly the no-LTO risk, made visible for the price of one compile.
   This is the gate for every split; it is cheap enough to run every time.
2. **`PROSPER_RENDER_TIMING` phase buckets** on a routed run, per the existing harness.
3. **F8 `.prperf` capture** (`PROSPER_PERF_CAPTURE_AFTER_MS`), read with
   `tools/perf/performance_capture_report.py`. Check the capture can *see* the leaf being claimed —
   the 2.4%-vs-67% correction in the charter is what happens when it cannot.
4. **Frame rate**, windowed and uncapped (`--present-mode immediate`), quoting `distinct` rather
   than `presented`, dropping the samples either side of any capture. `tools/perf/ab_compute.sh`
   already refuses to measure against a contended GPU; a two-build variant of it is listed under
   *Tool work* below.

Steps 2–4 are required only for a body on the per-draw or per-submit list. Step 1 is required for
every split, including the cold ones, because it is also the cheapest way to notice that a "pure
move" was not one.

## The plan

Ordered by value per unit of risk, not by size. Each phase names its verification.

### Phase 0 — make the map honest (docs only, no code)

Write the 45 missing `AGENTS.md` files, worst-first by line count. This is the cheapest thing on the
list and it directly serves "find it by looking": a folder that says what belongs in it is what
makes the next agent put the file in the right place instead of beside whatever they had open. Keep
them maps, not documentation of the code — `tests/fixtures/AGENTS.md` has grown into the second kind
and should be cut back when its subject moves in Phase 2.

*Verification:* the docs gate, and the charter's rule that a folder holding one file or a leaf whose
name says everything does not need one.

### Phase 1 — the recompiler: biggest bodies, coldest path, lowest risk

The 2nd, 4th, 7th and 11th largest bodies in the tree are here, plus the largest class, and the
shader cache means none of them is per-frame.

1. **`SpirvCompute` (4,338 lines, 229 methods, 212 fields) → several classes by responsibility**, in
   `src/gpu/recompiler/spirv/`. A header parsed by 11 TUs, so this is also a build-time win.
2. **`emit_alu` (8,648 lines) → one file per instruction family** — VOP1/VOP2/VOP3/VOPC, SOP1/SOP2/
   SOPC/SOPK, VOP3P, DPP/SDWA. It is SEAMLESS, so this is *not* extract-method: it is lifting switch
   arms into named functions, which is the mechanical operation `map_symbols` can plan and
   `split_file` can execute once the arms are functions. Expect `promote_internal.py` first — the
   emitters share anonymous-namespace helpers.
3. **`emit_cfg_state_machine` (4,939 lines, 338 statements, 25% lambda) → extract**, with
   `extract_function.py`, which is what it was written for.

*Verification:* the full `ctest --no-tests=error` (quote the count, not just the exit code),
`tools/spv_validate`, `tools/vkval/vk_validation_scan.py`, and `nm` A/B. A green ctest says nothing
about SPIR-V validity — run the validators before pushing any emitter change.

### Phase 2 — move the renderer out of `tests/`

`tests/fixtures/render_runner.h` → `src/gpu/backend/` (or `frontends/shared/backend/`; the
`src/`-vs-`frontends/` call is the one open design question in this plan, and it turns on whether
the backend may depend on SDL/Vulkan surface state — today it must not, per its own AGENTS.md).
`namespace prosper::test` → `prosper::gpu::backend`, and it stops being header-only.

This is the highest-value structural change in the tree and it is also the one most likely to
collide: `render_runner.h` took 53 of the 535 commits landed in the last 30 days. Sequence it as:

1. `move_module.py` for the relocation and every include rewrite (one commit, `git mv` plus include
   changes, no logic).
2. A mechanical namespace rename (second commit).
3. `map_symbols.py --clusters --cap-lines` then `split_file.py` to break the 12,702 lines into
   compiled translation units by responsibility — persistent caches, colour targets, texture upload,
   pipeline cache, the draw pass itself.
4. `render_draw_pass_rgba` (6,400 lines, 524 statements) extracted afterwards, not during.

*Verification:* `nm` A/B at each step, then the full per-draw measurement chain — this is the hot
path. Steps 1 and 2 should be genuinely behaviour-free and should show an identical symbol set
modulo the namespace name; step 3 is where inlining can move, and step 3 is the one to measure hard.

### Phase 3 — split the grab-bag files

`split_file.py` applies to all 35 SPLIT files today. Take them in this order, because the first two
are the ones where the folder structure is wrong rather than merely crowded:

1. **`hle_service.cpp` → one file per Sony library, in the folders that already hold that
   library's helpers.** This line first read `src/hle/{media,np,savedata,ime}/`, and **three of
   those four would have duplicated an existing folder** — checked by listing `src/hle/` rather
   than by guessing:

   | library group | destination | what is already there |
   | --- | --- | --- |
   | Videodec2, Vdecsw, Psml, AvPlayer | `src/hle/video/` | `video_backend.*`, `h264_sps.*` |
   | Ime, IME injection, ImeDialog | `src/hle/input/` | `ime_input.hpp`, `hle_pad.cpp` |
   | SaveData + transactions + save-data memory + params + dialog | `src/hle/fs/` | `save_param.*`, `save_paths.*` |
   | NP, Trophy2, entitlement, UDS, Share, NetCtl | `src/hle/np/` | **the one genuinely new folder** |
   | user service, app content, PlayGo, error dialog, mouse, random | `src/hle/service/` | `hle_addcontent.*`, `platform_ui.*` |

   Recorded because it is the plan's own failure mode in miniature: a restructure proposed from the
   *names of the things being moved* rather than from the tree it is moving them into invents
   folders that already exist, and the result is worse than leaving it alone.

   **The split has a measured prerequisite.** `map_symbols.py` reports 1,123 of 4,147 body-region
   references (27.1%) crossing that partition over 194 symbols — but it is three things and only
   one is real coupling: 378 crossings are the shared `svc_*` helpers (one `promote_internal.py`
   run), most of the rest are per-library error constants the line-range partition misfiled, and
   the NID handlers reached by `register_service_hle` (each library gets its own `register_X()`).
   Cheapest cut first, measured per group: AvPlayer needs **four** symbols from outside, then
   Videodec2, Ime, PlayGo, SaveData, NP last. Tracked in #3735.
2. **`gpu_executor.cpp` (12,399 lines, 329 regions)** → shader cache / compute realization / stage
   table / ordered execution+present / guest-memory probes / diagnostics. `map_symbols --clusters 8`
   reports **447 of 3,128 references crossing (14%)** — that crossing list *is* the
   `promote_internal.py` work list, and it should be a separate commit before the split.
3. `command_processor.cpp`, `hle_audio.cpp`, `hle_agc.cpp`, `gpu_timeline.cpp`, `gpu_execute.hpp`.

*Verification:* `split_file.py`'s own byte-identity and tiling checks prove nothing was lost; they
do **not** prove it compiles, and brace balance depends on which regions went where. Build, then
`ctest --no-tests=error`, then `nm` A/B, then the hot-path chain for `gpu_executor` and
`gpu_execute.hpp` only.

### Phase 4 — platform splits, and the tests tree

- Split the 10 UNPARSED files along `#ifdef _WIN32` into `*_linux.cpp` / `*_win.cpp` beside a shared
  header, starting with `hle_kernel_mem.cpp` (52% inactive) and `hle_kernel.cpp` (30%). Until this
  is done, no AST tool can see half of those files, so every later measurement of them under-reports.
- Run `classify_tests.py --plan` and move the 339 classified tests; hand-place the 26 ambiguous and
  16 unclassified. Then break the giant test `main`s: `test_rdna2_to_spirv.cpp` (13,060 lines /
  4,106 statements), `test_game_compute.cpp`, `test_dynfetch_fold.cpp`, `test_gpu_capture.cpp`.

*Verification for the test `main`s:* the invariant this project already found — a per-block delta of
assertions *executed*, `const int before = checks; run_block(); if (checks - before != N) return 1;`
A global floor was tried and was wrong in both directions at once.

### Phase 5 — `live_renderer.cpp`, last, and by hand

10,748 lines, 98% lambda, nesting depth 15, and 47 commits in 30 days. No tool applies. The only
route is to convert the registration lambdas into named functions first — which is the very
operation being asked for — so this is hand work, a few callbacks at a time, by whoever is already
changing that code. It goes last because every earlier phase reduces the amount of it, and because
it is the file most likely to conflict with an active lane.

## Tool work this plan needs

| # | tool | what is missing |
| --- | --- | --- |
| 1 | `census_bodies.py` | **written and registered in ctest as part of this plan** — ranks every function/method/class body tree-wide, which `survey_sizes.py` structurally cannot do |
| 2 | `map_symbols.py --cap-lines` | **added as part of this plan** — the cap was `total/parts`, which asserts every output should be the same size; false for a semantic split, and it made `hle_service.cpp` report 96 groups and a 39% cut where an explicit 1,200-line cap gives 25% |
| 3 | named-anchor clustering | `--clusters` is greedy over the reference matrix plus a size cap. It cannot be told "these symbols belong together", so it mixes IME with NP-entitlement in `hle_service.cpp` even at the right cap. Wanted: seed groups from a symbol list or from banner comments, then assign the rest by affinity |
| 4 | switch-arm lifting | nothing can act on a SEAMLESS body. `emit_alu` and `apply` need a tool that turns `case X: <block>` into `case X: return emit_x(...);` with the capture analysis done properly — the same analysis clangd already does, applied to a shape clangd will not accept |
| 5 | `nm` A/B harness | the cheap no-LTO guard in *Performance* above; a script that builds one target twice and reports newly-external symbols |
| 6 | two-build `ab_compute.sh` | the existing harness A/Bs a `PROSPER_*` switch on one binary and refuses a contended GPU. A refactor needs the same refusal logic across two *builds* |
| 7 | `classify_tests.py` for `tests/hle` | the include signal is weak there — everything includes `dispatch.hpp` — so 60 files still land at `hle` itself. The stronger signal is which NIDs or Sony functions the test names |

## Ruled out

* **`clang-refactor extract` cannot do extract-method.** Its own help says *"(WIP action; use with
  caution!)"*; measured, it emitted a no-parameter function for a loop reading one variable and
  mutating another. clangd's implementation is a different one and is correct, but is reachable only
  over LSP. (`tools/refactor/README.md`.)
* **clangd's `ExtractFunction` will not help with `register_live_renderer`**, and the reason is in
  its source, not in a measurement: the tweak returns `nullptr` on finding a `LambdaExpr`. 98% of
  that function is inside lambdas. `extract_function.py --probe` reporting 0 of 10 accepted is
  correct behaviour and is *not* on its own sufficient evidence — a hand-built lambda-free control
  probes 0 of 1 too, because of the candidate geometry. The conclusion rests on clangd's source plus
  that control.
* **`clang-move` is class-oriented**, so it cannot move free functions or anything in an anonymous
  namespace, which is most of what these files are.
* **Statement-run extraction has nothing to offer `emit_alu`.** Not a tool limitation: the body has
  seven top-level statements. Measured by `census_bodies.py`, and independently visible as 378 `case`
  labels in 37 switches.
* **Splitting files does not, on its own, make functions smaller** — and for `live_renderer.cpp` it
  achieves nothing at all, because 89% of the file is one function. The two jobs need different
  tools and the plan keeps them in different phases.
* **A restructure planned from symbol names invents folders that already exist.** Three of the four
  destinations in this plan's first draft (`src/hle/{media,savedata,ime}/`) duplicated
  `src/hle/{video,fs,input}/`, each of which already held the moved library's own helpers. Ruled out
  by `ls src/hle/`, which is the check that should precede proposing any destination.
* **The AST's nesting depth is not brace depth.** `census_bodies.py` counts `COMPOUND_STMT` only; a
  flat 197-line function reports AST depth 21 and brace depth 6. Verified against a textual
  brace-balance count on `linker.cpp:56-252` — both give 6.
