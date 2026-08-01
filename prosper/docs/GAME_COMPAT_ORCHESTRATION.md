# Game compatibility orchestration

This is the durable playbook and current handoff for concurrent game-compatibility work. It is written for an
orchestrating agent that delegates one bounded investigation to each subagent, integrates real progress, and keeps
the shared repository, GPU, evidence, issues, and pull requests coherent.

Read the repository-root `CLAUDE.md` before this document. `CLAUDE.md` remains authoritative for filesystem,
verification, evidence, review, privacy, and release policy. This document explains how to apply those rules when
several game agents work at once.

The current-state sections below are dated **2026-07-31**. Update them whenever an investigation materially moves,
a PR merges, or ownership changes. Do not let this document replace issue comments: issues are the durable evidence
log, while this document is the map that helps the next orchestrator find and interpret that evidence.

## Current checkpoint

- Repository: `mattias800/prosper` (renamed from `mattias800/ps5ys`).
- Remote branch: `master`.
- Exact master at this update: `ede320b6` (2026-07-31 session; eleven merges).
- Active title lanes: Astro Bot, Dragon Quest VII, The Plucky Squire, Alex Kidd, and The Pathless.
- GPU state: sharing is the default; see the scheduling rules below.

**Read the falsification record before planning anything.** The three lanes handed over on `81a9548e` each had
their assigned premise **falsified** during the 2026-07-31 session, including both items the previous handoff
called its highest-value work. Re-running them wastes a session:

| Previous "highest-value" item | Outcome |
|---|---|
| Astro op221 R11G11B10 storage lowering | **FALSIFIED.** Stored (packed R32ui) and current-raw (native typed float) take provably different backend paths yet produce byte-identical output `28b5ea9a3fbd2ca9`. |
| Dragon Quest binding-36 sampling | **EXONERATED entirely.** LUT contents, tile27 detile, DCC, 10:10:10:2 unpack, 3D coordinate wiring, dimensional opcode, dmask, swizzle, address modes, sampler filtering, and draw90's shader translation are all ruled out. |
| Plucky pc915 barrier-spanning branch | **PROVEN workgroup-uniform and merged** (#1572). |

Two lanes were added because the inherited allocation was skewed to the hardest titles while easier ground sat
idle: **Alex Kidd** (PPSA02664) and **The Pathless** (PPSA01826). Alex Kidd reached the gameplay rung in one
session. Prefer breadth when a hard lane stalls — overall library progress matters more than any single title.

### Tooling hazards that cost real time this session

1. **Build inside the `ps5ys` distrobox.** The host has only the Vulkan *loader*, not the headers. A host
   configure **succeeds while silently omitting** `gpu_replay`, the offscreen graphics harness, and every Vulkan
   **execution** test, logging just `Vulkan not found`. A green host `ctest` is therefore not evidence that
   execution tests ran — confirm the binaries exist and report test names, not an exit code. Wipe the build dir
   when switching; a stale cache keeps Vulkan disabled.
2. **`shader_inspect` was misreporting ~96% of known-good shaders** (109 of 114, across all three stages) as
   `rejected`, because a raw dump carries no descriptors. Fixed in #1575 — it now reports
   `undetermined-no-resource-table` with exit 3. Any shader verdict taken from it before `37dfe752` is suspect.
3. **`-DGAME_DUMP` against a non-Messenger title fails 3 dump-parameterized tests** (`module_loads_eboot`,
   `boot_reaches_first_syscall`, `real_shader_render`). This is expected, not a regression — current master is
   **162/162** when configured against `PPSA24651`. Tracked as #1573.

### The instrument-not-the-subject list — read this before believing any surprising measurement

**Thirteen phantom defects came from the measuring apparatus, not the subject** (eleven in one session on
2026-07-31, two more on 2026-08-01). Several cost hours; one cost two sessions. This is the single
highest-value page in this document.

| # | Instrument | How it lied |
|---|---|---|
| 1 | `shader_inspect` | Reported **109 of 114 known-good shaders** as `rejected` — a raw dump has no descriptors, so table-dependent MIMG/MUBUF/MTBUF (and SMEM in graphics) correctly refuse. Fixed #1575. |
| 2 | `PROSPER_FS_TAP` | Exports `(dst, dst+1, dst+2, dst+3)`, so taps with different `dst` compare **different registers**. Manufactured a "~14x amplification". |
| 3 | Snapshot guards | `blue-prince-hall` and `terminator-boot` "FAILED" on **fully-rendered healthy frames** whose routes had drifted to another scene/animation phase. A control run on master reproduced both. |
| 4 | `sgpr:106` | Is `VCC_LO` **scratch**, not a uniform — a static constant-buffer read cannot resolve its value at point of use. |
| 5 | `--through-operation N` | Renders the **last executed draw target**, so adjacent prefixes show *different surfaces*. #1486's founding "corruption boundary" was 1920x1080 vs 3840x2160. Fixed by #1580's `target=`. |
| 6 | `git checkout <file>` | Restores from the **index** — it is not an undo for a temporary edit, and silently wipes uncommitted work. |
| 7 | A stale `in-progress` label | Read as occupancy. #1284's claim had been dead six days (no open PR, no activity); CLAUDE.md expires claims at 24 h. |
| 8 | An orchestrator's instruction | "If it is a missing capability, implement it" — all three opcodes were **already implemented**; following it would have duplicated a working `image_gather4_lz`. |
| 9 | `[compute] skip unsupported program` | **Deduped per `code_addr`**: one line means "failed at least once", NOT "always fails". A program running 763 times and failing once looks identical to a permanently blocked one. |
| 10 | `PROSPER_COMPUTELOG_CODE=<addr>` | Correctly protects a realtime route from ~13k submits of logging, and thereby **suppresses `execute` lines for every other program**. Right for one question, silently wrong for the next asked of the same log. |
| 11 | SHA reachability after a **squash** merge | `git merge-base --is-ancestor` reports "unmerged" for work that *is* on master. Verify by **content**, not by SHA. |
| 12 | `PROSPER_REGWATCH` **joined to** `PROSPER_EXECLOG` by stream adjacency | The two timestamp at **different pipeline stages** — REGWATCH at command-processor *decode*, EXECLOG at *realization* — and the phases interleave per submit. Read by log order, #1606 said **1,049 of 1,050** suppressed draws resolved against a *future* register write. Clean, compelling, and entirely false. Fixed by #1633's `order=`. |
| 13 | A **print-capped** diagnostic read as a frequency | `[agc] WaitRegMem … NOT satisfied` prints the first 40 then every 1024th (`ln < 40 \|\| (ln & 1023) == 0`); `[agc] out-of-range indirect reg write dropped` stops after **4**; `[agc] indexed indirect draw skipped` after 24. #1606 called the `WaitRegMem` volume "the loudest signal, dozens per second"; the true rate is **~1.5/s** (see below). A line count from these is a cap, not a rate. Same trap as #9's dedupe, different mechanism. |

**Working rules that follow:**

- **Joining two instruments requires an explicit shared ordinal, never stream adjacency.** If two diagnostics
  are emitted at different pipeline stages, their interleaving in the log is an artifact of *when each stage
  ran*, not of when the events happened. Print a common ordinal (`command_order`) on both and join on it. This
  is why the `order=` field exists on the `[exec] skip draw early` line — **it is not redundant with the
  register values on the same line, and removing it silently re-opens the phantom above.**
- **Before quoting a diagnostic's line count as a rate, grep for its cap.** These sites use a
  `static std::atomic<int>` counter or a per-key dedupe set; check for one before concluding "this fires N
  times" or "this is rare". When the site prints its own counter — `WaitRegMem #%d` *is* `ln` — the true
  total is free: the highest `#N` in the log bounds it, and a run whose maximum stays under the cap has
  printed *every* occurrence. That read turned #1606's "dozens per second" into 18 and 21 events for two
  entire runs.

- Prefer experiments that **detect their own invalidity** — e.g. render two adjacent operations and assert both
  return the same resolution, so a surface switch invalidates the comparison instead of producing a phantom.
- **Open the image.** Believe neither a metric win nor a metric failure without looking. A diagnostic clear can
  out-score real content on colour count; a healthy frame can fail SSIM on animation phase.
- A measurement that is **right for one question can be silently wrong for the next one asked of the same data**.
- Never re-derive a bisect boundary without confirming both sides render the **same target**.
- Check whether a later comment or merged PR already **superseded** an inherited premise before acting on it
  (Bendy's "CPU software-detile dominates" had been overtaken by #1179/#1190/#1269 in its own issue).
- Write `Refs #NN`, not `Fixes #NN`, when a PR only partially addresses an issue — a parenthetical qualifier does
  **not** stop GitHub's auto-close, and #1554 was closed that way despite a comment saying to keep it open.

## The orchestration contract

### Orchestrator responsibilities

The orchestrator owns integration and scheduling, not every line of investigation. It should:

1. Read `CLAUDE.md`, this document, the title status document, and the active issue before assigning work.
2. Fetch `origin/master` and give every subagent a private worktree and a fresh branch from the exact remote head.
3. Assign one title or one precisely bounded shared-infrastructure question per subagent.
4. Prevent duplicated hypotheses by requiring agents to read the current issue evidence before running anything.
5. Keep each agent on an evidence ladder: retained offline artifact first, bounded live capture only when the artifact
   cannot answer the question, implementation only after a generic behavioral contract is identified.
6. Decide GPU scheduling. Ordinary correctness replays may overlap; reserve exclusivity only when measurements,
   memory pressure, or observed interference require it.
7. Review every proposed diff itself, including assumptions, scope, tests, `git diff --check`, and exact base/head.
8. Open short-lived PRs for proven progress, wait for every applicable CI job, confirm release publication skipped on
   an ordinary PR, then merge when authorized.
9. Keep issues and PR descriptions self-contained. Post exact commands, hashes, conclusions, and falsified hypotheses.
10. Update compatibility docs and representative screenshots when a title reaches a new visible checkpoint.
11. Rebase or restart long-lived investigation branches from current master before they drift.
12. Stop stale processes and keep evidence off the repository and off RAM-backed `/tmp`.

The current user explicitly authorized the orchestrator/subagent arrangement as pair programming and allowed the
independent-review step to be skipped for PRs produced by this coordinated work. The orchestrator still has to inspect
the complete diff, verify the exact head, wait for all CI, and merge deliberately. Treat this as engagement-specific
authorization: if a future user has not granted it, follow the independent-review default in `CLAUDE.md`.

### Subagent responsibilities

Every game subagent should receive and follow this contract:

- Work only in the assigned private worktree and worktree-local build directory.
- Start from the exact remote master SHA supplied by the orchestrator; report branch, head, merge-base, and status.
- Read the title issue and status docs before running a new experiment.
- State one falsifiable question before each run and the outcome that would distinguish the competing explanations.
- Prefer immutable F9/timeline/capsule evidence and offline `gpu_replay` over repeated game boots.
- Put captures, logs, screenshots, raw shaders, and scratch analysis under `~/`, never in git and never in `/tmp`.
- Never publish private absolute host paths. Public text uses `~/`, `<REPO_ROOT>`, `<WORKTREE>`, and `<DUMP_ROOT>`.
- Do not make title-address special cases. Shared GPU/recompiler behavior needs a generic contract and regression tests.
- Do not silently skip unsupported guest behavior. Retain fail-visible rejection until semantics are proven.
- Do not edit after the orchestrator asks for a freeze or design review.
- Report meaningful progress early: evidence paths/hashes, issue comment links, exact next blocker, and whether a code
  change is ready. Avoid holding a useful generic fix for days while exploring a later title-specific blocker.
- Commit and push a focused branch when the orchestrator approves implementation. The orchestrator owns PR creation,
  CI gating, merge, and cross-agent integration unless explicitly delegated otherwise.
- A visual milestone requires an unmodified frontend screenshot. Diagnostic substitutions may illustrate a hypothesis
  but are labeled diagnostic and never presented as progression evidence.

### Worktree and branch protocol

Never build or edit in the shared checkout. A typical assignment begins with:

```bash
git fetch origin master
git worktree add .claude/worktrees/<agent-title> \
  -b investigate/<title-question> origin/master
cd .claude/worktrees/<agent-title>
git status --short --branch
git rev-parse HEAD
```

Each worktree gets its own build directory. Put compiler temporaries on real disk:

```bash
mkdir -p prosper/build-linux/tmpdir
TMPDIR="$PWD/prosper/build-linux/tmpdir" \
  cmake -S prosper -B prosper/build-linux -DGAME_DUMP=<DUMP_ROOT>/<TITLE>-app0
TMPDIR="$PWD/prosper/build-linux/tmpdir" \
  cmake --build prosper/build-linux -j6
```

Before publishing, fetch master, inspect divergence, synchronize when needed, rerun the relevant checks on the exact
head, and report:

```text
branch / head / merge-base / worktree status
focused build and test commands with pass counts
artifact or live evidence hashes
git diff --check result
known limitations and deliberately unaffected paths
```

Do not keep coding on a branch whose useful part has already merged. Create a fresh branch from the new master for the
next hypothesis.

### GPU scheduling

Exclusive GPU ownership is **not** the default.

Safe to overlap in normal circumstances:

- `shader_inspect`, graph generation, disassembly, capture inspection, and CPU analysis;
- short deterministic correctness replays where timing is irrelevant;
- game-list or no-game frontend tests that only render light ImGui content;
- builds and non-Vulkan tests.

Ask the orchestrator for an exclusive lease when:

- profiling or reporting FPS/frame time;
- comparing small performance changes where contention would swamp the signal;
- a capture/replay has unusually high VRAM or host-memory demand;
- concurrent jobs have already caused timeouts, device loss, or unstable images;
- the user specifically asks to watch one live run without interference.

An exclusive lease has an owner, purpose, start time, and bounded expected duration. The owner announces when it is
released. Correctness replays do not become exclusive merely because they use Vulkan.

Before starting a GPU run, inspect existing processes. A Prosper process older than roughly 30 minutes is presumed
stale because known tests should finish in minutes. Resolve the exact command and owner, then terminate that exact
process safely. Never use a broad pattern that could kill another agent's unrelated work.

**Count with `pgrep -x`. Both `ps | grep` idioms are wrong, in opposite directions.** This is a safety rule, not a
style preference: a wrong count either blocks you from a free GPU or makes you kill a peer's live run.

| Idiom | Failure | Observed |
|---|---|---|
| `ps aux \| grep -cE "prosper-app\|boot_trace"` | **Over-counts** — matches the Bash wrapper shell and any `sleep` watcher whose command line contains the literal | reported **23** when the true count was **0** |
| `ps -eo comm \| grep -c '^prosper-app$'` | **Under-counts** — `ps` pads the `comm` column, so the `$` anchor never matches | reported **0** against a **live** process |

The second is the dangerous one. On 2026-07-31 an agent reported "GPU released" from that idiom repeatedly; a
blind `pkill -f prosper-app` on the strength of one such reading would have killed a **successor agent's run 35
seconds after it started**. It was caught only because the release check was re-run with `pgrep -x`.

Use:

```bash
pgrep -c -x prosper-app        # count, exact name
pgrep -ax prosper-app          # list with full command lines
readlink /proc/<pid>/cwd       # whose worktree is this?
tr '\0' '\n' < /proc/<pid>/environ | grep PROSPER_   # and whose run?
```

Before terminating anything, confirm the exact PID **and** that its `cwd`/`environ` identify it as yours. Never
kill from a pattern count. Note `comm` is truncated to 15 characters by the kernel, so a longer binary name needs
`pgrep -f` with a **bracketed** pattern (`pgrep -f "prosper[-]app"`) so it cannot match your own shell.

### Fast evidence loop

For rendering correctness, use this order:

1. Reuse an existing `.prgcap`, `.prgbundle`, timeline, raw shader, or issue attachment.
2. Run `gpu_replay --inspect-only`, graph/resource dumps, shader coverage, and source inspection.
3. Replay a bounded operation prefix or one draw/compute stage.
4. Use a narrowly scoped diagnostic substitution with a selector-miss control.
5. Add a reusable diagnostic seam only if current tools cannot separate the remaining stages.
6. Run the title live only to capture missing state or validate the integrated fix.
7. Measure performance only after correctness and with an exclusive GPU window when precision matters.

Every A/B must hold incidental behavior constant. For example, `PROSPER_TESTTEX` enables a generic CPU diagnostic
copy path, so a binding-miss run with the same environment is the control for a binding-hit run.

### Testing, PRs, releases, and screenshots

- Focused tests are part of iteration. The full snapshot matrix is optional for ordinary development and PRs; the PR
  author decides whether it is useful.
- The snapshot matrix is mandatory before every release. A release delay is acceptable; day-to-day development should
  not be delayed solely to keep master regression-free.
- Master may regress during heavy development. A correct fix may intentionally expose a different title regression.
  Record the tradeoff rather than treating a permanently green master as an end in itself.
- Artifacts are uploaded only for a release. A release is created only for a Git release tag. Ordinary PR CI must show
  the release-publication job skipped.
- Keep PRs short-lived so other agents inherit generic improvements quickly.
- The orchestrator opens the PR with a self-contained behavioral contract, issue links, evidence, exact tests, risks,
  and known limitations. Wait for all Linux, Windows, and macOS checks before merging.
- When a game changes from black to visible content, reaches title, reaches gameplay, or materially improves visuals,
  update `README.md`/compatibility docs as appropriate and attach direct unmodified frontend captures. A black frame
  needs no screenshot. A checker/forced-state/debug draw may be attached only as clearly labeled diagnostic evidence.

## Recently merged foundation

These changes are already on current master and should not be reimplemented:

| PR | Merge commit | Result |
|---|---|---|
| [#1561](https://github.com/mattias800/prosper/pull/1561) | `0b20522a9d8e212f2813d75c299d33b198908aee` | Materializes exact scalar 16-bit buffer tails. |
| [#1562](https://github.com/mattias800/prosper/pull/1562) | `cee8b1525de226551814fe55a64eb1ffa19e513c` | Records Dragon Quest VII name confirmation. |
| [#1563](https://github.com/mattias800/prosper/pull/1563) | `7f89e60b4d70f4f843be22202d396cf75b0762ed` | Records Dragon Quest VII first-run onboarding. |
| [#1564](https://github.com/mattias800/prosper/pull/1564) | `1634c1e7597731b4d2a5da8d616e2755ef202518` | Structures scalar multi-loops; live Plucky stage `0x3017450000` executes. |
| [#1565](https://github.com/mattias800/prosper/pull/1565) | `2d6ff9fb800d5e8689209958c17963d4f49b6a46` | Captures frames at exact guest-log phases; enabled deterministic Astro world-map capture. |
| [#1566](https://github.com/mattias800/prosper/pull/1566) | `81a9548e4cf79f9b653bbf78acb38e149f0c373e` | Composes multiple structured preludes with counted-loop lowering. |
| [#1572](https://github.com/mattias800/prosper/pull/1572) | `276b8f92` | Admits barrier-spanning VCC branches **proved** workgroup-uniform. Plucky stage `0x3017460000` reaches `unsupported=0`. |
| [#1574](https://github.com/mattias800/prosper/pull/1574) | `24a98629` | `gpu_replay --inspect-only` reports resolved `SPI_PS_INPUT_CNTL` linkage, so a synthesized-constant varying is visible offline. |
| [#1575](https://github.com/mattias800/prosper/pull/1575) | `37dfe752` | `shader_inspect` reports a missing resource table as `undetermined` (exit 3), not as a shader rejection. |

Current master is **162/162 ctest** when configured against `PPSA24651`, with all of the above composed.

PR #1566's focused translator/render set was:

```text
recompile_coverage
rdna2_spirv_struct
game_compute_exec
game_compute_exec_no_adaptive_storage_result
rdna2_to_spirv_exec
recompiled_fragment_render
recompiled_shaders_render
```

All seven passed locally, all six platform CI jobs passed, and release publication skipped. The exact retained Plucky
shader moved from four unsupported control-flow instructions to one deliberately rejected barrier-crossing branch.

## Lane A: Astro Bot world map

### Public state

- Issue: [#1459 — world map renders as a dark grayscale pattern](https://github.com/mattias800/prosper/issues/1459).
- Existing status background: `docs/ASTROBOT_LINUX_HANDOFF_2026_07_19.md`.
- Frozen outgoing branch: `investigate/astro-worldmap-closure`, clean at `2d6ff9f`; it contains no source changes and
  is behind current master. Start a fresh branch from `81a9548e` rather than extending it blindly.

Durable evidence comments:

- [submit 6279 bounded failure classification](https://github.com/mattias800/prosper/issues/1459#issuecomment-5141854392)
- [submit 6284 dependency map](https://github.com/mattias800/prosper/issues/1459#issuecomment-5141880607)
- [compute-writer closure source audit](https://github.com/mattias800/prosper/issues/1459#issuecomment-5141933098)
- [seven-submit relevance result and pivot](https://github.com/mattias800/prosper/issues/1459#issuecomment-5142003672)

### Retained evidence

Evidence root:

```text
~/.local/state/prosper/evidence/astro-worldmap-c5698376-froute.KyRLdP/
```

Important files:

```text
worldmap.prgbundle
closure-analysis-2d6ff9fb/submit-{6278..6284}.prgcap
closure-analysis-2d6ff9fb/submit-{6278..6284}.inspect-all.log
closure-analysis-2d6ff9fb/submit-{6278..6284}.graph.json
closure-analysis-2d6ff9fb/current-master-replay.log
closure-analysis-2d6ff9fb/current-master.bmp
closure-analysis-2d6ff9fb/submit-6284-op19-b5-captured.bin
closure-analysis-2d6ff9fb/submit-6284-op19-b6-captured.bin
closure-analysis-2d6ff9fb/submit-6284-compute8-stored.spv
closure-analysis-2d6ff9fb/submit-6284-compute15-stored.spv
closure-analysis-2d6ff9fb/submit-6284-compute15-current.spv
closure-analysis-2d6ff9fb/submit-6284-compute15-stored-v-current.diff
```

The b6 seed SHA-256 is `571bd231a552e2613817c87ca2d7fe911cbda679dd41fbf77c4f80fe0c35ab8b`.
The current full replay hash recorded by the outgoing agent is `28b5ea9a3fbd2ca9`.

The route used PR #1565's exact marker gate:

```text
LevelDocument Loaded: worldmap [worldmap]
```

It captured seven submits after one present, ending at submit 6284. The selected world-map output is still dark/near
black, so no progression screenshot was published.

### Proven facts

1. Submit 6279's apparent Vulkan failure is benign. Its only unrealized operation is an intentional no-effect draw:
   color write mask zero and no depth/stencil write. Both shaders have zero unsupported instructions.
2. Submit 6284's 171 unrealized operations are also benign: 170 no-effect operations and one zero-vertex draw.
   There are no ShaderRecompile, DescriptorContract, or Vulkan rejects.
3. Only one unresolved temporal leaf reaches final scanout:
   - op19 compute program `0x500758300`, raw hash `b88321fae48353e1`, reads b6 and writes b7 at
     `0x556760000`, a 1x1 Float32x4 tile-27 image backed by a 64 KiB footprint;
   - op221 compute program `0x500656100`, hash `5ad826b552f23191`, later reads the same allocation at b6;
   - final op236/draw206 fragment hash `5cb70f55e3902078` reads it at b49.
4. There is no earlier access or overlapping color target in submits 6278 through 6284. The graph leaf means “no
   producer in this bundle,” not “no captured backing.”
5. The full 65,536-byte b6 allocation is captured, materialized, detiled, and seeded. Op19's b7 writeback mutates the
   shared capture-owned resource instance, so later consumers observe it.
6. Extending runtime producer history to reflected compute writers would improve future closure diagnostics but
   cannot change this retained bundle's pixels. Do not implement that abandoned proposal as an Astro visual fix.
7. Op19's inputs at texel zero are exact Float32 values `(1, -1, 0.5, 0.180000007)`, with b2=1.0 and b3=0.1.
   Stored shader math yields b7 `(0.8, -1, 0.5, 0.180000007)`.
8. Current raw op19 recompilation is byte-identical to stored: 1,053 words, hash `b88321fae48353e1`.
9. Op221 only reads this 1x1 allocation and writes b15/b16 at `0x5460b000`, a 1920x1080 R11G11B10 image. It does
   not overwrite the 1x1 allocation. Both op221 and final op236 use only the control value's x component, `0.8`.
10. The 1x1 value is sane and finite. Black does not begin at op19. Current evidence has not yet distinguished
    op221's 1920x1080 result from later/final composition.

### Falsified: the R11G11B10 storage fork

Op221's stored (14,543 words, `5ad826b552f23191`) and current-raw (12,019 words, `4d55000d1c32f046`) SPIR-V are
not binary-identical, and the difference does concentrate in R11G11B10 storage lowering — stored packs float RGB
into an R32ui image, current writes a float vector with `StorageImageWriteWithoutFormat`.

**This is not a defect.** Forcing each module explicitly with `--override-compute-spv 15 PATH` proved the two arms
take genuinely different backend paths (`native-storage=0` "exact packed R11G11B10 storage via R32_UINT" versus
`native-storage=1` typed float) and still produce **byte-identical** output `28b5ea9a3fbd2ca9`. The divergence is
by design: `gpu_executor.cpp` forces `native_storage_format_support=0` whenever a capture is bound, keeping stored
modules device-independent. Do not spend a PR here.

Also falsified: seed-skip/poison corruption (`PROSPER_NO_SKIP_SEED=1` gives an identical hash); op221 itself
(it writes zeros, but from a genuinely empty upstream); and the b49 1x1 control (draw 206's only real image input
is `0x5420f0000`).

### Current Astro frontier: a frame-wide zero colour write mask

The world map is **not** a "dark grayscale pattern" — contrast-stretching the output shows a uniform diagonal
sawtooth with **zero scene content**. The prior "min 0 / max 13 / mean 5.97" reading is a content-free ramp.

**No geometry draw writes colour anywhere in the captured frame.** Across 7 submits / 43 realized draws, 28 draws
target the G-buffer `0x520440000` with `cwm=0` *and* `cwm1=0` while carrying real indexed geometry
(2,776–5,736 verts) and a **140,825-word** material fragment shader; ~170 further draws are suppressed as
no-effect for the same reason. Only 3 draws write full RGBA. Meanwhile the *captured* G-buffer holds 12,267,587
non-zero bytes of real content.

`render_state.cpp` resolves an **absent** `CB_TARGET_MASK`/`CB_SHADER_MASK` to **write-all**, so `cwm=0` means one
of those registers is **present with value zero**, or `MODE` is `DISABLE`. The live trace never once observed
`MODE=1 (CB_NORMAL)` — only mode 0 and mode 6 — which is not what an ordinary renderer looks like.

A tempting prior was the Gen5 stale-register-fold class. It was **investigated and not confirmed**, and the prior
itself turned out to rest on a superseded diagnosis:

- `SetRegsIndirect` reads a flat `{offset,value}` array and drops only `offset >= kRegOffsetLimit`, so the
  present-and-zero *capability* is real and generic.
- But #1364's "second record format / bit31-tag misparse" family diagnosis was **overturned**. The real mechanism
  was an HLE **success stub**: the SDK interpolant helper `dbOlWdppb4o` advertised 32 output records it never
  wrote, leaving **host stack residue** whose dword halves fold as present-and-zero registers. Fixed by #1368
  and #1411; current master emits zero `[dbbase-clobber]` events on 380 s+ routes.
- PRs #1344 and #1363 are downstream *symptom recoveries*, not fold-level invariants.
- Residue historically lands in the **low** offset cluster (DB `0x00-0x1F`, scissors `0x0C-0x0D`), whereas
  `CB_TARGET_MASK` is `0x8E`, `CB_SHADER_MASK` `0x8F`, and `CB_COLOR_CONTROL` `0x202`. That is counter-evidence.
- `PROSPER_DBBASETRACE` **cannot decide this**: it is hardcoded to DB offsets and only fires on a
  nonzero-then-zero-within-one-array signature, so it never fires for a register the guest never writes.

**Next Astro step.** Capture **v43** (PR #1576) retains the raw colour-state triple per draw with presence flags
and surfaces it in `--inspect`, which is what makes `cwm=0` attributable offline. The retained bundle is v42, so
it reports `color-state unavailable` — a **fresh v43 capture** is required. Take one, then determine whether the
zero masks are genuine guest intent or a prosper defect. If they are genuine intent, "real content exists but
nothing reaches the screen" via small mipped 4 KiB surfaces becomes the natural next suspect.

Do not recapture the world map for any other reason, revisit submit 6279, or implement compute-writer closure:
b6 is already captured, materialized, detiled and seeded, and no earlier producer exists in the bundle.

## Lane B: Dragon Quest VII overexposed composite

### Public state

- Issue: [#1486 — 4K composite is overexposed](https://github.com/mattias800/prosper/issues/1486).
- Status document: `docs/DRAGON_QUEST_STATUS.md`.
- The outgoing worktree was clean with no new repository changes. Its built replay tool and retained capsule were
  based on `2d6ff9f`; compare relevant tile/sampler source with current master before treating byte parity as final.

Durable comments:

- [corrected op89/op90 diagnosis](https://github.com/mattias800/prosper/issues/1486#issuecomment-5141895497)
- [exact op116 to op117 corruption boundary](https://github.com/mattias800/prosper/issues/1486#issuecomment-5141942105)
- [raw recompile result and b36 metadata](https://github.com/mattias800/prosper/issues/1486#issuecomment-5141986842)
- [controlled LUT substitution A/B](https://github.com/mattias800/prosper/issues/1486#issuecomment-5142022769)
- [full-volume offline decode and sampling pivot](https://github.com/mattias800/prosper/issues/1486#issuecomment-5142103444)

### Retained evidence

```text
~/agent-tmp/dq-cap3/black.prgcap
  SHA-256 c7a863d69823964a2f18ced544385caa9986c04c419da2f449524b330cf237cf
  capture v38, submit 1060, 124 operations

~/dq-submit1060-bisect.goFHHE/
~/dq-op117-lut-ab.xEjqqT/
~/dq-op117-lut-decode.WjwK2H/
```

The A/B manifest SHA-256 is `f2f44602c178226539ba456d9a9879c1ab5b04634f3cb590010434577771290d`.

### The "exact corruption boundary" is a BISECT ARTIFACT — retained as a warning

> **FALSIFIED 2026-07-31.** The op116 -> op117 "corruption boundary" below, on which this entire lane was
> founded, **does not exist**. `--through-operation N` renders **the last executed draw target**, so
> consecutive cutoffs display *different surfaces*. The retained logs show it plainly — every cutoff is a
> different resolution: op104 30x34, op111 480x270, op114 960x1080, **op116 1920x1080, op117 3840x2160**.
>
> **op116 and op117 were never the same buffer, so the transition between them cannot be a corruption event.**
>
> op116's identity is pinned independently: it is a *dispatch*, so its output selects op115's target
> `0x3083db0000` = draw90's `b35`, and its content (R .0124 / G .0443 / B .1865) matches `tap-142` —
> draw90's own sample of b35 — to three decimals. **op116 is a post-process input that draw90 consumes.
> It is dark because such buffers are dark**, not because it is the last good frame before corruption.
>
> Do not re-derive a boundary from prefix replay without first confirming both prefixes render the **same
> target at the same resolution**. This is the single most expensive mistake recorded in this document: it
> shaped every hypothesis in the lane for two sessions.

The original (now-falsified) table, retained only so the claim is recognisable if it resurfaces:

| Prefix | Output | Hash / metrics |
|---|---|---|
| op116 | coherent dark ocean, 1920x1080 | replay `bee945066f3fc113`; BMP SHA `1273a16acea96c465db75021903cd0dadf2f16b1b3fa37e84296a0668866806e` |
| op117 | first overexposed/cyan frame, 3840x2160 | replay `31d794e48c2860dc`; BMP SHA `4dcf4f64ea9ed8e6901a92735b76cfa92bf41845c1a62d0f6818175a77728198` |
| op118 | orange overexposure | luma about 0.819; about 80.3% bright pixels |
| op122 | final opaque black overlay | not the first corruption |

Op117 is source draw90/order1436911. It is a realized fullscreen indexed triangle targeting `0x30867e0000`,
3840x2160 format44, write mask `f`, with blend/depth/stencil disabled.

Inputs:

- b34/b37/b38 from op88 at `0x308cfc0000`, 4K;
- b35 from op115 at `0x3083db0000`, 1920x1080;
- b36 grading LUT at `0x3021fe0000`;
- VS b7 from op103.

Stored VS/FS hashes are `6cde109bb55150ba` and `2810ab556bbfbc4d`.

`--recompile-raw --through-operation 117` is pixel-identical to stored replay. A no-render inspection changed the
installed VS hash to `ed25159f6e9d0b87`, and the mass path swaps VS and FS atomically, proving the draw used regenerated
stages. Stale stored shader translation is not the cause.

### Binding 36 identity

| Property | Value |
|---|---|
| Address / extent | `0x3021fe0000`, 32x32x32 |
| Format | enum21 `Unorm2_10_10_10`, four components |
| Declared / captured | 131,072 bytes / 2,097,152-byte footprint |
| Captured content | 107,001 nonzero bytes; hash `b53c7e7c2e9910e3` |
| Tile mode | 27, `SW_64KB_R_X` |
| Address modes / swizzle | 2/2/2; 4/5/6/7 |
| Filters | 1/1/1 |
| DCC metadata | enabled; address `0x306e9b0000`; 131,072 bytes of `0xffffffff`; hash `3735b73346670383` |

All-`0xff` metadata selects the renderer's ordinary uncompressed-base path. Current evidence does not indicate that
compressed blocks need decoding.

### Controlled LUT A/B

All variants reached op117 and used the same global diagnostic-copy path. Binding999 was the selector-miss control.

| Case | Output | Meaning |
|---|---|---|
| Baseline | hash `31d794e48c2860dc`; RGB mean 0.746193; luma 0.742344; 51.96% pixels at luma >=0.75 | Overexposed reference. |
| Selector miss, draw90/b999 | byte-identical to baseline | Diagnostic-copy side effects are excluded. |
| Checker, draw90/b36 | hash `7e9bec1982cda52e`; RGB mean 0.436836; luma 0.498721; no pixels >=0.75 | Every pixel changes; scene structure remains. |
| Zero, draw90/b36 | hash `ccc433ff6d980383`; one-color all-black frame | b36 materially drives the complete composite. |

The binding is active, its 3D view reaches the shader, and the selector is correct. Do not repeat unused-binding,
wrong-draw, or generic diagnostic-copy hypotheses.

### Full-volume decode result

The existing pre-detile resource dump exported the 2 MiB tiled allocation without a GPU replay. A temporary offline
analyzer called the project's own `detile_volume(..., 32, 32, 32, tile27, 4)` and the renderer's packed conversion.

Hashes:

```text
tiled allocation  60c100ff926e6ae226802e12b1510c54047b1472bbca120f6a47618fe9a0cf1b
linear packed     e9de4555ad4b65ea21de498c6d00443741773efa19269bfd98406ef64e61bfe1
linear RGBA8      eb27c3589d67553f86bf6f833894cee5f4ab3970c3b0bf2e0a51873ed0c89dca
metrics           b606c230b3ce953c9158015eacd5fe8eaf15eb9e7dbd223358bceb6d99abc257
z-slice sheet     e928a699a160757b6605c11a289633851440cb84279c989feeb2fd91371e31cb
```

Decoded channel metrics:

| Channel | Min | Max | Mean | Unique values |
|---|---:|---:|---:|---:|
| R | 0 | 255 | 176.9178 | 255 |
| G | 0 | 255 | 178.4504 | 254 |
| B | 0 | 255 | 153.9832 | 250 |
| A | 0 | 0 | 0 | 1 |

Mean adjacent differences are small and smooth:

| Axis | R | G | B |
|---|---:|---:|---:|
| X | 5.0448 | 2.8005 | 3.1537 |
| Y | 3.9890 | 5.1508 | 3.9264 |
| Z | 3.4532 | 3.5657 | 6.3019 |

The 32-slice contact sheet is visually coherent, with smooth RGB transfer surfaces and no checkerboard, tile-sized
scrambling, alternating bad slices, or discontinuity at 8/16-voxel boundaries. Tile27 detiling and packed
10:10:10:2 unpacking are now unlikely primary causes. Before citing exact runtime parity, confirm that current
master's tile conversion source matches the tested `2d6ff9f` version.

### Binding 36 is exonerated — do not re-open it

A point-versus-linear A/B produced `linear` byte-identical to baseline **because the T# already reports
`filt=1/1/1`**, with `point` (`b005d9f5045a2f6d`) differing on 8,243,869 of 8,294,400 pixels. Both arms provably
took distinct paths, so this is a real negative, not a broken experiment.

`PROSPER_FS_TAP` measured draw90's actual sample coordinates as **(0.385, 0.536, 0.688)** — none saturated — and
the LUT correctly returns (0.424, 0.768, 0.975) there. draw90's FS is a faithful UE4 tonemapper with
`unsupported=0`. Ruled out and not to be re-run: LUT contents, tile27 volume detile, DCC, 10:10:10:2 unpack,
binding/view/3D-coordinate wiring, dimensional opcode, dmask, swizzle, address modes, sampler filtering, and
draw90's shader translation.

The LUT's corners look absurd as an RGB cube (pure R gives 255,224,190) but are exactly right for UE4's
**log-encoded `CombineLUTs`** volume.

**Caution:** the older "`--recompile-raw` at op117 is byte-identical" result **predates a `rdna2_to_spirv.cpp`
change** and must be re-measured before being cited again. Verify source parity by hashing the specific
translation units rather than assuming.

### Also cleared: the auto-exposure producer and delivery

- **Producer format** — `b15` is `fmt=4 nc=4` Float16x4, `declared=16320 = 60x34x8`, seed tagged `rgba16f`. Exact
  agreement, so this is **not** a #773-class native-format loss.
- **Producer content** — draw77 writes R .0973 / G .3472 / B .9731, luma `.33929`, with `rgb_nonblack=2040`
  (all texels) and `HIT` on every sample.
- **Stale seed** — falsified; the near-black buffer is draw79 (operation 105, *after* op103), which is capture
  ordering rather than a bug.
- **Delivery** — `1=param1`, a real `PARAM1` export consumed at `pc=0177`. No draw90 input is a
  `ConstantDefault` (`valid=ffffffff`, only `param0..3`), and every PS attribute read falls inside its VS
  export's EN mask. Confirmed with the #1574 seam.

The eye-adaptation buffer reporting average luminance **629** is **not** the cause, and the arithmetic shows why:
too-high average luminance implies too-*small* exposure implies a too-**dark** frame, but the frame is too
**bright**. Against the oracle the error is ~1–2 stops (a factor of **2–4**), nowhere near the ~8192x a broken
exposure implies.

Two retractions worth preserving so they are not re-derived: a "multiplier is exactly 1.0" claim derived by
inverting *means* through a `log` is invalid under Jensen's inequality — it was re-established by direct
measurement instead (`tap-177` uniform at `0.984314`, one distinct value across all 8.29M pixels). And a
"~14x amplification" in the post chain was an artifact of comparing **different registers** via `PROSPER_FS_TAP`'s
`(dst..dst+3)` export window.

### Not bloom: film grain and chromatic aberration

pc=214–256 decodes to literals summing to `0.99999999` (Rec.601 luma), then `1/(2π)` → `v_sin_f32` (which prosper
lowers as `sin(x·2π)`, cancelling it) → `493013.0` → `v_fract_f32`: `fract(sin(x)·493013)` hash noise applied as
`c·(1 + noise·s)`. Zero-centred and multiplicative, so it averages to ~1.0 and **cannot** produce systematic gain.
pc=154/157/164 use the *same* T# and sampler with different coordinate VGPRs and dmasks 1/2/4 — one texture
sampled three times at offset UVs, one channel each: **chromatic aberration**. b34/b37/b38 aliasing one surface is
three descriptor slots on one texture, not three reads.

### Next Dragon Quest assignment

The lane reduces to one number: the frame is bright because b34 is bright and the effective exposure is ≈1.0,
since `s97 x ExposureScale` is a measured uniform `0.984314`. `tap-171` renders exactly 0 across all 8.29M pixels,
bounding delivered `ExposureScale < 1/510`, so **`s97 > 502` is already established** and `s97 = 1.0` is excluded.

Measure `s97` at `b33+0x878` and `s106` at `b32+0x50` directly with two bounded `--dump-resource 90:ps:32` and
`90:ps:33` runs rather than inferring them. Two outcomes:

- **`s97 ≈ 8192`** — a UE4 **PreExposure** reciprocal, meaning the ≈1.0 cancellation is by design and the defect
  is upstream: prosper's base pass never applied the matching PreExposure factor. That would be a shared UE4-path
  change touching Plucky, ArcRunner and The Pathless, so it needs a generic contract and regression first.
- **`s97` is some other value > 502** — op103 wrote a different value in replay than the capture recorded,
  relocating the defect to op103's execution with no cross-title blast radius.

Note the standing counter-evidence against the PreExposure reading: if the base pass had applied
PreExposure = 8192, b34 would hold values in the thousands, but b34 is a smooth gradient at (0.098, 0.348, 0.971).
And the ~2–4x magnitude needed to match the oracle is far too small for a PreExposure-scale defect.

## Lane C: The Plucky Squire skipped compute

### Public state

- Issue: [#1554 — three compute programs skipped after chapter-one setup](https://github.com/mattias800/prosper/issues/1554).
- Current clean branch/worktree may be transferred: `investigate/plucky-pc915-proof` at exact current master
  `81a9548e`. It has no edits or commits.
- Live proof: [issue comment](https://github.com/mattias800/prosper/issues/1554#issuecomment-5141779159).

Retained evidence:

```text
~/plucky-work/post1564-live-retry.fwpVFI/
  guest.log
    SHA-256 b6c9ba2bf32762796cdf342d900c109105f3f80f31934207728e3e7fc7f71c0d
  shader-dumps/exec_cs_3017460000.bin
    SHA-256 993af8855a62ac24203b12bcf2a968efb1ba163b64937f325353217d411d7342
```

### Proven progression

PR #1564 made the preceding target `0x3017450000` execute successfully 48 times with zero skips. The immediate next
unsupported stage is `0x3017460000`.

PR #1566 then composed its two disjoint pre-loop choices with the canonical counted-loop lowering. The exact retained
stage is:

- 8,448 bytes / 2,112 dwords;
- 1,964 decoded dwords / 1,327 instructions / valid END_PGM;
- coverage `total=1326 alu=1260 exp=0 table=65 unsupported=1`;
- 65 table-dependent MUBUF operations: 33 loads and 32 stores;
- one remaining structural rejection: `s_cbranch_vccz` pc915 to pc1963/END_PGM, whose fallthrough arm contains
  `S_BARRIER` pc1118.

Raw stage recompilation without the live resource table first stops at table-dependent MUBUF pc31. That is separate
from the control-flow coverage result.

### Pc915 predicate proof so far

The scalar chain is known exactly:

```text
pc900: load four scalar dwords into s8:s11 from s12:s13 + 0x20
pc903: load s16:s17 from descriptor s8:s11 + 0x0c
pc906: vcc_lo = s17 - 1                         # VCC_LO used as scalar data here
pc907: SCC = unsigned(s16 > 0)
pc908: VOPC writes its wave mask to s8:s9       # does not feed pc914 VCC
pc910: s10:s11 = SCC ? EXEC : 0
pc911: SCC = unsigned(s14 == vcc_lo)
pc913: VCC = SCC ? EXEC : 0
pc914: VCC &= s10:s11
pc915: branch if VCC is zero
```

Therefore, immediately before pc915:

```text
VCC = EXEC iff (s16 > 0) && (s14 == s17 - 1); otherwise VCC = 0
```

This proves uniformity within one guest wave. It does **not** prove that every wave in the workgroup makes the same
decision. `s14` has no decoded writer before pc915 and is an entry/system/user SGPR. `s12:s13` are also entry values
and source the descriptor/data chain. The retained raw file and log do not include enough launch/register metadata to
map those values.

Do not infer safety merely from the fact that the guest shader exists. If these scalar entry values differ by wave,
placing a Vulkan workgroup barrier inside an ordinary structured arm can violate uniform barrier participation.

### Resolved and merged (#1572)

The pc915 branch is **proven workgroup-uniform** and admitted by a generic rule. Two obligations are proved
independently and mechanically:

1. **EXEC is full at the branch on every path**, by forward must-dataflow (greatest fixpoint, sound across
   back-edges). This leg was missing from the original framing and is load-bearing: `VCCZ` is *also* true for a
   wave whose EXEC is empty, a genuinely per-wave property no amount of scalar uniformity recovers.
2. **VCC is `SCC ? EXEC : 0`**, optionally combined by `s_and_b64`/`s_or_b64`, with every contributing SCC from a
   scalar compare whose every scalar input traces to launch data.

Unproved provenance stays rejected: EXEC/SCC as data, `V_READFIRSTLANE`/`V_READLANE`, VOPC masks, SGPRs above the
launch range, and definitions entering from outside the straight-line region.

The uniformity guarantee is a property of **prosper's own compute entry seeding**, not an assumption about the
guest: user SGPRs come from `load_push_constant` (dispatch-uniform), system SGPRs from `b.groupid[]`
(`gl_WorkGroupID`, workgroup-uniform), TG_SIZE is a constant, and local invocation IDs are seeded into **VGPRs**,
which the provenance walk rejects outright. Nothing at compute entry places a wave-varying value into an SGPR.

Supporting CFG facts for the retained stage: in `[916,1963)` there are 12 barriers and 35 branches, all forward
`s_cbranch_execz` with no back-edges, none spanning a barrier; across the whole program only three edges cross any
barrier (`pc83→899`, `pc898→82`, `pc915→1963`), making pc915 the sole barrier-crossing edge. Coverage moved from
`unsupported=1` to `total=1326 alu=1261 table=65 unsupported=0`.

### Live confirmation, and a retraction that matters

**`0x3017460000` executes live**: 196 dispatches, every one `result=ok`, zero skips, deterministic at a single
SPIR-V hash, zero Vulkan errors across 51,808 log lines.

The proof was **genuinely exercised, not trivially satisfied**: `local=128x1x1` with `subgroup=0` (the portable
path) means **more than one guest wave per workgroup**. In a single-wave workgroup, wave-uniform would imply
workgroup-uniform for free and the barrier promotion would never have been under load. Twelve barriers inside a
VCC-entered region, multiple waves that must agree, 196 times.

All three programs named in #1554 now execute: `0x3017d90000` (#1561), `0x3017450000` (#1564),
`0x3017460000` (#1572).

**RETRACTED — the "three remaining blocked programs" finding was wrong.** A follow-up investigation reported that
`0x30133e0000`, `0x3013430000` and `0x30194c0000` were blocked by descriptor-resolution failures. Unfiltered
logging showed they execute **3,501 times** with a **single** transient failure (1 in 763, self-recovering, 108
successes before and 655 after). See instruments #9 and #10 in the list above: deduped skip lines plus a targeted
`PROSPER_COMPUTELOG_CODE` filter hid thousands of successes. Tracked as **#1581**, retitled to match.

What survives unchanged: `image_gather4_lz` (0x47), `image_store` (0x08) and MUBUF `<=0x07` are **all already
implemented**, and offline coverage is `unsupported=0` for all three. This was never an instruction-coverage gap.

Also: **#282's own comment retracts its premise** and its design doc is gone from master, so neither #282 nor
#485 covers this area — do not inherit them as background.

### Next Plucky assignment

**Rung 4 needs a human.** The one thing no agent here can produce is a visual milestone: a screenshot attempt
returned the KDE lock screen, and was correctly quarantined as non-evidence rather than published. An unlocked,
human-present session is required.

The remaining lead is narrow and **not diagnosed**: for the single failing dispatch the realized resource table
is **byte-identical** to succeeding ones, yet `[compute-cfg]` reports `branches=2` on failure and `branches=1` on
success with the same `hash4k`. Recorded on #1581 as a lead. Severity is low — one dropped dispatch in 763,
self-recovering, visible impact unmeasured — so **do not spend a lane on it** until something visible is
attributed to it.

## Lane D: Alex Kidd in Miracle World DX (PPSA02664)

Added 2026-07-31 because the inherited allocation was skewed to the hardest titles. **It reached the gameplay
rung in one session**, and the root cause was a **generic tiling defect**, not a title quirk.

`tiled_mip_level_layout` converted `mip_x`/`mip_y` — which count whole **256-byte** blocks — to elements using a
multiplier derived from the *macroblock* (`block_width >> 4`). That is correct only when the macroblock is 16x the
256-byte block (the 64 KiB modes) and **four times too small for every 4 KiB mode**. The struct contradicted
itself, reporting `byte_offset=0x800` alongside `tail_x=4` (byte `0x80`) for the same level, and the consumer used
`tail_x`. **53 of 130 packed-tail levels violated the origin-agreement invariant; zero after the fix.**

The correct multiplier is the 256-byte block's element extent, which depends only on element size (AddrLib
Block256_2d): 1B=16x16, 2B=16x8, 4B=8x8, 8B=8x4, 16B=4x4 — every entry multiplying to exactly 256 bytes.

The visible consequence shows how far a texture-origin bug can travel: mipped 4x4 SpriteMask sprites decoded as
foreign, fully transparent texels, so every mask fragment failed its alpha test and a **legitimate** "visible
outside mask" fill covered the entire frame. A black screen, several layers from its cause.

Falsified along the way: both prior size hypotheses (the guest's T# really is 4x4 with `base_level=0`), descriptor
selection, Gen5 size decode, the 4x4 fallback, blend/stencil semantics, guest-state divergence, and stale texture
content. The much older "`fs2949` premultiplies by vertex-colour alpha 0" diagnosis remains falsified — that
capsule was the intro narration crawl, where alpha-0 is correct.

**Two things to carry forward.** Existing `.prgcap` files **bake the resolved tail coordinates**, so pre-fix
capsules replay unchanged even on a fixed binary; re-verifying anything tiling-related needs a **fresh** capture.
And scene identity must be established from rendered *semantic content*, never from submit ordinals, guest VAs,
draw counts, or shader-hash signatures — that invariant was violated once here already.

**The title now reaches rung 6.** The `alexkidd-gameplay` guard is reviewed and landed, so PPSA02664 is
"done" by the ladder's own definition and the lane can close.

The adopted contract keeps the proposed `min_colors=20000` but moves the window to **95–145 s**. Measuring the
route first (`tools/snapshot/profile_route.py`, added with the guard) showed the level loads at 75 s at scale 4
and 72 s at native, so the proposed 70–120 s window would have straddled the load transition: five dark frames
sat inside it, and a run loading only ten seconds later would have pushed the guard below its content-match
ratio and failed on healthy output. The window now opens twenty seconds after the observed load **and after the
route's last pad press at 89 s**, so it samples a settled scene whose only motion is cloud drift and enemy
animation. That is what buys the margin: the worst of 100 reviewed frames scores SSIM **0.93** against a
0.85 floor, where the phase-sensitive `blue-prince-hall`/`terminator-boot` guards fail on healthy frames.
Non-black is `min_nonblack_ratio=0.5` (the tool's conservative half of the reviewed 1.0), not the proposed
0.95 — the approval flow derives that floor and it still sits 3.6x above the pre-fix 0.137. Colour count is one
of five conjoined conditions and must never become the contract alone.

Three independent runs: verify CONTENT-STABLE (richest 35,308/34,984, 50/50 qualifying in both), then
`check` OK (50/50 qualifying, 50/50 structural, 50/50 non-black). Pre-fix was 3,010 colours at 13.7%.

**#710 (title/menu renders incorrectly) is substantially resolved but not closed.** Against the hardware oracle
the panel base is now (253,183,0) vs (250,184,1) — it was (91,66,14) — and the logo, previously "faded /
semi-transparent", matches the oracle's regional mean within 3/255. What remains is narrow and measured: the
panel's watermark overlay lifts the base by only about **half** the hardware amount (G +14 vs +26, B +53 vs
+104). That ratio is a consistent 0.52 in display space but 0.51/0.26 in linear, which argues **against** a
linear-vs-sRGB blend-space explanation and leaves the earlier "semi-transparent tiles composite at too-low
alpha" factor as the live hypothesis. Re-verify any tiling-related claim with a **fresh** capture: pre-fix
`.prgcap` files bake the resolved tail coordinates.

## Lane E: The Pathless (PPSA01826)

First triage 2026-07-31. **Not greenfield** — #1213 exists with prior work, was closed apparently by accident,
and has been reopened. Current state is tracked in **#1570**.

UE4 4.27-class with Wwise. Boots remarkably far — Wwise up, full UE4 thread topology including a Navmesh Tile
Builder (so a real level is loaded), 6,150 submits / 28,859 draws / 27,350 dispatches in 75 s — but **every
presented frame is a flat colour** (blue then white) and progression freezes at ~18 s. That is **rung 0**.

The #1213 worker SIGSEGV at `eboot+0x1b70cd0` **did not reproduce** across four runs. An offline capture (485
draws, 1 dispatch, **0 failures**, replaying hash-exact at `a5e7b61cbf984383`) shows 482 draws plus 1 compute
writing a 768x768 Slate/UMG surface, and only **3** draws touching the 3840x2160 front buffer — all three binding
a 1x1 white texture. prosper faithfully realizes what the guest submits; the defect is composition/progression.

Later stepping showed the front buffer goes black → white, and the three front-buffer draws are the ones painting
white. Their pixel shader's entire constant buffer is a single RGBA of `(1.0, 0.4545, 0.0, 1.0)` — **orange** — and
the export is a COMPR packed-FP16 MRT0 export fed by `V_PACK_B32_F16`, consistent with `export=4` (FP16_ABGR). So
the shape is "a tint constant that should produce orange yields white", pointing at the **packed-FP16 export
path** rather than texel decode. The 1x1 white texture looks like a deliberate Slate white-brush dummy.

Open question the capture cannot yet answer: whether the composite lives in a different submit or the game is
genuinely blank — the capture is 1 submit of ~6,150. A `.prgtl` timeline survey of all submits is the next step
and is CPU-only.

Generic gaps found, all `area:infra` and all benefiting other UE titles: DCC `tile=27` sampled images and metadata
extents unsupported; a 400x400x212 image exceeding the 512 MiB backend bound; and 40+
`[agc] WaitRegMem q=F … dependency violated` — the same per-queue barrier gap as ArcRunner #1226. Estimated cost:
**rung 1 ≈ one medium focused investigation** (answerable offline from the existing capture), **rung 3 ≈ several
sessions**.

## Suggested allocation for a new orchestrator

| Agent | First bounded task | GPU |
|---|---|---|
| Astro Bot | Take a **fresh v43** capture, then attribute the frame-wide `cwm=0` from the raw colour-state triple | One bounded capture |
| Dragon Quest VII | Measure `s97`/`s106` with two `--dump-resource` runs; decide PreExposure versus op103 execution | Two ~2 s runs |
| The Plucky Squire | Live route: confirm `0x3017460000` executes; identify the next skipped stage | ~10 min routed run |
| Alex Kidd | Land the snapshot guard; re-check #710 against the tiling fix | Snapshot verify |
| The Pathless | `.prgtl` survey of all ~6,150 submits for the real composite; then the packed-FP16 export path | CPU first |
| Orchestrator | Review diffs, sequence GPU, publish/merge, keep this document current | Coordinates leases |

**Balance breadth against depth.** Astro Bot is a AAA showcase title and has produced no shippable title fix
across two sessions; Dragon Quest and Plucky are deep but tractable. Alex Kidd went from stuck to gameplay in one
session, and The Pathless was never attempted. When a hard lane stalls, reassign rather than persist — overall
library progress matters more than any single title, and the user has said so explicitly.

**Expect the generic wins to come from tooling and shared subsystems.** This session's merges were a recompiler
uniformity rule, two diagnostic seams, a tooling-honesty fix, and one shared tiling fix. Every one of them helps
titles beyond the lane that found it.

## Ready-to-send subagent prompts

Give every agent: its worktree path, its branch, the exact `origin/master` SHA, the dump root, the **distrobox**
build command, and the instruction to keep evidence under `~/` and out of `/tmp`, the repo, and public text.

### Astro Bot

```text
Read CLAUDE.md, docs/GAME_COMPAT_ORCHESTRATION.md (Lane A), and issue #1459.
Do NOT re-run the R11G11B10 storage fork (falsified: byte-identical output across different
backend paths), compute-writer closure, submit 6279, or the b49 1x1 control.
The frontier is that no geometry draw writes colour anywhere in the frame: 28 realized draws
resolve cwm=0 and cwm1=0. An ABSENT target/shader mask resolves to write-all, so cwm=0 means
present-and-zero, or MODE=DISABLE. Capture v43 retains the raw triple; the retained bundle is
v42 and reports "color-state unavailable", so take a FRESH v43 capture first. Then determine
whether the zero masks are guest intent or a prosper defect. The Gen5 stale-fold prior was
investigated and NOT confirmed; read the lane notes before reviving it.
```

### Dragon Quest VII

```text
Read CLAUDE.md, docs/GAME_COMPAT_ORCHESTRATION.md (Lane B), docs/DRAGON_QUEST_STATUS.md, and #1486.
Binding 36 is EXONERATED and the op103 auto-exposure producer and interpolant delivery are CLEARED.
Film grain and chromatic aberration are both excluded. Do not re-open any of them.
Measure s97 at b33+0x878 and s106 at b32+0x50 with two bounded --dump-resource 90:ps:32 and
90:ps:33 runs. s97 > 502 is already established. If s97 is approximately 8192 it is a UE4
PreExposure reciprocal and the defect is upstream in the base pass (shared UE4 blast radius —
name a generic contract first); otherwise op103 wrote a different value in replay than captured.
Note the counter-evidence: b34 is a smooth gradient, not thousands-valued, and the oracle gap is
only 2-4x. Re-state tile.cpp parity against current master rather than citing an old blob hash.
```

### The Plucky Squire

```text
Read CLAUDE.md, docs/GAME_COMPAT_ORCHESTRATION.md (Lane C), #1554, and PRs #1564/#1566/#1572.
The pc915 uniformity work is MERGED; do not redo it. It is not yet a live-execution claim.
Run the ~10-minute native-cadence route and confirm 0x3017460000 executes, then identify the
next skipped stage. Absence of "[compute] skip unsupported program" proves zero skips. Use the
targeted PROSPER_COMPUTELOG_CODE=<addr>, not blanket PROSPER_COMPUTELOG. Prefer the live path's
own rejection reasons over any offline verdict. Keep PROSPER_RENDER_SCALE=1 and RENDER_EVERY=1.
```

### Alex Kidd

```text
Read CLAUDE.md, docs/GAME_COMPAT_ORCHESTRATION.md (Lane D), and #320.
The generic packed-mip-tail tiling defect is FIXED; the title reaches gameplay. Remaining work is
the snapshot guard: route scripts/ppsa02664/reach-first-gameplay.pad, sample 70-120 s, require the
richest frame to have non-black >= 95% AND >= 20,000 distinct colours, paired with the existing
SSIM comparison. Never use colour count alone — a seed-miss gradient outscores real content.
Run snapshot.py verify and inspect every retained image before adopting a baseline. Then re-check
#710, plausibly the same root cause. Pre-fix .prgcap files bake resolved tail coordinates, so any
tiling re-verification needs a fresh capture.
```

### The Pathless

```text
Read CLAUDE.md, docs/GAME_COMPAT_ORCHESTRATION.md (Lane E), #1570, and #1213 (reopened).
The title is at rung 0: it boots deep into the UE4 frame loop with real GPU work but presents flat
colour and freezes at ~18 s. The #1213 SIGSEGV does not reproduce. Survey the .prgtl timeline
across all ~6,150 submits to find whether the real composite lives in another submit — this is
CPU-only. The current lead is the packed-FP16 MRT0 export path: a tint constant of
(1.0, 0.4545, 0.0, 1.0) (orange) is presenting as white through a COMPR V_PACK_B32_F16 export.
Generic gaps worth fixing for all UE titles: DCC tile=27 sampled images, the per-queue WaitRegMem
barrier model (shared with ArcRunner #1226), and the 512 MiB backend image bound.
```

## Orchestrator cadence and reporting

A useful cadence is:

1. Ask each agent for a compact status after one bounded experiment or one source-design pass.
2. Immediately redirect agents that falsify their premise; do not let them implement a now-diagnostic-only idea.
3. Keep one integration step active at a time: inspect diff, verify exact head, publish PR, watch CI, merge.
4. Let the other agents continue evidence gathering while CI runs, provided they are not modifying the same files or
   depending on the unmerged result.
5. After every merge, fetch master and restart dependent branches from the new exact head.
6. Report to the user in plain language: what became visible/correct, what was ruled out, what is running, and whether
   the GPU is shared or exclusive. Performance numbers are ballpark unless the GPU was isolated.

When an agent appears frozen, distinguish a real long computation from a helper blocked on stdin. Avoid `gh ...
--body-file -` in unattended commands; use a real bounded argument/file so the process cannot wait forever for input.

## Updating this document

After a material event:

- replace the exact master SHA;
- move merged work into the foundation table;
- update the lane's “proven,” “falsified,” and “next” sections;
- link the new issue comment/PR;
- record retained artifact hashes, but sanitize paths;
- remove obsolete experiments so a new agent does not repeat them;
- keep unresolved risks explicit;
- preserve enough history to explain why an attractive old hypothesis was rejected.

This document is successful when a new orchestrator can allocate agents, reuse the retained evidence, and run the next
three discriminating experiments without asking the previous session what it meant.
