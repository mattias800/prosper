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
- Exact master at handoff: `81a9548e4cf79f9b653bbf78acb38e149f0c373e`.
- Active title lanes: Astro Bot, Dragon Quest VII, and The Plucky Squire.
- Repository state: every outgoing game agent froze with a clean worktree and no uncommitted source changes.
- Process state: no Prosper, `gpu_replay`, compiler, or game process was intentionally left running.
- GPU state: unclaimed. GPU sharing is the default; see the scheduling rules below.
- Highest-value next experiment: Dragon Quest VII binding-36 point-versus-linear sampling A/B on the retained
  submit-1060 capsule.
- Highest-value implementation uncertainty: Astro Bot operation 221's R11G11B10 storage output under stored versus
  current raw shader lowering.
- Highest-risk proof frontier: Plucky Squire's barrier-spanning pc915 VCCZ branch; no implementation is justified
  until workgroup-uniform entry-SGPR provenance is established.

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

### Highest-value unresolved fork

Op221 stored and current raw SPIR-V are not binary-identical:

| Variant | Words | FNV/hash | SHA-256 |
|---|---:|---|---|
| Stored | 14,543 | `5ad826b552f23191` | `060d685ecae0a3832861bacacd113e82ca40155586cc8820c25863f227b2ee8e` |
| Current raw | 12,019 | `4d55000d1c32f046` | `d2c54a3292b7b59513c904f158d4cee81b7462292a61370cd9422918b783ce6a` |

The difference concentrates in R11G11B10 storage lowering. Stored SPIR-V manually packs float RGB into an R32ui
image; current SPIR-V writes a float vector to a typed/unknown-format image with
`StorageImageWriteWithoutFormat`. Static intent may be equivalent, but output parity is unproved.

Final draw206/op236 also needs a targeted raw-recompile proof. A mass raw run retained the stored hashes, but only
7 of 36 draw pairs were substituted, so unchanged hashes alone do not prove this pair rebuilt.

### Next Astro assignment

Start with two short retained-bundle GPU A/Bs; neither requires exclusive GPU ownership:

1. Compare default stored op221 against `--recompile-raw` on the same bundle and bounded output. Confirm op221 was
   actually substituted and compare image/output hashes. Existing `--override-compute-spv 15 PATH` can force the
   stored or current module explicitly.
2. Trace/substitute final b49 with `PROSPER_RESOURCE_HASH_DIM=1x1` and a selector-miss versus binding-49 hit using
   `PROSPER_TESTTEX_DRAW=206`, `PROSPER_TESTTEX_BINDING=49`, and either checker or zero. Run draw206 with its compute
   prefix or the full submit.

Interpretation:

- Stored-versus-current op221 divergence localizes the first implementation target to R11G11B10 storage lowering.
- Identical op221 results plus a material b49 substitution effect moves the investigation to later consumers or final
  fragment composition.
- If current tools still cannot attribute the boundary, add a reusable post-operation resource dump: after
  `--through-operation 221`, export capture-owned `host_data` for `0x5460b000` before later consumers. The current
  pre-submit resource dump cannot do this.

Relevant code:

```text
prosper/src/gpu/gpu_capture.cpp
prosper/src/gpu/gpu_dependency_graph.cpp
prosper/frontends/shared/live_compute.cpp
prosper/frontends/shared/live_renderer.cpp
prosper/tools/gpu_replay/gpu_replay.cpp
```

Do not recapture the world map, revisit submit 6279, or implement compute-writer closure until contradictory evidence
appears.

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

### Exact corruption boundary

Adaptive operation-prefix replay found:

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

### Next Dragon Quest assignment

The leading area is b36 sampling semantics in the translated fragment shader or backend sampler state.

1. First post the full-volume result to issue #1486; it was produced after the last existing public comment.
2. Run the binding-36-scoped `PROSPER_TESTTEX_FILTER=point` versus `linear` A/B through op117 while retaining the
   original pixels. Include baseline, hashes, image metrics, and confirmation that both reached op117. Two bounded
   correctness replays do not require exclusive GPU use.
3. If filtering is not explanatory, inspect draw90's exact b36 sample path:
   - normalized versus integer coordinates;
   - 3D dimensional opcode and coordinate component order;
   - descriptor swizzle;
   - address modes;
   - sample versus load behavior;
   - min/mag/mip filter construction.
4. If current seams cannot separate these, prefer a reusable draw/stage/binding-scoped decoded-resource dump that
   records hashes for pre-DCC/base, post-DCC, post-detile packed, and post-unpack upload data.
5. A draw/binding-scoped identity 3D LUT (`RGB = normalized XYZ`) after unpack is a useful synthetic control if
   existing checker patterns cannot isolate coordinate semantics. It is diagnostic, not a title fix.

Relevant current controls:

```text
--dump-resource DRAW:ps:BINDING PATH
PROSPER_DUMP_RAWTILE
PROSPER_DUMP_RAWTEX
PROSPER_DUMP_RESOURCE_VERSION
PROSPER_NODETILE                 # global; avoid as a clean A/B
PROSPER_TESTTEX_FILTER=point|linear
```

There is no current binding-scoped DCC bypass, complete post-unpack 3D-volume export, or `--override-resource`.

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

### Next Plucky assignment

This lane is static/offline first and needs no GPU:

1. Enumerate every SOPP branch/backedge and every `S_BARRIER` in `[916,1963)`. Determine whether pc915 is the only
   edge crossing each barrier and classify every nested branch.
2. Inspect retained capture/log state for compute program metadata, `COMPUTE_PGM_RSRC2`, user/system SGPR enables,
   local dimensions, and dispatch mapping. Map s12:s14 before considering code.
3. Trace compute launch construction in the executor/resource table. Mechanically prove or disprove that s12:s13,
   s14, and the scalar memory values s16:s17 are identical for every wave in one workgroup.
4. Only after that proof, design a generic rule that promotes a barrier-spanning wave branch to workgroup-uniform when
   every predicate source has dispatch- or workgroup-invariant provenance. Unknown entry SGPRs, wave/lane values,
   mask-derived values, and readfirstlane-derived values must remain rejected.
5. Add positive and negative structural/execution tests before touching the live path. Do not specialize on program
   address `0x3017460000`.

Relevant machinery:

```text
prosper/src/gpu/rdna2_to_spirv.cpp
  detect_forward_ifs
  analyze_barrier_phased_compute
  emit_cfg_state_machine
  structured_compute_wave_cfg / top_level_pc
  scalar provenance and constant analysis

prosper/tests/test_rdna2_to_spirv.cpp
  kernel43a2 VCC-with-barrier negative
  barrier-phased compute tests
  exact-wave CFG tests

prosper/tests/test_rdna2_spirv_struct.cpp
prosper/tests/test_game_compute.cpp
prosper/tests/test_recompile_coverage.cpp
prosper/tools/shader_inspect
```

The existing barrier-phased path peels a workgroup-uniform SCC outer guard, emits barrier-free phases, and leaves
barriers in a uniform shell. It is a plausible transformation model, but pc915's VCC predicate requires a stronger
provenance proof first.

## Suggested initial allocation for a new orchestrator

Use three title agents plus the orchestrator:

| Agent | First bounded task | GPU |
|---|---|---|
| Dragon Quest VII | Post decode evidence, then point/linear b36 A/B and inspect exact 3D sample semantics | Two short shared correctness replays |
| Astro Bot | Stored/current op221 A/B, then binding49 selector-miss/hit value trace | Short shared correctness replay |
| Plucky Squire | Map pc915 entry SGPRs and barrier-region CFG; no code without uniformity proof | None initially |
| Orchestrator | Validate evidence, prevent overlap, review/publish worthy diffs, keep master/current docs synchronized | Coordinates leases |

Dragon Quest currently has the shortest path to a visually meaningful correction. Astro has the strongest candidate
for a shared R11G11B10 storage-lowering fix. Plucky is valuable recompiler work but must remain proof-driven because
an invalid barrier transformation can hang or silently misexecute a workgroup.

If one lane stalls on missing evidence, reassign that agent to an easier title rather than manufacturing speculative
code. Real progress in any game is more valuable than preserving the original title allocation.

## Ready-to-send subagent prompts

### Dragon Quest VII

```text
Read CLAUDE.md, docs/GAME_COMPAT_ORCHESTRATION.md, docs/DRAGON_QUEST_STATUS.md,
and issue #1486. Create a private worktree and branch from the exact current origin/master.
Reuse ~/agent-tmp/dq-cap3/black.prgcap. First post the retained full-volume LUT decode result
from the orchestration doc to #1486 with public paths sanitized. Then run only the bounded op117
binding36 PROSPER_TESTTEX_FILTER point-versus-linear A/B, including a baseline if needed. Report
exact commands, hashes, op117 completion, pixel metrics, and interpretation. If filtering does not
explain the overexposure, inspect the exact draw90 b36 3D sample semantics and backend sampler state.
Do not edit until you can name a generic contract and regression. No live game run and no exclusive
GPU request for these short correctness replays.
```

### Astro Bot

```text
Read CLAUDE.md, docs/GAME_COMPAT_ORCHESTRATION.md, the Astro handoff doc, and issue #1459.
Create a fresh private branch from current origin/master; the old closure branch is evidence-only.
Reuse ~/.local/state/prosper/evidence/astro-worldmap-c5698376-froute.KyRLdP/worldmap.prgbundle.
Do not implement compute-writer closure: b6 is already captured and seeded, and no prior included
writer exists. Run a bounded stored-versus-current op221 replay, prove module substitution, and compare
outputs; then use a selector-miss and draw206/b49 hit to prove final 1x1 contribution. Localize black
to op221 output or downstream composition before editing. If a tool gap remains, propose the smallest
generic post-operation resource dump. No live recapture and no exclusive GPU request initially.
```

### The Plucky Squire

```text
Read CLAUDE.md, docs/GAME_COMPAT_ORCHESTRATION.md, issue #1554, and PRs #1564/#1566.
Transfer the clean investigate/plucky-pc915-proof worktree at current master or create a fresh equivalent.
Reuse ~/plucky-work/post1564-live-retry.fwpVFI/shader-dumps/exec_cs_3017460000.bin.
Enumerate the pc915 arm CFG/barriers and map entry SGPRs s12:s14 from compute launch metadata.
Prove whether (s16>0)&&(s14==s17-1) is identical across every workgroup wave. No code until a
mechanical workgroup-uniform provenance rule exists; guest shader existence is not proof. Report a
generic positive/negative test design or a defensible conclusion that the branch must remain rejected.
This task is CPU/offline and needs no GPU.
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
