# GPU replay

**Getting a capture:** either the env-driven `PROSPER_GPU_CAPTURE`/timeline flows below, or — the fastest
for a graphical bug or FPS drop you can *see* — press **F9** while a title runs in `prosper-app` to grab
the current frame into a replayable `.prgbundle` + screenshot (`PROSPER_CAPTURE_DIR`, default cwd; it seeds
the renderer-owned RTTs the frame samples, so deferred titles replay for real). Replay it with
`--bundle <file>`; see `tools/AGENTS.md` (interactive frame grab).

`gpu_replay` inspects, validates, graphs, and renders a local `.prgcap` without booting the guest.
Capsules contain title-derived shaders, resource bytes, addresses, ordered DMA endpoints, optional rendered RTT
pixels, and optional exact persistent Vulkan depth/stencil checkpoint planes.
They are gitignored local artifacts and must never be committed or shared as project fixtures.

For a long scripted route, first set `PROSPER_CAPTURE_SCREENSHOT_AT_FRAME=N` to read back the Nth
successfully presented host frame without paying the cost of a command capture. The one-shot writes
`scheduled_frame_N.bmp` under `PROSPER_CAPTURE_DIR`; set `PROSPER_CAPTURE_SCREENSHOT=/path/out.bmp`
to choose an exact path. Use that visual checkpoint to select the nearby
`PROSPER_CAPTURE_BUNDLE_AT_PRESENT` value on the next run, then replay the resulting bundle normally.
The app logs both the guest-present count saved when it arms the screenshot and the count observed after
the readback finishes. Use the armed count, not the host frame number or later written count: swapchain
presents and guest VideoOut flips are separate clocks, and the guest can advance during readback.

Normal capture preflights merged resource ranges and rejects plans above 512 MiB before allocating.
`PROSPER_GPU_CAPTURE_MAX_MB=1..3072` overrides that bound. If descriptor metadata is the evidence you
need, `PROSPER_GPU_CAPTURE_METADATA_ONLY=1` writes a thin capsule with shaders, operations, pipeline
state, and resource descriptors but no guest resource or RTT bytes. Thin capsules support
`--inspect-only`, `--validate`, and `--graph`; rendering exits with a concrete error.
Inspection reports each descriptor's declared size, capture-planned footprint, and captured byte
count separately, so a thin capsule can still expose a pathological range. Capture v28+ retains the exact
planned span and reports the resolved `row-pitch` for linear sampled images; older captures derive the
guest pitch while retaining their historical tight byte span.
The `vs=` and `fs=` summaries identify whether each recompiled shader is retained as `owned` or
`shared`; their size and hash always describe the accessor-selected words the renderer consumes.

Build from `prosper/`, using the worktree-local Linux build:

```bash
cmake --build build-linux -j8 --target gpu_replay
```

Create a capsule with the native-speed workflow in
[`../gpu_timeline/README.md`](../gpu_timeline/README.md). Then use:

```bash
./build-linux/gpu_replay --inspect-only /tmp/submit.prgcap
./build-linux/gpu_replay --validate /tmp/submit.prgcap
./build-linux/gpu_replay --graph /tmp/submit.prgcap
./build-linux/gpu_replay --graph-json /tmp/graph.json /tmp/submit.prgcap
./build-linux/gpu_replay /tmp/submit.prgcap /tmp/replay.bmp
./build-linux/gpu_replay --prepend /tmp/producer.prgcap /tmp/consumer.prgcap /tmp/closure.bmp
./build-linux/gpu_replay --bundle /tmp/window.prgbundle /tmp/closure.bmp
./build-linux/gpu_replay --bundle /tmp/window.prgbundle --bundle-tail 2 /tmp/tail.bmp
./build-linux/gpu_replay --bundle /tmp/window.prgbundle \
  --bundle-intermediate-through-target 642x362 /tmp/closure.bmp
./build-linux/gpu_replay --bundle /tmp/window.prgbundle \
  --bundle-final-capsule /tmp/final-seeded.prgcap /tmp/closure.bmp
./build-linux/gpu_replay --bundle /tmp/window.prgbundle \
  --bundle-extract-submit 19046 /tmp/submit-19046.prgcap
./build-linux/gpu_replay --bundle /tmp/window.prgbundle --bundle-compact /tmp/compact.prgbundle
./build-linux/gpu_replay --bundle /tmp/window.prgbundle --bundle-zero-boundary /tmp/zero-ab.bmp
```

## Raw-vs-realized draw check — `raw=` on each `--inspect-only` draw line (#1256)

Each `draw[ID] item=I … vcount=N indices=M voffset=V modifier=D topo=T raw=…` line reports the **raw
draw-packet state** the guest submitted, decoded BEFORE realization. `raw=` is available in capture
v23+, while `voffset=` and `modifier=` are available in capture v27+:

```
draw[12] … vcount=6 indices=0 topo=3 raw=6              # healthy: realized == guest's DrawIndexAuto count
draw[12] … vcount=1024 indices=0 topo=3 raw=6 [INFLATED] # realized swept more than the guest asked for
```

- **non-indexed** (`indices=0`): the realized `vcount` MUST equal `raw` (the DrawIndexAuto count). `[INFLATED]`
  = realized > raw+1 (the GTA #1163 shared-vertex-pool signature); `[DIVERGENT]` = realized < raw. A realized
  `raw+1` is NOT flagged — a PS5 RectList draw legitimately synthesizes one extra procedural corner (3 -> 4).
- **indexed**: `raw=N(idx)`; the fetched index count (`indices=`) must equal `N`, else `[INDEX-COUNT-MISMATCH]`
  (also fires when an indexed draw fell back to non-indexed on an unreadable index buffer — a real signal).
- `raw=?` = a pre-v23 capture (the raw count was not retained). No flag is possible for those.
- `voffset=V` is the signed `GE_INDX_OFFSET` applied as Vulkan `firstVertex` for non-indexed draws or
  `vertexOffset` for indexed draws. Pre-v27 captures report zero because they did not retain it.
- `modifier=D` is the raw 64-bit `ShaderDrawModifier`, retained for offline decode diagnostics in v27+.

This makes a whole class of decode/realization-divergence bug — where prosper renders geometry the guest did
not ask for — visible **offline** from a capsule, without a live boot. It was added because #1163 (the black
menu wedges) could only be diagnosed by booting the game repeatedly with `PROSPER_DRAWLOG`; the capsule stored
only the realized `vcount=1024`, hiding that the guest's real count was 6.

## Reading graph output

- `edge producer=P consumer=C` means operation `C` reads a range last written by earlier operation `P`.
- `external` means no earlier writer exists inside this capsule. Static assets are normally external.
- `consumers=N first=C` deduplicates one external address/range used by multiple operations.
- `future-writer=N` means the selected submit reads the range before rewriting it. This is a temporal
  version and the strongest signal that an earlier submit must join the replay closure.
- `future-writer=-1` means this capsule never writes the range. It may be static/CPU data or have a producer
  further back in GPU history. Cross-submit resolution is tracked by #595.
- `missing operation` is semantic work whose shader/resource contract could not be realized. A graph or replay
  containing one is explicitly partial; do not treat its rendered image as a faithful oracle.

The JSON contains `nodes`, `edges`, and deduplicated `external_leaves` with the same operation indices,
logical addresses, byte ranges, dimensions, bindings, stages, and future-writer identities.

## Pixel oracles

Renderer-invocation captures record an expected byte count/hash and replay exits nonzero on mismatch.
Timeline-selected captures occur before Vulkan and report `oracle=no`; they render normally but do not invent an
expected hash. `--allow-mismatch` is for an intentional differential such as `--draw N:M`.

## Cross-title regression gate — `regress.py` (#1258)

`tools/gpu_replay/regress.py` replays a whole **corpus** of `.prgcap` capsules and diffs each one's rendered
output hash against a committed baseline. It runs in seconds (deterministic offline replay, no live game boot),
so a shared GPU/executor/recompiler change can be checked against many scenes across many titles **before** the
~18-minute live-boot snapshot suite — the fast pre-snapshot gate the GTA #1163 fix (#1255) lacked.

```bash
export PROSPER_GPU_REPLAY=build-linux/gpu_replay
python3 tools/gpu_replay/regress.py update <corpus-dir>   # record baseline hashes after an INTENDED change
python3 tools/gpu_replay/regress.py check  <corpus-dir>   # exit 1 if any capsule's hash CHANGED (a regression)
python3 tools/gpu_replay/regress.py list   <corpus-dir>   # just print each capsule's current hash
```

The corpus (`.prgcap` files) is **local and gitignored** — capsules are large and carry game imagery, exactly
like the game dumps. Only the small baseline (`regress-baseline.json`, basename → hash) is meant to be committed
and shared. `check` fails on a changed hash or a capsule that failed to replay; `new`/`missing` capsules are
reported informationally (pass `--strict` to also fail when a baselined capsule is missing from the corpus,
so a locally-deleted capsule can't silently shrink coverage). The child gpu_replay runs with `PROSPER_*`
scrubbed from its environment, so a diagnostic left in your shell can't perturb a hash into a false regression.

**Scope (important):** a replay hash validates the **translation** path (recompiler, AGC/PM4 decode,
render-state resolve, executor ordering, detile) — it is the right guard for changes to *that* code. It does
**not** exercise live GPU residency; a change to `live_gpu_targets`/residency can be hash-identical yet wrong
(the capture/replay blind spot, #1103). Use it to catch translation regressions fast, not as full coverage.

## Isolation and extraction

```bash
./build-linux/gpu_replay --draw 12:18 /tmp/submit.prgcap /tmp/pass.bmp
./build-linux/gpu_replay --draw 18 --draw-with-compute-prefix /tmp/submit.prgcap /tmp/draw.bmp
./build-linux/gpu_replay --through-operation 52 /tmp/submit.prgcap /tmp/prefix.bmp
./build-linux/gpu_replay --draw-steps /tmp/steps/s --draw-steps-target 3840x2160 \
  /tmp/submit.prgcap                              # visual bisection filmstrip (see below)
./build-linux/gpu_replay --warmup-repeats 2 /tmp/submit.prgcap /tmp/converged.bmp
./build-linux/gpu_replay --dump-resource 18:ps:34 /tmp/texture.bin /tmp/submit.prgcap
./build-linux/gpu_replay --dump-rtt-seed 0x7f9f504b0000 /tmp/history.bmp \
  --inspect-only /tmp/submit.prgcap
./build-linux/gpu_replay --dump-shader 18:fs /tmp/fragment.spv /tmp/submit.prgcap
./build-linux/gpu_replay --dump-realized-shader 18:fs /tmp/fragment-rdna2.bin /tmp/submit.prgcap
./build-linux/gpu_replay --dump-compute 0 /tmp/compute.spv /tmp/submit.prgcap
./build-linux/gpu_replay --compute-only 0 /tmp/submit.prgcap
./build-linux/gpu_replay --compute-only 0 --override-compute-spv 0 /tmp/reduced.spv \
  /tmp/submit.prgcap
./build-linux/gpu_replay --dump-compute-resource 0:2 /tmp/storage.bin /tmp/submit.prgcap
./build-linux/gpu_replay --dump-failed-shader 0:1 /tmp/failed-fragment.bin /tmp/submit.prgcap
```

## Per-draw "fragment funnel" — `PROSPER_DRAW_STATS` (first thing to run on a missing/wrong draw)

```bash
PROSPER_DRAW_STATS=1 ./build-linux/gpu_replay /tmp/submit.prgcap /tmp/out.bmp
# [draw-stats] draw=4 verts=1024 prims=341 after_clip=0   fs_inv=0      samples=0        GEOMETRY-VANISH(...)
# [draw-stats] draw=5 verts=1024 prims=341 after_clip=68  fs_inv=0      samples=12102811 passed-samples(...)
```

Wraps every realized draw in Vulkan pipeline-statistics + occlusion queries and prints one line per draw
showing **where its pixels vanished** — objective per-draw truth, no oracle needed. A GPU draw can only
disappear in four places, and this pinpoints which:

- `prims>0, after_clip=0` → **GEOMETRY-VANISH**: every primitive was clipped/degenerate/off-screen (a
  vertex-shader / vertex-fetch / transform problem — the geometry never reached the rasteriser).
- `after_clip>0, samples=0, fs_inv=0` → **NO-RASTER**: rasteriser produced no fragments (backface cull,
  empty scissor, zero-area).
- `after_clip>0, samples=0, fs_inv>0` → **TEST-KILLED**: fragments ran but the depth/stencil test rejected
  every sample.
- `samples>0` → **passed-samples**: colour and/or stencil was written (for a colour-write-disabled
  stencil-only draw `fs_inv` is 0 because the fragment shader is optimised out, but `samples` still counts —
  which is why `samples` is the ground truth for "survived", checked before `fs_inv`).

This is the fastest first-order triage for "why did this draw render nothing/wrong": it replaces manual
`--through-operation` bisection with a single glance. Localised GTA V's menu black-wedge defect (a stencil-
counting clip whose first mask draw came back `GEOMETRY-VANISH`) in one run. Requires the device to advertise
`pipelineStatisticsQuery`; inert (no output, no cost, byte-identical rendering) when the env var is unset.
Works on both `gpu_replay` and the live app because it lives in the shared render path.

## Per-draw geometry probe — `PROSPER_GEOM_PROBE=N` (where did draw N's vertices actually land?)

```bash
PROSPER_GEOM_PROBE=4 ./build-linux/gpu_replay /tmp/submit.prgcap /tmp/out.bmp
# [geom-probe] draw=4 item=2 operation=19 reused stored VS with xfb in GS (1700 VS words, 676 GS words)
# [geom-probe] draw=4 verts=1024 finite=1024 on-screen=0 clipped=1024 (offscreen=1023 w<=0=1 nan/inf=0)
# [geom-probe]   clip-bbox x[-4.86,0] y[-1.14,3.81] z[0,0] w[0,1] -> ALL-VERTS-OUTSIDE-CLIP-CUBE(...)
# [geom-probe]   v0 = (-2.9, -0.771, 0, 1) ...
```

The natural follow-up when the fragment funnel reports **GEOMETRY-VANISH**: capture semantic draw N's **post-transform
clip-space vertex positions** via `VK_EXT_transform_feedback` and report where they landed — clip-space
bounding box, on-screen vs clipped counts, `w<=0` (behind-camera), NaN/inf, degenerate (all-collapsed), plus
the first few raw `vec4`s. This distinguishes the ways a `GEOMETRY-VANISH` can happen:

- all verts at one point → **DEGENERATE** (the vertex fetch returned constant/zero data);
- `w<=0` for all → **behind camera** (transform / w defect);
- spread but the bbox sits outside `[-1,1]` → **off-screen** (a transform/matrix defect shifting it away);
- NaN/inf → a numeric defect.

Read it **with** the funnel: the funnel owns the "does it rasterize" verdict (`after_clip`), the geom probe
owns *where the geometry is* — a large full-screen quad can have every vertex just outside the cube yet still
rasterize, so "all verts clipped" is not "renders nothing"; the bbox tells them apart.

Mechanics: on a capsule (stored SPIR-V) gpu_replay resolves semantic draw N to its compact item, then
decorates the **last pre-rasterization stage** for XFB. That is normally a recompiled raw RDNA2 VS. If explicit
fragment interpolation inserted Prosper's generated GS, gpu_replay rebuilds that GS with XFB output and
reuses the stored VS instead; Vulkan sources transform feedback only from the final such stage.
A v19+ capsule is sufficient for an ordinary vertex shader; v31+ additionally retains a separately-installed
NGG main stage and its graphics-LDS allocation, allowing linked prolog+main programs to be probed faithfully.
Existing guest geometry stages cannot yet be rebuilt and are rejected visibly. The live app follows the same
stage selection while recompiling fresh. Requires `VK_EXT_transform_feedback` (RADV, lavapipe). The captured
final `gl_Position` values are faithful; inert and byte-identical when the env var is unset.

### Geometry-health line (`[geom-health]`, printed alongside the probe)

The probe also emits a one-line **geometry-health** verdict for draw N, computed from the same
transform-feedback data (consecutive triples are the actual rasterized triangles — TF decomposes
strips/fans). It surfaces the tells that localize a vertex-fetch / draw-range bug without hand-analysis:

```
# [geom-health] draw=12 verts=1023 unique-pos=12 (max-mult=918) triangles=341 degenerate=334(98%) real=7 duplicate-tri=326 -> DEGENERATE-HEAVY(...)
```

- **`unique-pos` / `max-mult`** — distinct post-transform positions and the multiplicity of the most-repeated
  one. Few unique + huge `max-mult` = most verts collapsed onto a point → the fetch/transform returned a
  constant, or the draw read **past its real vertex range** into zero/stale data.
- **`degenerate(%)`** — triangles with ~zero NDC area (collapsed/stitching).
- **`duplicate-tri`** — exact-repeat triangles (same three positions) = pure overdraw.
- **verdict**: `COLLAPSED` (<=2 distinct positions), `DEGENERATE-HEAVY` (>=80% zero-area — e.g. a shared vertex
  **pool** swept past a small draw's real count, or strip-stitching read as a list), `DUPLICATE-TRIANGLES`
  (exact repeats), or `ok`. Overdraw itself belongs to the funnel — read this **with** `PROSPER_DRAW_STATS`
  (occlusion `samples` > covered pixels). This is the exact signature that localized GTA V #1163's
  non-indexed vertex-count inflation (real count 6 rendered as 1024 → 98% degenerate, 12 unique positions).

`PROSPER_GEOM_PROBE_DUMP=path` additionally writes **every** post-transform vertex (`i,x,y,z,w` CSV, in
primitive-assembly order) for offline per-triangle analysis. Both are inert when unset.

## Shader I/O value tap — `PROSPER_SHADER_TAP=PC` (what did the shader compute at instruction PC?)

```bash
PROSPER_SHADER_TAP=60 PROSPER_GEOM_PROBE=4 ./build-linux/gpu_replay /tmp/submit.prgcap /tmp/out.bmp
# [geom-probe] draw=4 SHADER-TAP: values below are the tapped VGPR (dst+3) at that PC, not clip positions
# [geom-probe]   v0 = float(0, 2.04e-41, 0, 0) hex(00000000 000038bc 00000000 00000000)
```

The deepest probe: capture a **shader's intermediate value** — a `MUBUF` vertex-fetch result, a decoded
constant, a pre-transform coordinate — at the instruction whose PC you name (the same PC `shader_inspect`
prints). The recompiler snapshots that instruction's destination VGPR (and the next 3) as a vec4 and redirects
the vertex-position export to it, so the geometry-probe capture reads the value back per vertex. Output shows
both float and raw hex (intermediate values are frequently integers/bitfields — read the hex for those).

Workflow: `shader_inspect` the draw's VS (`--dump-realized-shader N:vs`) to find the PC whose value you want,
then `PROSPER_SHADER_TAP=<pc> PROSPER_GEOM_PROBE=N`. This answers "is prosper fetching the right vertex data /
computing the right intermediate?" without an oracle — it revealed GTA V #1163's mask draws fetch plausible
large-integer control coordinates (so the off-screen result is data-driven, not garbage). Vertex stage only; the
tapped draw's render is garbage in a tap run (position is redirected) but the captured values are exact; inert
and byte-identical when the env var is unset. Tap a PC in the shader's **straight-line region** (e.g. the
vertex-fetch/transform prologue) — a PC inside a loop or if-body defines the value in a block that doesn't
dominate the position export, so the shader fails to compile (fail-visible: the draw drops, no output line).

## Fragment I/O value tap — `PROSPER_FS_TAP=DRAW:PC` (what did the fragment shader compute at PC?)

```bash
PROSPER_FS_TAP=6:27 ./build-linux/gpu_replay /tmp/submit.prgcap /tmp/out.bmp
# [fs-tap] draw 6 FS re-recompiled with colour tap (1740 words)
# -> draw 6's pixels in out.bmp now show its sampled texel (the value at pc=27), not its real colour
```

The fragment-stage sibling of `PROSPER_SHADER_TAP`, for "why is this **pixel** the wrong colour?" — capture a
fragment shader's intermediate at instruction `PC` (the same PC `shader_inspect` prints) and **redirect the
MRT0 colour export to it**, so semantic draw `DRAW`'s pixels in the rendered frame *are* the value visualised. Unlike the
VS tap it needs no separate capture — the output BMP is the readout. Point `--dump-realized-shader DRAW:fs` +
`shader_inspect` at the FS to find the PC (an `image_sample`/MIMG result, a UV, a pre-blend colour), then
`PROSPER_FS_TAP=DRAW:PC`. gpu_replay re-recompiles only that draw's FS (needs a v19+ capsule with the raw FS
stream); the rest of the frame renders normally, so read the tapped draw's region.

Precision: the value goes through the colour attachment, so it inherits that target's format — values in [0,1]
(UVs, colours, factors, alpha) show directly; larger values clamp (8-bit targets) — read them as colour, not
exact floats. It confirmed GTA V #1163 draw 6 samples the correct artwork texture (so the colour path is fine;
the defect is the stencil masks). Fragment stage only; tap a straight-line PC (same loop/if caveat as the VS
tap); inert and byte-identical when the env var is unset. The re-recompile drops system inputs, so a tapped
value derived from `gl_FragCoord`/sample position reads 0 (visualise fetch/sample/colour intermediates, not
FragCoord-dependent terms).

## Offline recompiler A/B — `--recompile-raw` (does a recompiler change fix/regress this frame?)

```bash
./build-linux/gpu_replay ~/cap/frame.prgcap /tmp/stored.bmp                    # stored (capture-time) shaders
./build-linux/gpu_replay --recompile-raw ~/cap/frame.prgcap /tmp/current.bmp   # CURRENT recompiler
# [recompile-raw] substituted vs=1563 fs=1563 kept-stored vs=0 fs=0 of 1563 draws
# [recompile-raw] substituted compute=52 kept-stored=0 of 52 dispatches device-formats=0x3ff
```

A capsule stores already-recompiled SPIR-V, so recompiler changes are invisible to a default replay.
`--recompile-raw` re-recompiles **every** retained raw VS/FS with the current recompiler and substitutes the
results — any v19+ capsule becomes a deterministic ~seconds-per-iteration A/B vehicle for graphics work
(compare the replay hashes / BMPs), instead of a multi-minute live route per experiment. This was the
iteration loop that localized #1394 for #1287. Items without a retained raw stream, or whose re-recompile
fails, keep their stored SPIR-V — the `[recompile-raw]` line counts them; a nonzero `kept-stored` means the
A/B is partial, never silent. Composes with `PROSPER_FS_TAP` (masked during the mass loop so the per-draw tap
block keeps exclusive ownership of its semantic draw). Interface hints are re-derived from the raw streams,
the same contract as the probes above.

Capture v39 extends the same loop to realized compute dispatches. It retains the bounded raw RDNA2 stream,
user SGPR push constants, launch/thread ABI, wave/TGID/LDS controls, and the capture host's optional typed-
storage capability mask. A rendering replay initializes Vulkan before recompilation and selects storage
formats and subgroup contracts from the replay device's actually enabled capabilities. Non-rendering
`--inspect-only`, `--validate`, and `--graph` invocations can rebuild without Vulkan using the recorded
capture-host policy; combine `--dump-compute` with `--inspect-only` for the same offline-only behavior. The
stored SPIR-V remains the default replay artifact. Capture-bound live translation deliberately stores its
portable module, so the stored/current A/B directly measures device-specific native-width paths without
making them mandatory for other replay hosts. Pre-v39 captures stay readable and report that their compute
modules were kept rather than inventing raw source or launch state.

Two render_runner provenance logs pair with it when a capsule replay disagrees with expectations
(both inert by default): `PROSPER_MIPLOG=1` prints each texture binding's mip-eligibility inputs
(extent, declared chain, format, RTT/DS identity, LOD clamps, filters — which #1272 gate leg starved a
chain), and `PROSPER_BUFLOG=1` prints each buffer binding's upload provenance (set/binding/word count/
identity/first floats, capped) — ground truth for "which bytes did the shader actually see" (the #1287
palette-UV audit).

`--dump-shader DRAW:vs|fs PATH` writes the recompiled SPIR-V. `DRAW` is the semantic ID printed by
`--inspect-only`, as it is for the probes above. Capture v19 adds
`--dump-realized-shader DRAW:vs|vs-main|fs PATH` for the exact bounded raw RDNA2 stream that produced that realized
draw stage, suitable for `shader_inspect`. The VS/FS streams use the same content-addressed 64 KiB-per-stage,
64 MiB-total store as failed-shader diagnostics. Captures v1-v18 remain readable; because they did not retain
realized-stage source identities, this command reports the raw stream as unavailable instead of guessing.

Selected draw ranges and ordered prefixes write the BMP at the final selected draw target's extent when
the returned RGBA byte count confirms that native size. Presentation-scaled replays retain the capsule's
top-level extent, so a recorded native target cannot mislabel a deliberately scaled pixel buffer.

### Failed operations

Capture v7 and later retain bounded diagnostics for draws and dispatches that semantic PM4 ordering contains but the
executor cannot realize. `--inspect-only` prints the failure reason, decoded target/pipeline or compute-launch
state, every referenced stage address, resource-table presence/count, descriptor issues, recompile coverage,
and the first rejected opcode/format at its exact dword PC. It also prints the raw RDNA2 content hash, byte
count, and whether the retained stream reached `s_endpgm`.

`--dump-failed-shader FAILURE:STAGE PATH` writes that raw stream for `shader_inspect` or a focused recompiler
fixture. Both indices are the zero-based values printed by `--inspect-only`; they are not draw indices. Each raw
stage is fault-safely read once, content-deduplicated, capped at 64 KiB, and stopped at the first decoded
`s_endpgm`/unknown instruction or the cap. Total failed-stage data is capped at 64 MiB, diagnostics cannot
outnumber semantic operations, and every reference/hash is validated while reading. Captures v1-v6 remain
readable and print `failure-diagnostics: unavailable (capture predates v7)` rather than inventing evidence.
An indirect compute dispatch whose argument producer failed remains explicitly unrealized with an unknown/zero
launch, but current runtime capture enriches that failure from its retained register snapshot so the exact raw
compute program and descriptor metadata remain inspectable and dumpable. It does not invent the missing launch
specialization; failed-compute retry still requires a later capture whose dispatch arguments were resolved.

`--retry-failed-stage FAILURE:STAGE` reruns one retained stage through the current recompiler with its exact
captured resource table. For split vertex programs, `--retry-failed-chain FAILURE` instead reconstructs the
first two retained vertex stages as one prolog/main chain and calls the production chain recompiler. Both modes
exit successfully only when SPIR-V is produced and avoid Vulkan rendering, so they are the fast feedback path
after a translator change. Chain retry requires the exact resource metadata added in capture v35 and rejects
older or incomplete diagnostics explicitly.

Capture v8 adds persistent depth/stencil checkpoints. Each seed stores the renderer's complete guest cache
identity (depth/stencil read/write bases, HTILE base, extent, and D32/D32S8 format), independent depth/stencil
validity, and raw valid-plane bytes. Counts, extents, formats, duplicate identities, per-plane lengths, and a
1 GiB total are validated before allocation or Vulkan upload. Captures v1-v7 remain readable with zero DS
seeds. `--inspect-only` prints every seed's identity, validity, byte counts, and content hashes.

Compute selectors use the realized compute index printed by `--inspect-only`; v39 also prints `raw=yes` when
that dispatch has complete current-translator replay state. The resource selector is
`COMPUTE:BINDING`; it writes the captured pre-dispatch storage-buffer bytes, while `--dump-compute` writes
the exact specialized SPIR-V executed by replay (the rebuilt module when combined with `--recompile-raw`).
`--compute-only N` retains just that realized dispatch and
its captured resources, making a driver or recompiler failure deterministic without running unrelated draws
or dispatches. `--override-compute-spv N PATH` replaces that dispatch's module after capture materialization;
the input must be a 20-byte-to-16-MiB SPIR-V binary with the standard magic word. Overrides intentionally
disable the capture pixel oracle and are for differential diagnosis, not correctness evidence.

Validated tiled 1D/2D storage upload/writeback runs by default. Setting
`PROSPER_DISABLE_COMPUTE_TILED_2D_STORAGE=1` restores the old skip for an explicit replay A/B. Use a bounded
capsule and the compute-only selector when localizing a host Vulkan compiler or execution failure. With
`PROSPER_COMPUTELOG=1`, replay identifies whether failure occurred while
creating the pipeline, submitting it, or waiting for the dispatch.

For a long live run, `PROSPER_COMPUTELOG_CODE=0x...` and
`PROSPER_COMPUTELOG_SIZE=N` restrict before/after hash diagnostics to dispatches matching the configured
program address and storage-buffer byte size. Either filter may be used alone.

`--dump-rtt-seed ADDR PATH` writes one serialized temporal color surface as an inspection BMP. RGBA8
seeds retain their bytes; RGBA16F seeds are clamped to 0..1 and converted to RGBA8 for viewing. The address
is the `guest_addr` printed by `--inspect-only`. This exposes the input to operation zero, not the surface after
the selected submit executes, and the inspection conversion is not a pixel oracle for HDR values.

Live runs can narrow intermediate-target dumps with
`PROSPER_DUMP_RTGROUPS=<min-nonzero-bytes> PROSPER_DUMP_RTGROUPS_ADDR=0x...`; only a target whose guest base
matches the optional address filter is written. A sampled-texture filter A/B uses
`PROSPER_TESTTEX_FILTER=linear|point` together with `PROSPER_TESTTEX_DRAW=N` and/or
`PROSPER_TESTTEX_BINDING=B`. It changes only the matching descriptor's minification and magnification filters.
These environment variables are localization probes, not fixes, and their output must not replace an unmodified
regression oracle.

Every user-facing `DRAW` selector (`--draw`, `--dump-resource`, `--dump-shader`,
`--dump-realized-shader`, `PROSPER_GEOM_PROBE`, and `PROSPER_FS_TAP`) uses the stable semantic draw ID
printed as `draw[ID]` by `--inspect-only`. The adjacent `item=I` is only the compact realized-vector offset;
it can differ after an unrealized draw and is never a selector. Mixed `operation[N]` ordinals remain a separate,
explicit namespace used by `--through-operation`. Replay restores serialized render-target seeds before
operation zero and uses owned resource memory; it must not dereference original guest mappings. Bundle
operation sources are semantic draw IDs and may contain holes; tooling resolves them through each realized
item's `draw_index`, never as offsets into the compact draw vector.

`--draw-with-compute-prefix` retains every compute operation before the selected draw while discarding the
other graphics draws. This isolates geometry whose vertex/indirect buffers are produced earlier in the same
submit without losing those producers. Plain `--draw` intentionally remains the cheaper graphics-only path.

`--through-operation N` executes the inclusive mixed graphics/compute/DMA prefix `0..N`, preserving operation
order and all earlier work. Prefix output uses the last executed draw target's native dimensions. Use it with
a hash-verified seeded final capsule for fast composition bisection; unlike `--draw`, it does not discard
DMA copies, compute dispatches, or earlier draws.

`--draw-steps PREFIX [--draw-steps-every N] [--draw-steps-target WxH]` is the **visual bisection** primitive —
the fastest way to localize a composition defect (a black quad, a lost layer, a wrong-blended overlay) to the
exact operation that introduces it, with **no oracle screenshot**. It renders the `--through-operation` prefix
for each operation-step and dumps a numbered BMP filmstrip (`PREFIX_op<N>.bmp`) plus a per-step log line:

```
[draw-step] op<=N WxH nonzero-px=… visible-px=… hash=… -> path
```

Scrub the filmstrip (or the log's `visible-px`/`hash` columns) to the first step where the defect appears, then
`--through-operation` / `--draw` around it for the precise draw and `--graph` + `--dump-resource` for what it
samples and why. `--draw-steps-every N` sets the step size (default auto ≈ total/30, a ~30-frame contact sheet);
narrow it to 1 near the divergence. `--draw-steps-target WxH` restricts dumps to steps whose prefix actually
renders that target (matched on the real pixel count) — use it to watch one surface, e.g. the `3840x2160`
scanout, build up instead of every intermediate render target. Prefer `visible-px` (RGB-nonzero pixels) over
`nonzero-px` as the on-screen-content signal: a fully-opaque black frame is ~25% `nonzero-px` from the alpha
channel alone, but `visible-px == 0`. Steps use mixed semantic **operation** indices (like `--graph`/
`--through-operation`), not realized `--draw` indices.

`--prepend` materializes and executes one earlier capsule in the same renderer instance before the consumer.
Its rendered targets take precedence over consumer RTT seeds at matching addresses; unrelated seeds are still
restored. The tool rejects a predecessor whose submit number is not earlier. A changed image proves that the
consumer sampled the retained producer output, but it does not prove faithful closure: graph the predecessor
and continue if it also has temporal leaves.

`--bundle` reconstructs each captured submit through the normal versioned validator, executes them in
ascending order through one renderer instance, and releases each materialized submit before the next.
The summary reports logical versus unique bytes, per-submit output hashes, and every temporal image leaf:

- `stop=included-producer` names the earlier bundled submit whose target matches the leaf's resource
  contract.
- `stop=initialized-seed` means serialized RTT pixels establish the version without another submit.
- `stop=configured-bound` means the earliest bundled submit still needs older history.
- `stop=unresolved-producer` means a later submit has no matching earlier bundled target; a contiguous
  bundle should not produce this and the capture/replay evidence is incomplete.

A successful replay with `configured-bound` is an explicitly partial closure, not a faithful pixel oracle.
Bundle files use `.prgbundle`, contain title-derived data, are gitignored, and must not be committed.

The dependency graph follows the renderer's resource contract rather than treating every overlapping byte
range as interchangeable. Reflected compute descriptors contribute only their proven read and/or write
access; legacy or unreflectable modules retain the conservative read/write fallback. Image producers match a
consumer's exact programmed guest base, while buffers and `DMA_DATA` retain byte-range overlap. A programmed
color attachment is a producer only when its resolved target write mask is non-zero. These rules keep bundle
closure from inventing temporal leaves for write-only compute outputs, neighboring image allocations, or
disabled MRT slots.

`--bundle-tail N` replays only the latest `N` manifests while retaining the bundle's exact shared resource
dictionary. Use it when lifetime evidence proves that earlier submits cannot be target producers; the first
replayed submit becomes the explicit configured boundary and its graph must still be inspected for leaves.

`--bundle-intermediate-through-target WxH` stops each non-final submit after its last realized draw to that
target extent. It avoids rendering later presentation/UI passes when timeline and dependency evidence proves
that only the named target family crosses submit boundaries. The final submit remains complete. Replay aborts
if an included temporal image leaf has another extent or an intermediate submit has no matching draw. This is
an evidence-gated optimization, not automatic dependency closure.

`--bundle-final-capsule PATH` snapshots the complete live color RTT and persistent Vulkan depth/stencil caches
before executing the final submit and writes them as seeds in a standalone current-version capsule. Capture v9
introduced the base-level depth of every 3D image resource; v11 also retains each image T#'s GFX10 DCC flags,
block-size fields, and metadata address. Version 12 captures the exact DCC control-surface bytes for the
validated single-sample/base-level SW_64KB_R_X layout; metadata-only and older capsules keep their absence
explicit. Version 13 tags every temporal color RTT seed as `rgba8` or `rgba16f`, preserving its native byte
width through standalone replay while reading v1..v12 seeds as the historical RGBA8 default.
Version 14 appends address-backed `DMA_DATA` records with exact source and destination guest identities, byte
counts, PM4 order, and content-addressed endpoint blob references. Replay mutates the same owned resource
instances used by later draws and dispatches and invalidates renderer caches for the guest destination range;
v1..v13 capsules remain readable and do not invent DMA operations.
Version 37 appends each realized compute module's required subgroup size. A non-zero value recreates the exact
`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` plus full-subgroups contract used by live rendering;
inspection reports it as `subgroup=32` or `subgroup=64`. This lets captures retain native-wave compute modules
instead of recompiling a different portable emulation solely because capture is armed. Older capsules default
to `subgroup=0`, and replay fails visibly when a host cannot satisfy a captured native contract. Native lowering
defaults to one guest wave per workgroup; `PROSPER_NATIVE_COMPUTE_MULTIWAVE=1` enables the exact experimental
multi-wave contract and records it the same way.
Version 38 extends the append-only RTT-seed format enum with native `r8` and `r32ui` surfaces. Capture now
validates every CPU mirror against the surface's current extent and bytes per pixel, materializes an
authoritative GPU-only target before export, and never labels an unsupported or stale scalar surface as
RGBA8. Version 37 and older capsules retain their existing enum values and byte layout.
`--inspect-only` reports the RTT format and the planned/captured byte counts, non-zero and unique-byte counts,
first control word, and content hash. A software DCC decode is still not inferred. The capsule's standalone
output must match the bundle's final hash before using it for fast `--draw`, operation-prefix, resource, or
shader experiments. Export validates every surface and fails when a required temporal color surface is absent;
it never substitutes zero pixels. The file is installed only after the source final submit renders, and embeds
that output byte count/hash as its replay oracle. A legacy pre-v7 manifest with unrealized operations is migrated explicitly to
`Unknown` failure diagnostics with the original operation kind/source/order, so the current writer remains strict.
The DS snapshot uses transfer barriers to return every image to depth/stencil-attachment layout before the final
bundle submit executes; equality therefore also checks that readback did not perturb the source run.

`--bundle-extract-submit N PATH` materializes one run-local submit manifest as a normal capsule and exits
without Vulkan when no image output is requested. Use it to inspect, graph, or validate the exact predecessor
whose frontier or cache behavior is in question.

`--bundle-find-ds ADDR` scans only the compact submit manifests and reports every submit whose depth/stencil
read or write bases match `ADDR`, including draw range, target extent, depth-test/write counts, clear count,
and compare-op bitmask. It does not reconstruct resource payloads or initialize Vulkan. Use it to establish a
guest DS surface's lifetime before expanding a replay window or changing cache behavior.

`--bundle-ds-summary` scans every depth/stencil-active draw and groups it by the renderer's current guest
identity tuple (depth read/write plus stencil read/write bases). It reports lifetime, target extents,
test/write/clear totals, mixed-identity passes, anonymous passes, and identity transitions for each color
target. The query is manifest-only and can be combined with `--bundle-tail N`. Use it before changing cache
identity or invalidation rules: a mixed pass proves the one-attachment backend is collapsing state, while a
stable identity with no clear keeps the investigation focused on backing-memory/HTILE lifetime.

`--legacy-htile-before-stencil` is an explicit migration diagnostic for the preserved pre-v6 Dead Cells
closure. Capture v5 did not serialize `DB_HTILE_DATA_BASE`; that run's allocations and later v6 captures
prove its HTILE block is exactly 64 KiB before stencil. The switch supplies only that missing cache identity
so compute-write invalidation can be A/B tested against the immutable closure. Current captures retain the
real register and must not use this layout inference.

`--bundle-compact PATH` writes a validated copy containing only resources and chunks reachable from retained
submit manifests. It is useful after a rolling semantic capture; with no image output argument, compaction
exits without initializing Vulkan. Combine it with `--bundle-tail N` to write a compact suffix bundle.

`--bundle-zero-boundary` is an A/B diagnostic. It supplies transparent RGBA pixels for unseeded temporal
image leaves at the oldest bundled submit, labels them `diagnostic-zero-seed`, and leaves the bundle unchanged.
It can disprove stale guest backing versus zero initialization as the cause of a mismatch, but it is not a
faithful replacement for the missing producer history.

## Current Dead Cells reference

The preserved #608 playable closure selects exactly 90 semantic draws with the 738x420 pass at draw 79..81. Its
retained submits 18,165..19,047 represent 158.94 GiB logically and 739 MiB uniquely, resolving all 1,764 temporal
image dependencies. Its original pre-#611/#615 replay hash was `5759c125812154dc`; absolute hashes from that old
renderer revision are historical, while source/capsule equality remains the checkpoint invariant.

On current code, normal #611 invalidation produces `fac9ca4cbbba8196` from both the full bundle and standalone
v8 capsule. With `PROSPER_DS_GUEST_WRITE_INVALIDATE=0`, the exact stale-depth A/B produces
`535256588b67a536` from both paths and byte-identical BMPs. The self-contained capsule holds 12 RTT surfaces,
one valid 642x362 D32S8 depth plane (929,616 bytes), the effective invalidation/legacy-HTILE settings, and its
33,177,600-byte output oracle; standalone replay takes about 3.3 seconds instead of roughly 24 minutes. A
tail-two negative control remains `71b84bdfae53933c` in both bundle and capsule because its two color frontiers
are explicitly bounded. This closes #569 without hiding an incomplete history boundary.

Manifest scanning shows the same 642x362 depth address in all 883 submits, always with LESS_OR_EQUAL
tests/writes and never a draw/register clear. Timeline-v5 backing hashes and writer provenance resolved the
hardware boundary: supported compute program `0x401aec200` fills the exact 32 KiB HTILE allocation with
`0xfffffff0` before the scene draw span. The live backend invalidates overlapping persistent DS entries on guest
GPU writeback (#611); capture v8 preserves the state exactly when an investigation deliberately disables that
invalidation.

For new routed captures, timeline v6 replaces that historical endpoint with 91..94 semantic draws, exactly
8 dispatches, and the 636x420 pass at draw 77..85. `gpu_timeline --signatures` derives the conjunction, and
`--select` proves it has no splash/menu/loading or adjacent-index matches before capture.

The #594/#595 gameplay capsule at submit 18,750 has two external 642x362 temporal versions. Timeline-v4
full-run aggregation shows that both surfaces begin around submit 17,400 in current runs and are rewritten
about 1,200-1,350 times before the selected submit. Their first observed graphics writers have programmed
clear state, color mode 0, target mask `0xf`, and color format 10. These are raw register observations, not
proof that hardware performed an implicit clear. Resource addresses, submit numbers, and semantic operation
ordinals vary between runs; recompute them and never hardcode them into renderer behavior.

A same-run depth-16 bundle for submits 18,735..18,750 resolved all 30 internal temporal edges and reported
exactly two 642x362 leaves at the configured lower bound. Content-defined dedup stored 2.883 GiB of logical
capsules in 166.3 MiB (5.8%). No initialized RTT seed appeared in that window, and even/odd depths alternate
between the dark and overbright failure. Supplying transparent zero to those two lower-bound leaves produces
the same final hash (`62a46f15efa52a26`) as unseeded replay, so stale guest backing versus transparent
initialization is not the cause in this window.

Bundle v2 stores exact resource versions in a global content-defined chunk dictionary. On a fixed 1,200-submit
full-state run it folded 122.97 GiB of logical capture data into 301.1 MiB in 169.4 seconds. Reuse is proven by
hash plus byte equality; mutable versions are never reused from address identity. Linux capture reads use
fault-safe `process_vm_readv` rather than per-page pipe probes.

A timing-independent semantic run retained a rolling 1,400-submit window and selected run-local submit 23,350.
The two 642x362 surfaces first appear one submit earlier. Compacting to that exact two-submit suffix stores
142.6 MiB; replay resolves both temporal edges with zero seeded, bounded, or unresolved leaves and finishes at
hash `112e7db0001791cc`. The image is the dark opening vignette with `HOLD Skip`, not playable gameplay, so
extent plus 80 draws is still too broad. #608 tracks a stable playable checkpoint and first bad composition.
