# Syberia: Remastered (`PPSA30140`) — status and evidence

Unity / IL2CPP, Microids / Virtuallyz Gaming. **Rung 3 — gameplay reached** with real GPU draws on a
validated route (`scripts/syberia/reach-gameplay.pad`). Two composition defects remain, both tracked
on #1619.

Read **`## Ruled out`** below before repeating any experiment on the "right side of the frame is
black" question. Seven hypotheses about that frame — including the one this document originally
narrowed to — are dead: four falsified with evidence, three rejected as not-the-cause or not
established.

## Ruled out

One line per dead hypothesis, the evidence that killed it, and where that evidence lives. Do not
re-derive these without contradictory new evidence.

| Hypothesis | Verdict and evidence | Source |
|---|---|---|
| The published screenshot caught an unfinished slide-in animation | **Falsified.** The panel settles at t=183 s and holds byte-identical for the next 57 s (`crc32=453d400c`, 107,936 distinct colours). | #1619 |
| A scissor or viewport clips draws to the left of the frame | **Falsified.** Every realized draw's `viewport=` and `scissor=` covers its **full** target (1920x1080, or the 2048² / 1024² / 512² shadow-atlas extents). | #1619 |
| The menu is a flat 2D screen with nothing to draw on the right | **Falsified.** The frame is 472 draws + 81 dispatches with 2048² shadow cascades, 512² spot shadows, `R11G11B10F` HDR targets and a 960x540→15x8 bloom pyramid. It renders a real lit interior. | #1619 |
| The `R11G11B10F` gap in `set_live_target_image_written_notifier` is what blacks this frame | **Falsified as the mechanism for this frame** (the *defect* is real and is fixed by #1626). A routed live A/B — same route, same timings, no diagnostics in either arm, differing only in the notifier line — produced a **byte-identical** settled menu (`md5 1fc612606b4e32adc89db9bfd35c92b2`, `crc=bdf7a33a` for every capture from t=165 s to t=220 s), and the right ~55% is still pure black. The verdict rests on that A/B alone and does not need the mechanism. The mechanism, offered as the explanation: the notifier is called from exactly one site (`live_compute.cpp`, guarded by `bi.mirror_result_to_imported`) which the seed-skip path excludes, so it provably cannot fire for a write-only binding proved to cover every texel. That implication is verified by code; its **premise** — that this dispatch's binding 23 took the seed-skip path — rests on a `[seed-skip-verify]` line **recorded once in this issue's body and never re-observed**, which independent review could not confirm either. Treat the premise as a recorded observation, not re-derived fact. | #1619 comments, PR #1626 |
| The offline localization transfers to the default render path | **Not established — do not assume it does.** `PROSPER_RTTLOG` and `PROSPER_RESOURCE_HASH_DIM` both clear `live_gpu_targets` (`live_renderer.cpp:933`) and force the **CPU readback** RTT path. The frame is black on both paths, but reproducing on both does **not** make it the same mechanism on both; the issue's own wording is "black on both, by (at least partly) different code". The chain in `## Where the scene dies, exactly` is sound about the CPU-readback path only. | #1619 comments |
| The ten `reason=shader-recompile` dispatches black the scene | **Not the cause.** The lit composite is already correct at draw 465, after all ten have been skipped. They are real FATAL gaps per `CLAUDE.md` and are tracked separately on #1628. | #1619, #1628 |
| Syberia's ~1.9 fps is #1748-style AGC command-buffer churn | **Falsified.** Over a 45 s headless boot with `PROSPER_DCBFULL=1` the title builds 311,296 command packets and issues **zero** Dcb buffer-full callbacks — it never asks the guest for more command-buffer space at all. The probe prints an `armed` banner and a running `seen=/full=` tally, so this zero is the probe reporting zero rather than a silent log (it was measured once before *without* that banner, and that earlier reading proved nothing). Note the comparison titles are **not** cleared: Bendy runs at 297 callbacks/s, the same order as Asterix *after* #1748 (368/s), so a high rate is not itself the defect — what mattered on Asterix was that the chunks never came back. | #1756, `docs/AGC_PACKET_SIZES.md` |
| The `[agc] WaitRegMem … dependency violated` burst at t≈129 s is a finding | **Not a measurement.** The diagnostic is capped at 40 printed lines (`command_processor.cpp:2450`) and the counter in the message is the true total; an unsatisfied wait is documented in the code as normal handled state. The same over-read was recorded independently on Oregon Trail (#1606) and as instrument trap #13 in `GAME_COMPAT_ORCHESTRATION.md`. | #1619, #1606 |

**Still open:** what blacks this frame on the **default** path. Establish first whether the mirrored
write path is taken for that dispatch at all — `PROSPER_COMPUTELOG_CODE=<code_addr>` is outside the
`live_gpu_targets` exclusion list and therefore observes the real path, but guest code addresses are
run-local, so the recorded `0x2134c14100` cannot be pre-seeded from an older run (a fresh run confirms
it does not appear); filter by `PROSPER_COMPUTELOG_SIZE` or take the full trace.

**The leading unverified hypothesis for the default path**, stated so the next agent has a start, not
as a finding: a seed-skipped write-only storage binding still runs the ordinary writeback — it reads
the storage image back, packs the guest bytes, and calls `notify_guest_gpu_write`. That guest write
invalidates every overlapping renderer-owned target and erases the CPU RTT entry, and because the
mirrored-write path was skipped, **nothing republishes it** — the same end state the notifier bug
produced, reached by a different route. If that is right the contract is broader than the one #1619
states: *a compute dispatch that fully overwrites a renderer-owned target must leave the renderer
owning the result, whether it got there through a seeded mirrored write **or** through a proven
full-coverage write-only store.* To confirm or kill it, check whether `bi.imported` is true for that
storage binding on the default path, and whether the target is re-established for draw 469.

## Route and timing (Linux, hardware Vulkan, RADV, 1920x1080)

`screenshot` / `prosper-app` with `PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct
PROSPER_RENDER=1`. No input is needed to reach the menu.

| t | state |
|---|---|
| 0–9 s | black |
| 12–128 s | "Do not turn off your PlayStation®5 console" autosave notice, animated gears. Renders **full width, centred on x=960** — a useful control that the frame loop composites the whole 1920x1080 surface correctly. |
| 130–156 s | black transition |
| 159–180 s | profile-select menu slides in from the left at ~75 px/s |
| **183 s onward** | **settled.** Byte-identical for the next 57 s (`crc32=453d400c`, 107,936 distinct colours, 670,815 non-black px). |

The settled frame is the one in `assets/screenshots/syberia-profile.png`.

`scripts/syberia/reach-gameplay.pad` continues from there. It is **validated by the run that produced
`assets/screenshots/syberia-title.png` and `assets/screenshots/syberia-gameplay.png`** — do not change
it without re-running:

| t | state |
|---|---|
| 186 s | first Cross; the profile-select menu accepts it |
| 192–232 s | profile confirmation / transition |
| ~280–292 s | **title screen** — Valadilène, the Voralberg factory exterior, "B. Sokal / Syberia Remastered" (85,652 colours, 1,734,739 non-black px) |
| ~312 s onward | **gameplay** — Kate Walker in the factory hall, the "Leave" interaction prompt, the "Use the left stick to move" tutorial, the pause HUD (≈46,000–48,000 colours, 2,068,000+ non-black px, i.e. 99.7% of the frame) |

Gameplay composition is **degraded**: a translucent ghost of another scene is blended over the middle
of the frame and the image is over-dark. That is a second symptom on #1619 and is probably the same
post-process/RTT family as the menu, but it has **not** been localized — treat it as a separate
question until a capture says otherwise.

## The "right ~55% is black" question — answered

**It is a defect, not art direction, and it is not a scissor/viewport clip.** The whole **3D scene
layer** of the menu is black; the UI layer (leather panel, boarding passes, copyright) survives and
happens to occupy only the left 45%.

The hypotheses falsified along the way are listed in `## Ruled out` above.

### The pass chain in one settled frame

Recorded as one 1920x1080 submit (`submit=2611`, 472 draws, 81 computes, 607 operations, 54
unrealized):

```
draws  0..72    depth-only prepass + shadow cascades (target=0, 2048²/1024²/512² viewports)
draws 78..157   scene/G-buffer      -> 0x21089d0000
draws 158..207  second scene pass   -> 0x210b590000
draws 208..408  more shadow/depth   (target=0)
draw  409       clear               -> 0x2110310000
draws 415..462  lighting/HDR        -> 0x210fa50000  (+ bloom pyramid 0x2139520000/0x2110bd0000/0x210d710000)
draws 464..465  lit composite       -> 0x2110310000   <-- SCENE IS CORRECT HERE
compute[79]     code=0x2134c14100   -> 0x2110310000   <-- CONTENT LOST HERE
draw  469       tonemap             -> 0x2112bd0000   (uniform black)
draw  470       full-screen composite of 0x2112bd0000 -> 0x2117810000 (scanout) — black
draws 472..485  UI: leather panel, boarding passes, copyright — the only visible layer
```

### Where the scene dies, exactly

`gpu_replay --through-operation 558` renders `target=0000002110310000` and shows the **complete lit
interior with volumetric light shafts** — 2,066,070 non-black pixels of 2,073,600. The scene is
produced correctly.

Then, in `PROSPER_RTTLOG=1` order:

```
[rtt] pass target=0x2110310000 extent=1920x1080 (2 draws) px_nonzero=8279562 rgb_nonblack=2066070 cache_size=19
...
[rtt] sample tex addr=0x2110310000 1920x1080 fmt=20 -> miss (cache_size=18)
[rtt] pass target=0x2112bd0000 extent=1920x1080 (1 draws) px_nonzero=0 rgb_nonblack=0
```

and with `PROSPER_RESOURCE_HASH_DIM=1920x1080`:

```
[target-version]   target=0x2110310000 draws=464-465 hash=ff1b1567d04fad9a          (content)
[resource-version] draw=469 bind=35 addr=0x2110310000 rtt=0
                   raw=8294400/8294400:0f7752776a1bc383 sample=8294400:792bed5a3f02a383
                   rgb_nonblack=0 alpha_nonzero=2073600 writer=none
[target-version]   target=0x2112bd0000 draws=469-469 hash=0f7752776a1bc383          (uniform black)
```

Between the producing pass and the consuming draw, the RTT cache **loses the entry**
(`cache_size` 19 -> 18) and the consumer falls back to the guest backing, which prosper never wrote.

The operation in between is **`compute[79]`, `code=0x2134c14100`, order `1455519`, groups
240x135x1 local 8x8x1 (= 1920x1080)**. It binds `0x2110310000` both as a sampled texture
(`CS TEX b=5/9/11/12/14`) and as a **writable storage image** (`CS STORAGE b=23`), and
`[seed-skip-verify] code=0x2134c14100 binding=23 texels=2073600 … full-coverage` proves it
rewrites every texel. That storage write is reported as a guest GPU write, and
`invalidate_cpu_rtt_guest_write` (`frontends/shared/live_renderer.cpp:152`) **erases** the RTT
cache entry on a `color_plane` overlap.

Draw 469 (order `1455541`) is the next consumer and reads black.

### Narrowed: `R11G11B10F` is missing from the compute write-back notifier — real defect, **not this frame's mechanism**

> **Superseded as a diagnosis for this frame.** The gap below is real and is fixed by #1626, but a
> routed live A/B showed the settled menu is byte-identical with and without that fix, and the
> notifier provably cannot fire for this dispatch's seed-skipped write-only binding. See
> `## Ruled out`. The rest of this section is retained because the format contract it states is
> correct and the fix is landed.

`0x2110310000` is `fmt=122` = `R11G11B10_FLOAT` (`VK_FORMAT_B10G11R11_UFLOAT_PACK32`).
`LiveTargetPixelFormat` (`src/gpu/gpu_execute.hpp:566`) has three members, and `R11G11B10Float` is
plumbed through the snapshot reader (`live_renderer.cpp:699`), the direct GPU importer
(`live_renderer.cpp:763`) and 16 sites in `frontends/shared/live_compute.cpp`.

**One path was missed** — `set_live_target_image_written_notifier`, `live_renderer.cpp:811`:

```cpp
const VkFormat format =
    write.format == prosper::gpu::LiveTargetPixelFormat::Rgba16Float
        ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
```

An `R11G11B10Float` write falls into the `else`, is reported as `VK_FORMAT_R8G8B8A8_UNORM`, fails
`live_rtt_mirror_identity_matches` against the surface's real
`VK_FORMAT_B10G11R11_UFLOAT_PACK32`, and returns early — so
`restore_persistent_color_target_after_mirrored_write` never runs and `g_rtt[write.gpu_addr]` is never
republished. The ordinary guest-write invalidation for the same storage write still erases the entry,
which is the `cache_size` 19 -> 18 above.

> **Contract.** Every `LiveTargetPixelFormat` a compute dispatch may import must round-trip through the
> write-back notifier. A format the importer accepts but the notifier cannot name is reported as RGBA8,
> fails the mirror-identity check, and silently leaves the target invalidated — the renderer's pixels
> are then lost for every later consumer.

Same class as Dead Cells #773 (native-format loss): RGBA16F was plumbed through this path, R11G11B10F
was not. Map all three enumerators and fail closed on an unknown one, the way the importer at
`live_renderer.cpp:757-764` already does; prefer a shared helper over a third inline ternary.

`tests/test_game_compute.cpp` already drives `LiveTargetPixelFormat::R11G11B10Float` (lines 301, 358,
2156). A regression should assert that a mirrored compute write to an R11G11B10F live target
republishes the RTT entry and that the following sample reads the written pixels — and that it **fails**
on current master while the RGBA8/RGBA16F equivalents pass, so the test proves the format gap rather
than the general path.

**Not yet done:** an A/B proving the frame composites correctly with the notifier fixed.

### Earlier, broader framing (now the less likely of the two)

> A renderer-owned colour target that a **compute** dispatch reads must be materialized from the
> renderer's own copy before the dispatch runs. Today the graphics pass leaves the pixels only in the
> renderer's persistent Vulkan target / CPU `RttSurf`; a compute dispatch that samples the same
> address reads the guest backing, which prosper never wrote; it therefore writes black, and its own
> storage-write notification then erases the renderer's good copy, so every later consumer is black
> too.

What is proven: the surface has content before `compute[79]`, and is black for the draw after it.
What is **not** proven: that `compute[79]`'s *read* of `0x2110310000` resolves to guest bytes rather
than the renderer's copy. Confirm that first — a wrongly-scoped invalidation and a missing
graphics->compute handoff produce the same symptom and need different fixes.

Note that a `graphics -> compute -> graphics` in-place post-process on one surface is a common
engine pattern, so whichever of the two it is, the fix is shared infrastructure, not title-specific.

## Separately: real recompiler gaps in the same frame

Ten dispatches fail with `reason=shader-recompile` (`gpu_replay --inspect-only`). Per `CLAUDE.md`
these are FATAL gaps, not acceptable skips. First-reject sites:

| program | launch | first reject |
|---|---|---|
| `0x211197ea00` | 1920x1080 | pc=24 fmt=0 op=0xf |
| `0x211197ef00` | 30720x68 | pc=4 fmt=4 op=0x8 |
| `0x21341b1a00` (x3) | 960x544 | pc=55 fmt=9 op=0x354 |
| `0x2111a39700` | 1920x1080 | pc=51 fmt=8 op=0xf5 |
| `0x2111a39e00` | 1920x1088 | pc=6 fmt=0 op=0x1 |
| `0x2111a3a400` | 120x72 | pc=36 fmt=4 op=0x8 |
| `0x2111a3af00` | 1920x1088 | pc=73 fmt=3 op=0x13 |

A full boot also logs 13 distinct `[compute] skip unsupported program` addresses. Remember that line
is deduped per `code_addr`: one line means "failed at least once", not "always fails".

These are **not** the cause of the black scene — the composite is already correct at draw 465, after
all of them have been skipped — but they are real work.

## Other observations

- Four unimplemented NIDs across a full boot: `libkernel::NH6xARDOVv8`, `libSceAcm::ZIXln2K3XMk`,
  `libSceNpSessionSignaling::ysmw6J-P8Ak`, `libSceNpTrophy2::EwNylPdWUTM`. None blocks progress.
- `[agc] WaitRegMem … dependency violated` starts at t≈129 s, exactly when the menu scene loads. The
  diagnostic is **capped at 40 lines** (`command_processor.cpp:2450`) — the count is a log cap, not a
  measurement. An unsatisfied wait is documented as normal handled state; do not treat these as a
  finding without an A/B against `PROSPER_WAIT_DEFER=1`.

## Reproducing the evidence

```bash
# One settled-menu submit, ~680 MB capsule
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
PROSPER_NO_FRAME_DUMPS=1 \
PROSPER_GPU_CAPTURE=~/menu.prgcap PROSPER_GPU_CAPTURE_AFTER_MS=190000 \
PROSPER_GPU_CAPTURE_TARGET_DIM=1920x1080 PROSPER_GPU_CAPTURE_MIN_DRAWS=1 \
PROSPER_GPU_CAPTURE_MAX_MB=2048 PROSPER_CAPTURE_TITLE=PPSA30140 \
  ./build-linux/screenshot <DUMP_ROOT>/PPSA30140-app0 \
    --seconds 3 --count 80 --out ~/cap1 --timeout 400

# Then, entirely offline:
./build-linux/gpu_replay --inspect-only ~/menu.prgcap                 # targets, viewports, scissors, failures
./build-linux/gpu_replay --through-operation 558 ~/menu.prgcap ~/scene.bmp   # the correct 3D scene
./build-linux/gpu_replay --through-operation 589 ~/menu.prgcap ~/black.bmp   # the black tonemap output
PROSPER_RTTLOG=1 ./build-linux/gpu_replay --through-operation 589 ~/menu.prgcap ~/x.bmp
PROSPER_RESOURCE_HASH_DIM=1920x1080 ./build-linux/gpu_replay --through-operation 589 ~/menu.prgcap ~/y.bmp
```

Addresses and operation ordinals are run-local; re-derive them with `--inspect-only` on a fresh
capture. Always read the `target=` tag that `--through-operation` prints: adjacent prefixes can
report different surfaces (see the instrument list in `GAME_COMPAT_ORCHESTRATION.md`).

**Instrument warning.** `PROSPER_RESOURCE_HASH_DIM`, `PROSPER_RTTLOG`, `PROSPER_GPU_CAPTURE` and
several other diagnostics all clear `live_gpu_targets` (`live_renderer.cpp:933`), forcing the CPU
readback RTT path. The black scene reproduces both with and without them — the settled frame is black
in an ordinary `screenshot` run with no capture environment at all — but never compare a diagnostic
run against a default run without accounting for this.
