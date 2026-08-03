# Syberia: Remastered (`PPSA30140`) — status and evidence

Unity / IL2CPP, Microids / Virtuallyz Gaming. **Rung 3 — gameplay reached** with real GPU draws on a
validated route (`scripts/syberia/reach-gameplay.pad`). Two visual defects remain: the profile
menu's formerly missing 3D layer is restored but overexposed (#1790), and the routed gameplay
composite remains degraded (#1627).

Read **`## Ruled out`** below before repeating any experiment on the "right side of the frame is
black" question. Several hypotheses about that frame — including the one this document originally
narrowed to — are dead or restricted to the diagnostic CPU-readback path.

## Ruled out

One line per dead hypothesis, the evidence that killed it, and where that evidence lives. Do not
re-derive these without contradictory new evidence.

| Hypothesis | Verdict and evidence | Source |
|---|---|---|
| The published screenshot caught an unfinished slide-in animation | **Falsified.** The panel settles at t=183 s and holds byte-identical for the next 57 s (`crc32=453d400c`, 107,936 distinct colours). | #1619 |
| A scissor or viewport clips draws to the left of the frame | **Falsified.** Every realized draw's `viewport=` and `scissor=` covers its **full** target (1920x1080, or the 2048² / 1024² / 512² shadow-atlas extents). | #1619 |
| The menu is a flat 2D screen with nothing to draw on the right | **Falsified.** The frame is 472 draws + 81 dispatches with 2048² shadow cascades, 512² spot shadows, `R11G11B10F` HDR targets and a 960x540→15x8 bloom pyramid. It renders a real lit interior. | #1619 |
| The `R11G11B10F` gap in `set_live_target_image_written_notifier` is what blacks this frame | **Falsified as the mechanism for this frame** (the *defect* is real and is fixed by #1626). A routed live A/B — same route, same timings, no diagnostics in either arm, differing only in the notifier line — produced a **byte-identical** settled menu (`md5 1fc612606b4e32adc89db9bfd35c92b2`, `crc=bdf7a33a` for every capture from t=165 s to t=220 s), and the right ~55% is still pure black. The verdict rests on that A/B alone and does not need the mechanism. The mechanism, offered as the explanation: the notifier is called from exactly one site (`live_compute.cpp`, guarded by `bi.mirror_result_to_imported`) which the seed-skip path excludes, so it provably cannot fire for a write-only binding proved to cover every texel. That implication is verified by code; its **premise** — that this dispatch's binding 23 took the seed-skip path — rests on a `[seed-skip-verify]` line **recorded once in this issue's body and never re-observed**, which independent review could not confirm either. Treat the premise as a recorded observation, not re-derived fact. | #1619 comments, PR #1626 |
| The offline RTT-cache localization transfers to the default render path | **Not established — do not assume it does.** `PROSPER_RTTLOG` and `PROSPER_RESOURCE_HASH_DIM` both clear `live_gpu_targets` (`live_renderer.cpp:933`) and force the **CPU readback** RTT path. The frame is black on both paths, but reproducing on both does **not** make the cache-miss mechanism common to both; retain those logs only as endpoint/localization evidence for that diagnostic route. The source/capsule producer chain below is separate evidence. | #1619 comments |
| The ten `reason=shader-recompile` dispatches are unrelated to the black scene | **Falsified.** One of them, operation 570 / source 101 (`0x2111a3af00`, raw hash `abef7b52fb82e741`), is the missing producer of the post-process input at `0x21159d0000`. It samples the correct lit target `0x2110310000`, but rejection at `s_cmp_lg_u64 exec,s[16:17]` prevents that output; downstream source 102 and the final source-119 composite therefore consume zero. Supporting the compare recompiles the exact captured shader and a clean live run restores the full-width scene layer: 2,063,996 non-black pixels and 139,946–144,452 colours across t=190–223 s, versus 670,815 non-black pixels in the old settled black-layer baseline. | #1619, #1628 |
| Syberia's ~1.9 fps is #1748-style AGC command-buffer churn | **Falsified.** Over a 45 s headless boot with `PROSPER_DCBFULL=1` the title builds 311,296 command packets and issues **zero** Dcb buffer-full callbacks — it never asks the guest for more command-buffer space at all. The probe prints an `armed` banner and a running `seen=/full=` tally, so this zero is the probe reporting zero rather than a silent log (it was measured once before *without* that banner, and that earlier reading proved nothing). Note the comparison titles are **not** cleared: Bendy runs at 297 callbacks/s, the same order as Asterix *after* #1748 (368/s), so a high rate is not itself the defect — what mattered on Asterix was that the chunks never came back. | #1756, `docs/AGC_PACKET_SIZES.md` |
| The dominant save-warning compute program spends its flat ~10.5 ms in repeated pipeline compilation or shader arithmetic | **Falsified.** An exact-revision, stable-hash-selected phase capture on `cb7602b7` recorded 1,760 successful executions as 160 complete 11-dispatch atlas chains. Pipeline work averaged 0.010 ms (0.1%). The launch shrank from 8,160 workgroups to one while dispatch stayed flat at 1.607–1.671 ms and total stayed 10.394–10.629 ms. Fixed whole-atlas image setup averaged 5.109 ms (48.4%); writeback averaged 3.798 ms (36.0%), including 1.845 ms pack and 1.672 ms retile. Both offline phase checks passed with no dropped records or model warnings. | #1737 |
| Either native-2D transfer candidate exercised Syberia's save-warning atlas | **Falsified; both live measurements were self-invalidating.** On exact candidate `85d26879`, 45 s arms differing only by `PROSPER_NO_NATIVE_2D_COMPUTE_TRANSFER` both rendered 3.79 fps and both left the monotonic transfer counter at zero. Retained live evidence identifies the hot allocation as `img_dim=5`, depth 1, while that candidate and its fixture required `img_dim==1`. Candidate `7454407d` shared the established one-layer/non-arrayed-2D predicate between recompilation and transfer and passed its decisive fixture, but its first clean live ON arm again left the counter at zero; the run stopped before a control as required. Resource `dim=5` does **not** prove the producer or consumer's reflected SPIR-V view is non-arrayed, nor that a retained cache entry reached the borrow gate. No timing from either attempt is performance evidence. | #1737 |
| The repeated save-screen BC6H work is cache-key churn or guest mutation | **Falsified.** A bounded live `PROSPER_DETILE_STATS=1` run observed the same 2048x2048x6 cube at one address through address ordinals 1–8, 16 and 32. Its key stayed `0xfead9ae2753c7cea` with `key-changes=0`; every miss was classified `unsupported-candidate`, with a 33,570,816-byte six-face footprint and no cache entry or validation attempt. An independent 128x128x6 BC6H cube repeated through ordinal 256 with the same signature, proving the instrument's per-address counter moved. The cache excluded every cube by policy before content validation; this was not an invalidation. | #1737 |
| The `[agc] WaitRegMem … dependency violated` burst at t≈129 s is a finding | **Not a measurement.** The diagnostic is capped at 40 printed lines (`command_processor.cpp:2450`) and the counter in the message is the true total; an unsatisfied wait is documented in the code as normal handled state. The same over-read was recorded independently on Oregon Trail (#1606) and as instrument trap #13 in `GAME_COMPAT_ORCHESTRATION.md`. | #1619, #1606 |
| Vulkan's native typed `B10G11R11_UFLOAT_PACK32` storage conversion is bit-equivalent to the guest's round-to-nearest-even R11G11B10 contract | **Falsified on RADV.** A reduced live-Vulkan 4×4×4 producer→sampled-consumer handoff held tiling (`SW_64KB_R_X`), normalized voxel-centre `SAMPLE_LZ`, dispatch order, cache priming, and all 64 voxels constant while changing only the producer representation. Shader-side `R32ui` packing matched all 64 packed words and 256 sampled channels. Native typed storage differed in **54/64 packed voxels and 92 RGB channel values** (`R=34, G=34, B=24`); all 92 native codes were lower than the CPU f11/f10 oracle, none higher, and the sampled output carried the same 92 differences. The exact packed path now remains authoritative even when native support is advertised. This proves a real conversion defect, **not** that its mostly one-ULP bias causes the visible menu overexposure; the coordinated live A/B in the next row separately tests that causal claim. | #1790 |
| The native R11 storage rounding bias is the primary cause of the profile menu's severe overexposure | **Falsified at the restored profile-menu boot depth on the pre-#1800 branch state recorded in #1790 (the R11 change was rebased unchanged here).** Two bounded default-renderer runs changed only the exact-packed recovery switch and each retained ten five-second samples from t=170–215 s; neither used capture/RTT/resource-hash diagnostics. Source 118's exact arm directly reported the real 32³ tiled R11 LUT as `native-storage=0`, `R32_UINT`, module hash `7e01d14b7e21d0b2`; the same configuration lever's CPU-only reflection A/B produced distinct exact/native-recovery module hashes `d3509f259a1baa99` / `a7bcc1ec7d05db52`, asserted `R32ui` versus typed-float storage, and the live native run independently proved the device's typed 3D storage capability before source 118 executed. At identical rendered frame sequences 15 and 22, exact versus native-recovery images differed by normalized RMSE 0.00920 / 0.00940 (PSNR 40.73 / 40.54 dB), but both showed the same full-width, severely blown-out scene. Exact packing was slightly brighter, not corrective: mean luma was 0.80285 versus 0.80160 at frame 15 and 0.79367 versus 0.79276 at frame 22. This rules out the conversion bias as the **primary visible mechanism at this boot depth**; it does not erase the deterministic precision defect above, and it does not localize the remaining source-101→bloom→source-119 error. The runs are not a performance A/B: exact carried much broader compute tracing, and both stayed near the known ~1.4 composited fps in the sampled window. | #1790 |
| Replay uses a stale/no-op source-101 or source-119 writeback, or source 119 numerically explodes the whole HDR frame | **Falsified for the current captured host-renderer path; this is not a PS5 hardware oracle.** Two bounded `--recompile-raw --dump-post-compute-resource` runs executed the retained mixed-operation prefix and required both descriptor-visible change and the independently recorded live raw hash. Realized compute `62:14` (source 101, operation 566) executed once after 62 earlier successful computes: its captured seed equalled its immediate-before state (`raw=89a341fece902e2e`, `linear=4313b6f00a038092`), then changed to raw `e00d49942f888cad` (**exact live-hash match**) / linear `8aeefa2a7bc15ef9`. Its 2,073,600 R11 texels contained 6,220,800 finite channels, 2,839,736 zero, 1,234,144 greater than one, no Inf/NaN, maximum 12.375, mean 0.683273. Realized compute `80:23` (source 119, operation 584) likewise executed once after 80 earlier successful computes: captured/immediate-before `15af296b7f257522` / `cdbfc7204d266c16` changed to raw `2974a84058d4feb5` (**exact live-hash match**) / linear `7ba80885c5585b55`. Its output had no zero, Inf, or NaN channels, only 2,412 greater than one, maximum 1.03125, mean 0.35676. Thus source 119 strongly compresses the source-101 HDR population instead of causing a frame-wide numeric blow-up. The source-101 distribution itself remains **unverified against PS5 hardware** and may still enter the chain incorrectly; these runs prove the host live/replay boundary and falsify the stale/no-op and whole-output-explosion mechanisms, not visual correctness. | #1790 |
| The source-90/source-92 shader-recompile failures cause missing gameplay pixels | **Falsified.** Both captured packets are direct dispatches with `threads=0x0x0` and `groups=0x0x0`; they are hardware no-ops regardless of whether their shader bodies compile. A temporary diagnostic arm independently moved the lever by admitting both shaders as realized zero-group computes, while the routed gameplay image remained visually unchanged. Their unsupported instructions remain legitimate recompiler coverage work, but cannot repair this frame. | #1627 |

**Still open:** what causes the restored menu layer's substantial overexposure (#1790). Recompiling
source 101 restores the visible 3D layer on the **default** path, but does not make it visually
correct. The degraded gameplay composite is separate and unlocalized (#1627); do not merge the two
hypotheses without capture evidence. The earlier seed-skip/invalidation theory depended on treating
source 119 as an in-place pass over one address; exact resource inspection disproved that premise:
its sampled binding 14 and write-only storage binding 23 are different surfaces, and binding 23 was
already zero before guest packing/writeback.

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

That table records the pre-fix black-layer baseline. The current profile screenshot and live
validation are below.

## Live validation of source-101 recovery — 2026-08-03

A bounded Linux `screenshot` run used the default render path with no RTT, resource-hash, capture or
shader-debug diagnostics. It rendered every submit, suppressed ordinary frame dumps, used a 180 s
capture warmup, then retained nine samples five seconds apart. The first two raw-scanout samples at
t=180/185 s were black; every composited sample from t=190.0 through 222.7 s contained the full-width
3D scene behind the profile UI:

- all seven composited samples contain exactly **2,063,996 non-black pixels** of 2,073,600;
- distinct colours stay between **139,946 and 144,452** while `frame_seq` advances 1 → 43;
- every composited CRC differs, consistent with the scene's animated light/fog rather than a stale
  plateau; and
- the direct t=201.1 s frontend capture is `assets/screenshots/syberia-profile.png`.

This is a visible improvement over the recorded unmodified baseline (670,815 non-black pixels and a
black right ~55%), but not correctness: the restored HDR scene is substantially overexposed. The
title therefore remains rung 3 and the exposure/gameplay-composite work remains open.

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
of the frame and the image is over-dark. It is tracked separately on #1627 and has **not** been
localized; there is no evidence yet that it shares the profile menu's exposure mechanism.

## BC6H cube decode cost — 2026-08-03

A five-second sampling profile on the settled save/profile screen attributed **33.70%** of CPU
samples to `decode_bc6h_block`, in five regular bursts about 1.1 seconds apart. Adjacent live frame
captures contained the same seven BC6H identities and content hashes. The dominant cube is
2048x2048x6: decoding it executes 1,572,864 BC6H block decodes per submit. It has no producer or
future writer in either captured submit.

The live discriminator above then established the mechanism directly. Presentation held near
3.5 fps before the 2048 cube appeared and fell to 2.0 fps afterward. The cube has a stable key and a
33,570,816-byte contiguous span covering the selected level in all six independently-strided face
chains, but the persistent cache rejected it solely because `img_dim=3` was outside its candidate
set.

The cache now admits only supported **block-compressed** cubes and validates that complete six-face
descriptor footprint before reuse. The existing exact-byte comparison, GPU-write journal,
write-watch, ID/version, LRU and memory-budget rules remain unchanged. Non-BC cubes remain excluded:
a broader Float16 cube experiment on Plucky Squire churned the cache and write watches and reduced
performance, so this change deliberately does not revive it.

A single post-fix diagnostic run reached the same profile-screen phase. The large cube produced
exactly one observed miss: `cold-or-evicted`, `candidate=1`, `eligible=1`, with source and footprint
both 33,570,816 bytes. There was no later miss for that address through t=225.1 s / frame 611. This
proved the old unconditional candidate rejection was gone, but not that the identity was referenced
again: aggregate hit counters cover every texture, the detail logger suppresses resources below
0.5 ms, and an approved debugger breakpoint raced process exit. `PROSPER_DETILE_STATS` therefore
gained a separately bounded, one-line-per-address `[detile-hit]` witness.

The follow-up run closed that apparatus gap. Its run-local 2048x2048x6 resource used key
`0x9dc5b36201047cea`; it first emitted one exact 33,570,816-byte cold miss, then an identity-matched
`cache=persistent-hit` with the same key, footprint and source span, persistent ID 195/version 1, and
`validation=exact validated=33570816`. Submit-journal and write-watch queries both returned the clean
result 2. The process was stopped immediately after the witness. The decoded cube is therefore
actually reused; its disappearance from the miss stream was not caused by the texture disappearing
from the workload.

The post-fix frame is visually unchanged in the useful sense: a 1920x1080 composited profile screen
contains the expected full-width scene and UI, with no new cube seam, corruption or missing layer.
It still has the already tracked severe overexposure. The t=225.1 s sample has 177,717 distinct
colours and 2,063,887 non-black pixels. Performance did **not** measurably improve in this diagnostic:
present count rose 419→614 from t=125.057→225.102 s, or 1.949 fps. Do not treat that as a controlled
negative A/B: unlike the pre-fix miss trace, this arm also enabled verbose render timing, and the
scene has other large measured costs. It does establish that removing repeated BC6H misses alone did
not make the title fast under this instrumentation.

The screenshot tool exited 1 after saving 45/48 fresh, pixel-distinct samples because the requested
48×5 s schedule cannot finish inside its 230 s internal timeout; the timeout status was therefore an
apparatus error, not a stale-frame or title-progress failure. All 45 saved samples advanced source and
pixel content with zero stale seconds. The final image is retained only as local evidence; the existing
public profile screenshot remains representative and no compatibility rung changed.

## Initial save-warning atlas cost — 2026-08-03

An input-free automatic F8 capture on exact revision `cb7602b7` isolates the severe slowdown while
the initial animated “do not turn off” billboard is still on screen. The run used the default
full-resolution renderer and selected only stable SPIR-V hash `0xa57c763ae4d70d1d` for phase timing.
Its completeness checks found one accepted selector, one first match, one final matched summary,
1,760 selected phase records, and exactly 160 submits × 11 dispatches. The `.prperf` footer retained
20 pre-trigger samples, 21 post-trigger samples, 20 renderer records and 829 compute records with
zero drops. Both offline reporters exited cleanly.

The selected program constructs a 1920×1620 Float32 atlas as eleven successively smaller rectangles,
from 8,160 workgroups down to four one-workgroup tail dispatches. Every step samples and writes the
same tile-27 guest allocation. The phase clocks — which end before their diagnostic line is printed —
average 5.109 ms setup, 0.010 ms pipeline, 1.626 ms GPU dispatch, 3.798 ms writeback and 0.009 ms
cleanup, or 10.549 ms total. Position means remain flat across the 8,160× change in workgroup count.
The outer F8 interval reports 3.79 presents/s, but that throughput is explicitly
**instrument-perturbed** by one phase line per dispatch and is not an uninstrumented FPS baseline.

Source inspection accounts for the fixed shape. Each step reads back 12,441,600 row-major bytes,
retiles and publishes the complete 12,779,520-byte guest allocation, and thereby invalidates the
sampled cache used by the next step. That is 130.518 MiB of storage readback plus 134.062 MiB of
guest retile per rendered frame before sampled re-upload is counted. The completed storage result is
already the correct partial-write seed; the remaining setup repeats full guest validation,
snapshot/detile and upload for the distinct sampled image.

The resource is not a literal `img_dim=1` descriptor: the live detile census records `dim=5`, depth
1, with no layer stride. That is compatible with the existing base-slice path, but does **not** prove
that either shader reflects an ordinary non-arrayed 2D view. Extending the candidate and its fixture
to that shape still moved no live counter on exact revision `7454407d` (see `## Ruled out`). The next
discriminator must census the exact producer and consumer hashes independently: reflected image
shape and typed-storage choice, device format support, storage cache admission/persistence, and the
sampled borrow result. Until one exact gate is observed failing, the on-GPU copy remains a mechanism
proved only by the production-backend fixture, not an exercised Syberia optimization. Guest
writeback remains synchronous until a separately proven visibility policy can replace it.

## Live validation of gameplay `v_max3_f16` recovery — 2026-08-03

A current-master gameplay capture rejected source 87 at VOP3 opcode `0x354`. `llvm-mc` identifies
the eight exact packets as `v_max3_f16`; the first two select complementary packed-f16 source and
destination halves (`op_sel:[0,0,1,0]` and `[1,1,0,1]`). The recompiler now decodes those selectors,
computes a three-input numeric maximum, applies output modifiers, and preserves the unselected half
of the destination VGPR. Exact packet, structural SPIR-V, and live-Vulkan numeric tests cover the
two-packet pair; disabling only the emitter makes the named structural regression fail.

One bounded default-renderer run followed the validated gameplay route with ordinary frame dumps
disabled and captured exactly one live frame. Source 87 changed from an unrealized 120x68-group
dispatch to realized compute 85 (`shader=3052/cec569003c55c45b`). The operation graph records the
consequential dependency: operation 2057 writes a 960x540 R8 resource and operation 2059/source 89
immediately samples that same address at binding 6. The adjacent current-master frame had 108
realized computes and 25 failures; the fixed live frame had 109 and 23. Total draw/operation counts
differ slightly between animated frames, so the named source and dependency edge—not the aggregate
delta—are the decisive evidence.

The fixed screenshot is visually equivalent to the baseline: the same dark, overexposed ghosted
gameplay composite remains. This closes a real, consumed shader gap but does **not** localize or fix
the visible #1627 defect, and it is not a reason to advance the compatibility rung or replace the
published screenshot.
## The former "right ~55% is black" question — localized and fixed

Before source-101 support, this was **a defect, not art direction, and not a scissor/viewport clip.**
The whole **3D scene layer** of the menu was black; the UI layer (leather panel, boarding passes,
copyright) survived and happened to occupy only the left 45%.

The hypotheses falsified along the way are listed in `## Ruled out` above.

### The pass chain in one pre-fix settled frame

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
operation 570   source 101          -> 0x21159d0000   <-- UNREALIZED: REQUIRED POST-PROCESS INPUT
operation 571   source 102          reads missing input and produces zero bloom/downsample data
...
operation 588   source 119          samples 0x21159d0000 + bloom, writes 0x2110310000 (zero)
operation 589   draw 469            -> 0x2112bd0000   (uniform black)
draw  470       full-screen composite of 0x2112bd0000 -> 0x2117810000 (scanout) — black
draws 472..485  UI: leather panel, boarding passes, copyright — the only visible layer
```

### Where the scene died, exactly

`gpu_replay --through-operation 558` renders `target=0000002110310000` and shows the **complete lit
interior with volumetric light shafts** — 2,066,070 non-black pixels of 2,073,600. The scene is
produced correctly.

The decisive missing link occurs earlier. Operation 570 is source 101, raw program
`0x2111a3af00` (hash `abef7b52fb82e741`), launched as 1920x1088 threads. Its inputs include the
correct lit `0x2110310000` target and an earlier nonzero post-process surface; its storage output is
binding 14/15 at `0x21159d0000`. Recompilation rejects at pc 73:

```
pc 67  v_cmp_ge_f32 s[16:17], s26, v12
pc 69  v_cmp_lt_f32 s[18:19], v12, v13
pc 71  v_cmp_gt_f32 vcc, v11, v14
pc 72  s_and_b64 s[20:21], s[18:19], vcc
pc 73  s_cmp_lg_u64 exec, s[16:17]       <-- first reject
pc 74  s_cbranch_scc0 344
...
pc 96  s_cmp_lg_u64 exec, s[20:21]
```

Operation 571/source 102 immediately consumes binding 14 and produces zero bloom/downsample data.
Operation 588/source 119 (`f32339c9098f6b4a`) then samples binding 14 plus those downstream inputs
and has one storage output, binding 23 at `0x2110310000`. Exact SPIR-V dataflow reaches the sole
`OpImageWrite` from binding 14 and the other samples. Binding 23's raw output is already all-zero
before packing, so guest conversion/writeback is not what creates the black pixels.

The older `PROSPER_RTTLOG=1` trace remains useful as localization for the diagnostic CPU-readback
path:

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

That trace proves content exists before the post-process chain and draw 469 receives black on the
CPU-readback route. Its cache miss is real for that route, but it does not identify the default-path
cause. In particular, source 119 does **not** sample and store the same `0x2110310000` surface: the
old report conflated sampled binding 14 (`0x21159d0000`) with storage binding 23
(`0x2110310000`).

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

The routed notifier-only A/B was later run and left the frame byte-identical; that falsification is
recorded in `## Ruled out` above.

### Earlier CPU-readback framing (historical, not the default-path diagnosis)

> A renderer-owned colour target that a **compute** dispatch reads must be materialized from the
> renderer's own copy before the dispatch runs. Today the graphics pass leaves the pixels only in the
> renderer's persistent Vulkan target / CPU `RttSurf`; a compute dispatch that samples the same
> address reads the guest backing, which prosper never wrote; it therefore writes black, and its own
> storage-write notification then erases the renderer's good copy, so every later consumer is black
> too.

The CPU-readback trace did establish its endpoints, but exact source-119 resources disprove the
in-place premise above: source 119 samples `0x21159d0000` and writes a distinct `0x2110310000`
surface. Keep this section only as history for why the RTT cache was investigated; do not use it as
the current default-path diagnosis.

## Causal and remaining recompiler gaps in the same frame

The retained pre-fix profile capture reports ten dispatches with `reason=shader-recompile`
(`gpu_replay --inspect-only`). Per `CLAUDE.md` these are FATAL gaps, not acceptable skips.
First-reject sites (including gaps since repaired) are:

| program | launch | first reject |
|---|---|---|
| `0x211197ea00` | 1920x1080 | pc=24 fmt=0 op=0xf |
| `0x211197ef00` | 30720x68 | pc=4 fmt=4 op=0x8 |
| `0x21341b1a00` (x3) | 960x544 | pc=55 fmt=9 op=0x354 — now supported as `v_max3_f16` |
| `0x2111a39700` | 1920x1080 | pc=51 fmt=8 op=0xf5 |
| `0x2111a39e00` | 1920x1088 | pc=6 fmt=0 op=0x1 |
| `0x2111a3a400` | 120x72 | pc=36 fmt=4 op=0x8 |
| `0x2111a3af00` | 1920x1088 | pc=73 fmt=3 op=0x13 |

A full boot also logs 13 distinct `[compute] skip unsupported program` addresses. Remember that line
is deduped per `code_addr`: one line means "failed at least once", not "always fails".

The last row is causally connected to the former black scene. Operation 570/source 101 is the only
intended producer of binding 14 at `0x21159d0000` in this chain: it samples the correct lit
`0x2110310000` target, but rejecting pc 73 leaves binding 14 absent/zero. Operation 571/source 102
immediately consumes that missing input, and operation 588/source 119 later consumes binding 14 and
the derived bloom inputs before writing a zero binding 23 for draw 469. Supporting the exact
Wave64 mask comparison recompiles the exact retained source with all 14 captured resources and an
accepted descriptor contract, and the live run restores the layer. The remaining overexposure and
other recompiler gaps remain real but are not localized here.

The `0x354` row is the same instruction family recovered and validated in the routed gameplay
capture above. Its live output is consumed, but the gameplay image remains degraded; do not infer
that every consumed recompiler gap is the visible composite cause.

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
readback RTT path. The former black scene reproduced both with and without them — the pre-fix settled
frame was black in an ordinary `screenshot` run with no capture environment at all — but never
compare a diagnostic run against a default run without accounting for this.
