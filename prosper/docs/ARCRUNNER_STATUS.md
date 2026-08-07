# ArcRunner (`PPSA21406`) — status and evidence

Unreal Engine 4.27-plus. **Rung 0 on a default launch — but with the `PROSPER_SUBMIT_STALL_US=1500`
diagnostic throttle the title runs its whole intro cinematic in 4K *and then reaches and holds its
title screen* (2026-08-06, below). The blocker is a submit-timing race, not the renderer, and
nothing else is now known to stand between this title and rung 2.**

**Read `## 2026-08-07 (arc7)` before anything else about the race.** It is the per-fold account arc6
asked for, and it localises the corruption to the guest's builder thread being released **mid-fold**
by completion writes prosper applies while it is still executing the rest of the same command buffer.
The contract that forbids exactly that already exists in `command_processor.cpp` and is armed by
`if (version >= 13)`; ArcRunner requests SDK version 10. Forcing it on
(`PROSPER_POST_SUBMIT_VISIBILITY=1`, new, default OFF) survives the **default route with no throttle**,
3 of 3, and a 260 s run on that route **reaches and renders the title screen** with 1,977 delivered
video frames and zero faults. The same lever rescues *Crisis Core* (`PPSA07809`), the other title with
this dose-response, which is also SDK 10 — 3 of 3 against a 3-of-3 faulting control. That is a
candidate fix, not a lever; but three rung-6 guarded titles are also pre-13, so the version gate must
not be removed without a cross-title snapshot pass and a review.
The long-lived compatibility index is [#1817](https://github.com/mattias800/prosper/issues/1817),
and the primary allocator, barrier, and intro-movie investigation is
[#1226](https://github.com/mattias800/prosper/issues/1226).

Read `## Ruled out` before forming a hypothesis. The allocator investigation has already separated
the shape of the terminal corruption from its possible authors, and one early combined-arm run was
invalidated by its own incomplete census before the corrected run answered the narrow question.

## Current state

The title boots through UE4 initialisation and its `.pak` load, installs the renderer, submits real
GPU work, and **starts playing its intro movie** — then dies on the #1226 terminal fault after about
19 presented frames, having delivered 43-54 *video* frames. That is 6-17 frames short of the first
frame the movie is not black in, and about 100 frames short of full picture. The earlier
async-compute submit ABI failure is fixed.

**Read `## 2026-08-05: the black composite is the movie, and the movie is black there` before
anything else in this document.** It supersedes the rung-1 framing below in two ways: the composite is
not losing content, **and the terminal fault IS the rung-1 blocker after all.**

**Read `## 2026-08-06: ArcRunner renders real game graphics — the blocker is a TIMING race` FIRST.** It supersedes the rung framing everywhere below: with a diagnostic submit
throttle the title runs its whole intro cinematic in 4K with no fault, so the renderer was never
the blocker.

**Then read `## 2026-08-06: the init-side generation census`** — the last section in this document.
It answers the "blocked on #1756" question the previous handoff left as the concrete next step,
measures the generation overlap and the build→exec content drift at the init, falsifies four
hypotheses including the pend queue and the drift decline itself, and records the first real game
imagery this title has produced **without** the throttle.

**`## 2026-08-06: the submit-duration dose-response` measures that race.** Twelve arms, four doses:
0/3 survive at no stall and at 500 us, 3/3 at 1500 us and at 3000 us, so this title's threshold is
`(500, 1500]` — the same bracket Crisis Core measured, and no longer imported from it. That section
also kills the two cheapest explanations of the lever and records what a default build's own
diagnostics say about which arms die (#2084).

**Then read `## 2026-08-06: all three terminal values are prosper's own two label writes over a LIVE
pointer`.** It settles the *author* of the fault — the three values this document tracked as three
racing causes are one composition rule, and prosper performs both halves of it — and it withdraws the
`## Ruled out` row that excluded prosper's label writes, because the census behind that row is
structurally unable to represent a stomped vtable pointer. It also corrects the rung-1 arithmetic:
the movie advances one frame per guest `sceAvPlayerGetVideoDataEx` call, so a rung-1 arm must be
bounded by **poll count**, not seconds.

**The sibling fault's null register is prosper-manufactured**, and that correction still stands:

- The `addr=(nil)` sibling class is produced by prosper's own reserved-range lazy-commit handler
  zero-filling guest page `0x2100000000` under the guest's *load*, after which the guest dereferences
  that zero on the next instruction. This is [#1944](https://github.com/mattias800/prosper/issues/1944),
  in shared host mapping — not a guest resource-lifetime bug.

The usual terminal chain is:

```text
WORKER-THREAD FAULT: sig=11 addr=0x30016000 rip=eboot+0x127e751
insn @rip: 48 8b 01            (mov rax,[rcx])   rcx=0x30016000
[fault] thread='RenderThread 1' on-guest-TCB=NO(host-%fs leak?)
```

Two apparatus notes on that transcript, both from #2018 and both changing how it should be read.
**The `on-guest-TCB` verdict in it is void**: it was inferred from `rax`, which holds the corrupt
object pointer here rather than a TCB self-pointer (`OREGON_TRAIL_STATUS.md` § Ruled out). The line
is now driven by the verified `%fs` base instead, and the equivalent Crisis Core fault reports
`on-guest-TCB=yes`. **And a pre-#2018 report can be a mixture of two threads' faults** — the
context lived in process globals, so a second faulting thread rewrote them mid-dump; any
`insn @rip`, backtrace or `[faultobj] probe-pages` figure from an arm where more than one thread
faulted may not belong to the header above it.

`0x30016000` is far below the guest arena (`0x2000000000`–`0x9fc0000000`) and is read out of a
structure whose neighbouring fields are ordinary guest pointers. It is the misaligned dereference
of the tagged free-list value `0x2000000001`, not a separate poison value written directly by a
known PM4 path.

The same fault site reproduces under three guest-argument combinations. One important apparatus
detail is that `PROSPER_GUEST_ARGS=-force-gfx-direct` lets the signal kill the process with exit 139
and prints **no** worker-fault report, while empty `PROSPER_GUEST_ARGS`, or `gdb` with `SIGSEGV`
passed through, reports the same fault completely. Use the empty-argument route when fault evidence
is required.

Static import attribution now identifies what the sibling `eboot+0x117811f` path was trying to do.
The guest walks the object references retained by each submitted command-allocation page, reads a
side-command descriptor from `item+0x98`, opens a `sceAgcDcbSetPredication` window, and then passes
the descriptor's `{target, dword count}` to `sceAgcDcbJump` before marking that jump predicated. The
fault is the guest's first dereference of that descriptor, **before** Prosper receives the Jump:

```text
1178064: mov r14, [rax+0x98]   ; retained item -> side-command descriptor
117811a: call SetPredication   ; begin
117811f: mov rcx, [r14]        ; fault: r14 == 0
1178122: mov r8d, [r14+8]
117812d: call DcbJump          ; target=[r14], num_dw=[r14+8]
```

This excludes a silent null-Jump fallback as a legitimate fix. It does not yet say whether the
retained item is stale/corrupted or structurally valid with only its side segment absent.

The static producer/teardown map now makes that distinction testable. The allocation-page builder
gets each resource by calling method `+0x50` on a ref-counted wrapper (for example,
`eboot+0x1197760` returns the wrapper's embedded resource at `wrapper+0x60`). It appends that raw
resource pointer at `eboot+0x1177d90`, `+0x1177dcb`, or `+0x1177dfd` when resource fields `+0x88`,
`+0x68`, or `+0x78` are populated. The page owns only its pointer array: the append takes no wrapper
reference, and `eboot+0x1177ea0` later frees the arrays without releasing a resource.

For all ten mapped normal resource-creation calls, `eboot+0x1186a40` constructs the side command
before publishing the wrapper: the same `+0x68`/`+0x78`/`+0x88` condition that makes the page retain
the resource allocates the 16-byte descriptor, writes its dword count, and stores it at resource
`+0x98`. Conversely, the common resource teardown `eboot+0x1187300` frees that descriptor and writes
`+0x98 = 0`; wrapper destructors call this teardown for their embedded resource. Teardown also
replaces the resource vptr with `eboot+0x7007d60`. Therefore a sibling-fault peek of the retained
item's first qword can discriminate the leading lifetime hypothesis without changing guest
behaviour: `0x7007d60` with null `+0x98` means teardown ran after retention, a live secondary vtable
means construction/later mutation remains open, and any unrelated value means reuse/corruption.
The missing page-side reference is a concrete lifetime seam, not yet proof that external ownership
fails to cover the page's consumption interval.

One ordinary, unsuppressed rendered run on current master (`ce258440`) tried to recover those two
objects with the existing generic fault-memory peek. It was **void for this question**: after about
13 seconds the `AudioMixerRende` worker jumped to null first, so the process exited before reaching
`eboot+0x117811f`. The peek fired correctly but sampled the audio function's unrelated stack layout;
its `rbp-0xa8` and `rbp-0x70` values are not command-item evidence. Immediately beforehand the
barrier diagnostic recorded one D-queue unsatisfied wait at fold time, with both D- and A-queue
events in its retained history and the already-known DCB object `0x2420e48230`. Temporal adjacency
does not attribute the null audio call to that wait. The run had exact zero pre/post process censuses
and terminated naturally with worker-fault exit 90; no suppression or skip was armed.

A second ordinary arm on the exact documentation revision `bdae813d` (runtime code unchanged from
`ce258440`) was also **void for the sibling-lifetime question**, for a different reason. After 13.4
seconds the usual allocator chain won the race first: `RenderThread 1` faulted at
`eboot+0x127e8eb` while dereferencing `0x30016000`, alongside the guest fatal
`FMallocBinned3 Attempt to realloc an unrecognized block 2000000001`. A subsequent null fault was
part of the already-fatal shutdown race. The item/page peek was armed and fired, but neither dump
was at `eboot+0x117811f`, so their stack-relative fields are unrelated and must not be classified as
resource evidence. The arm was bounded to 45 seconds, terminated naturally with exit 90 after 13.4
seconds, had exact zero pre/post process censuses, and used no suppression or skip. This confirms
that ordinary runs can lose to at least two competing terminal paths before the target sibling; it
does not make the sibling absent or weaken the static lifetime seam.

The first address-filtered arm on exact revision `c0964271` was likewise **void**, while validating
the new apparatus far enough to use again. The hardware execute breakpoint armed at
`eboot+0x117811f` on the primary guest boundary and on every logged worker boundary, with the exact
`r14 == 0` gate and a one-record cap. No `[hwbp]` hit or `[hwbp-probe]` record occurred. After about
9.1 seconds the ordinary allocator predecessor won at `eboot+0x127e751`, dereferencing
`0x30016000`; the process terminated naturally with worker-fault exit 90. The arm had exact zero
pre/post process censuses and no suppression or skip. This is not evidence that the sibling is
absent: the instrument was armed, but the target instruction never executed before the competing
terminal path.

The allocator predecessor and descriptor sibling are not merely two arbitrary render-thread
failures. The allocator fault's first guest return address is `eboot+0x1177b4d`, inside the same
allocation-page builder mapped above. Immediately before that return, the builder gets the guest
allocator at `eboot+0x1177b35` and calls its virtual allocation method with size `0x20` and alignment
`0x8` at `eboot+0x1177b4a`. The callee is the `eboot+0x127e6c0` free-list pop whose bad-head
dereference is `eboot+0x127e751`. If that allocation succeeds, the builder proceeds to walk the
command entries and retain their raw resources; the later page consumer is where
`eboot+0x117811f` can dereference a null side-command descriptor. This establishes ordering and a
shared page-build/consume path, not a common corruptor: the allocator damage can still originate
elsewhere, and the null descriptor may be an independent later blocker exposed by suppression.

No screenshot belongs in the compatibility record yet. The latest content-selective capture
retained exactly two non-alpha-only frames: frame 12 was a uniform solid-yellow clear and frame 50
was a uniform solid-white clear. Full inspection found no geometry, text, or game imagery in either.
**Neither is a clear.** Each was presented immediately after one of the run's exactly two
`CB_COLOR_CONTROL.MODE=2` (`ELIMINATE_FAST_CLEAR`) draws, which prosper executes as an ordinary colour
draw ([#1588](https://github.com/mattias800/prosper/issues/1588)) — so they are that gap's artifact
blanketing the target, not content and not a guest clear. See the `## Ruled out` row.

## Rung-1 pass: what the presented frames actually contain

Three bounded runs on `9dcb6c4b`, ordinary and unsuppressed
(`PROSPER_GUEST_ARGS= PROSPER_RENDER=1`), each with exact zero pre/post process
censuses and no suppression, skip, or shim. Timing is deliberately not quoted: peer lanes shared the GPU.

**prosper executes what the guest submits.** With `PROSPER_DRAWLOG=1`, one run realized **456 of 468
submitted draws across 20 submits** — 19 of 20 submits realized every draw, one single-draw submit
realized none. There were **zero** `recompile-reject`, `cfg-recompile-reject`, `exec-recompile-reject`,
compute-skip, or unsupported-format lines in any of the three runs. So the black output is not a
dropped-draw or rejected-shader problem.

**The guest issues a complete but very small UE4 frame, and it composites to the scanout.** With
`PROSPER_COLORSTATETRACE=all`, 256 draw records over 7 frames (~36 draws/frame) resolve to a
recognisable UE4 4.27 post-process frame: one depth prepass, three 4K passes, a 512x512 atlas, **two
complete bloom pyramids** (1920x1080 → 960x540 → 480x270 → 240x135 → 120x68 → 60x34), a 16x2 histogram,
32x32 and 64x64 stages, 1x1 average-luminance targets, and **one final composite per frame into the
double-buffered scanout** (`0x9fc0000000` alternating with `0x9fc2000000`, 3840x2160, `mode=1` NORMAL,
`resolved-cwm=7`). ~36 draws/frame is far too few for gameplay geometry; this is a post-process-dominated
frame over a nearly empty scene.

**No draw in the sample has depth test or depth write enabled.** All 256 records carry `test=0 write=0`.
249 of them bind no depth surface at all (`z=0x0/0x0`, `raw-size=00000000`, extent decoded as `1x1`); the
remaining 7 — one per frame, the `mode=0` prepass into `0x310fea0000` — do bind a real 3840x2160 surface
at `zbase=0x3140f00000` and still resolve to `test=0 write=0`. A UE4 4.27 depth prepass with depth writes
disabled is anomalous. This is recorded as an observation, not yet a diagnosis: most of the 249 are
post-process passes that legitimately have no depth, and whether the prepass register decode or the guest
itself is responsible is unresolved. `CONFIDENCE: LOW` on any causal role in the black output, since
`test=0` rejects nothing.

**Readback localises where the content dies.** `PROSPER_DUMP_RTGROUPS=1` writes a BMP for every rendered
target group with at least one nonzero byte, so a drawn-but-absent file is a fully-zero target. All images
were opened, not just measured:

| target | extent | readback |
| --- | --- | --- |
| `0x315f4f0000` | 1920x1080 | **real content** — a geometric wedge with a soft gradient edge plus ~12 bright specular points; 184 distinct values around grey 175 |
| `0x315e4f0000` | 1920x1080 | uniform `(54,54,54)` with the **same** ~12 specular points at identical positions |
| `0x3157710000` | 3840x2160 | 100% white, same ~12 points inverted as dark dots |
| `0x3003800000` | 512x512 | shadow-atlas-shaped: 884 non-black pixels, thin yellow curve |
| `0x3151af0000` | 32x32 | 722 distinct colours in 1024 pixels |
| `0x3153380000`, `0x3163390000`, `0x31633b0000` | 1x1 | `(255,92,255)` / `(251,92,255)` — **magenta** |
| `0x312bee0000` | 3840x2160 | 100% black RGB |
| both bloom pyramids (`0x3160700000`, `0x3162380000` and every level) | — | **no file at all**: fully zero, alpha included |
| eleven `0x316*` targets | 1792x1080 | 100% black RGB |
| `0x9fc0000000`, `0x9fc2000000` | 3840x2160 | **0 of 8,294,400 non-black pixels** |

**Superseded — see § 2026-08-05.** This table was taken under `PROSPER_DUMP_RTGROUPS`, i.e. on the
CPU-readback path, and the 4K *scene-colour* target it does not list measures entirely zero on the
default live-target path. The 1920x1080 targets below are a different surface from that scene target,
so the two readings do not contradict each other.

So real geometry does reach `0x315f4f0000`; the two bloom pyramids and the 1792x1080 series are entirely
zero; the 1x1 auto-exposure targets hold magenta; and the final composite executes with `cwm=7` and
writes RGB zero. The presented-frame counter agrees exactly — frames go from `nz=0` to alpha-only
`nz=8294400` (RGB 0, alpha 255) and never carry colour except after a `MODE=2` draw.

The instrument is positively controlled from **inside the same run**: the same readback path that reports
zero for the scanout reports a 722-colour 32x32 image and a structured 512x512 atlas, so "black" is a
measurement of the target, not of a broken dump path.

**Superseded — see § 2026-08-05.** The bloom pyramids are zero because the scene is black, and the
1792x1080 series is the movie surface explained there. What follows is retained as the historical
reasoning only.

**The narrowest remaining rung-1 question** is the step between the scene-colour target that has content
(`0x315f4f0000`) and the post-process input that does not (`0x3160700000`, the root of both fully-zero
bloom pyramids), together with the magenta 1x1 auto-exposure value. A garbage average-luminance value is
exactly the shape that makes a UE4 tonemapper map a live scene to black, which is the failure mode the
charter warns about for skipped exposure work — except that here nothing is skipped, so the value itself
is wrong rather than missing.

## 2026-08-05: the black composite is the movie, and the movie is black there

This section **supersedes the rung-1 framing above**. The three findings below were taken on
`c3614f51` with `boot_trace`, ordinary and unsuppressed
(`PROSPER_GUEST_ARGS= PROSPER_RENDER=1`), three runs, exact zero pre/post process
censuses, no suppression, skip, or shim. See [#2011](https://github.com/mattias800/prosper/issues/2011).

**1. ArcRunner plays its intro movie, and prosper's AvPlayer path works.** With `PROSPER_AVPLOG=1`
and `PROSPER_AVP_LOG=1`:

```text
[avp] add source handle=0x1 guest='file://../../../arcrunner/Content/Movies/ArcRunner_Intro_A_1080p_PS5_30fps.mp4'
      host='<DUMP_ROOT>/PPSA21406-app0/arcrunner/content/movies/arcrunner_intro_a_1080p_ps5_30fps.mp4'
      mode=native auto_start=0 duration=63466ms
[avp] guest texture buffers requested=6 allocated=6 align=256 bytes=3317760
[avp-vaapi] software (libavcodec) decode selected for '...' (yuv420p -> NV12)
[avp] video-ex handle=0x1 result=1 data=0x... stop=0          (x43-54 per run)
```

The `file://` URL resolves, the case-corrected host path opens, VA-API declines the profile and the
real libavcodec software fallback takes over (#320's default), and `GetVideoDataEx` returns a rotating
6-buffer ring. The earlier "intro-movie decode" line of #1226 is **closed by observation**.

**2. The staged frames are correct, and they are legitimately black.** `PROSPER_DUMP_RAWTEX=1` dumps
the raw guest bytes of every sampled texture. The movie's luma and chroma planes read:

| plane | extent | bytes | content |
| --- | --- | --- | --- |
| luma | 2048x1080 | 2,211,840 | every valid byte `0x10`; 2,073,600 of 2,211,840 non-zero |
| chroma | 1024x540 | — | every valid byte `0x80` |

`Y=0x10, U=V=0x80` is BT.601 limited-range **black**, and the non-zero count is exactly
`1920 x 1080` — every valid pixel, with the 128-column pitch padding reading zero. So the staging is
byte-correct.

> **Do not re-measure this and read a difference as a regression.** These counts were taken on
> `c3614f51`, when the pitch padding was `memset` to `0`. #2032 changed the padding fill to
> limited-range black (`Y=0x10`, `U=V=0x80`) — because `Y=0, U=V=0` is not black but mid green
> (~`(0, 136, 0)`), which with the coded-extent `width` would have shown as a green right-edge
> stripe. On current master the same dump therefore reads **2,211,840 of 2,211,840** luma bytes
> non-zero, and the padding is no longer distinguishable from the picture's own black. That is the
> intended change, not a defect. The finding this table supports — the staged frame is legitimately
> black — is unaffected either way.

The independent control is the file itself. Decoded on the host with `ffmpeg` (nothing of prosper's in
the path), `arcrunner_intro_a_1080p_ps5_30fps.mp4` has mean luma **0.0 with max 0** at t=0 s and
t=1 s, first content at t=2 s (mean 9.1), and full picture by t=5 s (mean 77.5). **The movie's own
opening two seconds are pure black.**

**3. Every run stops short of the first picture, and it stops on the #1226 terminal fault.** Four runs
delivered 43, 47, 53 and 54 *video* frames and, separately, exactly **19 presented frames** each. At
30 fps those are 1.43-1.80 s of movie, against t=2 s (frame 60) where picture begins — so **6 to 17
frames short**.

**Every one of the four faulted.** The census is `WORKER-THREAD FAULT` occurrences per run: 2, 1, 1, 1
(one run took both the `0x30016000` pop and an `addr=(nil)` sibling). An earlier draft of this section
claimed one run "reached its 120 s timeout without faulting", and that was **wrong** — `rc=124` was the
timeout killing a process that had *already* faulted and wedged, and the fault report is in its log.
There is no evidence here of a stall independent of the fault; do not build on one.

Two numbers matter and they are different. The **first** non-black frame is at t=2 s, and its mean
luma is only 9.1 — barely above black. **Full picture** is t=5 s (mean luma 77.5), which is frame
150, i.e. **96-107 frames** beyond where these runs stop. Rung 1 needs a real visible frame, so quote
the second number when asking how far away rung 1 is; the first only says how close the *onset* is.

**This reverses the standing conclusion that the terminal fault is not the rung-1 blocker.** That
conclusion (recorded in `## Ruled out`) reasoned: frames are already being presented before the fault,
and they are all black, so the fault cannot be what prevents a first frame. The missing premise was
*why* they are black. Now that the movie is known to be black until video frame 60, the runs' 43-54
delivered frames land **before** the picture starts, and the fault is exactly what stops them getting
there. Fixing #1226 does not yield "more black frames" — past frame 60 it yields picture.

**Therefore the black scanout is the correct rendering of a black movie frame, and rung 1 is blocked
by the #1226 terminal fault killing the run before the movie reaches picture — not by the renderer.**

## The movie surface extent, measured before and after the `AvPlayerVideoEx` fix

This is the measurement the "Next discriminators" list used to ask the next agent to perform. **It is
done** — the numbers are here so nobody re-runs it (#2032).

The guest sizes its movie luma T# from the published **pitch** (`2048x1080`, chroma `1024x540` —
exactly half) and then renders the converted frame into a target sized by `width - crop_left -
crop_right`. Under the pre-fix contract that published the visible 1920 alongside
`crop_right_offset=128`, that expression double-counted the padding and yielded **1792**.

Ordinary unsuppressed `boot_trace`
(`PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_COLORSTATETRACE=all`), exact zero
pre/post process censuses, no suppression, skip or shim. Movie surfaces counted by their
`raw-format=10 resolved-format=50` signature:

| arm | `1792x1080` movie surfaces | `1920x1080` movie surfaces |
| --- | --- | --- |
| before (`width` = visible 1920) | 17 distinct | **0** |
| after (`width` = coded 2048) | **0** | 9 distinct |

The two arms ran for different lengths, so the raw counts are not comparable to each other; the
discriminating fact is the **category flip**, which is absolute in both directions. The before arm
found zero `1920x1080` surfaces of that signature *despite other 1920x1080 targets existing in the
run*, so the signature does isolate the movie surfaces. **ArcRunner therefore uses the `width`-based
spelling**, and the coded-extent contract is the right reading.

**No visible change follows from this on ArcRunner**, and that is expected: the movie is black in the
window the guest reaches (above), so the surfaces are now the right size and still black. The value of
the fix here is that it removes a real 128-column loss that would otherwise appear the moment #1226 is
fixed and the movie reaches picture.

**Cross-title control — the same change moves nothing on R-Type Delta** (`PPSA26414`), the other live
`Ex` consumer, whose 1920-wide movie also discriminates because 1920 is not 256-aligned. Measured with
`PROSPER_AVPCHROMA_LOG=1` on the deterministic `tools/dropcache.py` route, before and after the
one-line `width` change and nothing else:

| arm | luma T# | chroma T# | chroma verdict |
| --- | --- | --- | --- |
| before (`width` = visible 1920) | `2048x1080` | `1024x540` | `CHROMA matched-registered-pitch` |
| after (`width` = coded 2048) | `2048x1080` | `1024x540` | `CHROMA matched-registered-pitch` |

Identical, down to the guest allocation addresses. R-Type derives **both** plane descriptors from
`pitch`, never from `width`, so the concern that a chroma plane sized `width/2` would move 960 → 1024
and flip the resource layer's narrow-texture path selection is **falsified**: the chroma extent is
1024 in both arms and was never 960. #2005's investigation is not disturbed by this commit.

## Reproduction

Build and run inside the `ps5ys` distrobox, with `TMPDIR` on disk rather than `/tmp`:

```bash
cd prosper
PROSPER_GUEST_ARGS= PROSPER_RENDER=1 \
  ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0
```

The rung-1 census arms above are, in order, the realization check, the per-draw colour/depth state
census, and the rendered-target content readback. Write every artifact under `~/`, never `/tmp` or
`/var/tmp`:

```bash
# 1. submitted-vs-realized draws, plus any recompile/skip lines
PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_DRAWLOG=1 \
  ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0

# 2. one raw-to-resolved colour/depth record per draw (runs BEFORE the no-effect fast path,
#    so a dropped draw is still visible)
#    PROSPER_PRESENT_NZLOG needs a readback companion or it prints NOTHING -- see the
#    instrument note in the 2026-08-06 section. Alone it is a void arm, not a black frame.
PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_COLORSTATETRACE=all \
PROSPER_PRESENT_NZLOG=1 PROSPER_DUMP_RTGROUPS=1 PROSPER_FRAME_DIR=~/arc-work/rtt \
  ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0

# 3. a BMP per rendered target group with >=1 nonzero byte; a DRAWN target with NO file is
#    fully zero. ~1 GB per 14 s -- keep the run short and delete afterwards.
PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_DUMP_RTGROUPS=1 \
PROSPER_FRAME_DIR=~/arc-work/rtt ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0
```

The combined diagnostic arm used for the most recent narrow experiment is:

```bash
PROSPER_GUEST_ARGS= PROSPER_RENDER=1 \
PROSPER_INIT_SUPPRESS=ptr PROSPER_REL1_FORGE_SUPPRESS_ALL=1 \
PROSPER_FORGE_TRIP=1 PROSPER_PRESENT_NZLOG=1 PROSPER_DUMP_RTGROUPS=1 \
  ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0   # NZLOG needs the readback companion
```

`PROSPER_REL1_FORGE_SUPPRESS_ALL` is a default-off diagnostic, not a title fix. It suppresses every
exact REL1 `forges_freelist_ptr` candidate, including live paired fences. A valid run must prove the
lever with an exact terminal `FORGE-DECISION-TOTALS` census where `candidates == suppressed` and
`landed == 0`; malformed values print `NOT ARMED`.

## Combined init-plus-forge suppression experiment

The first arm, on `dfd89f3f`, was **provisional and void for causality**. Init suppression reached
sparse ordinal 4,096 and forge detail reached at least ordinal 64, but the only totals snapshot was
candidate 1. The worker-fault path calls `_exit(90)`, so no `atexit` summary can run, and the original
first-plus-every-256 reporting cadence could not prove that every later candidate was suppressed.

The corrected diagnostic reports every candidate through ordinal 256 and then every 256th. Its
schedule is mutation-tested: ordinals 1, 64, 127, and 256 report, 257 is quiet, and 512 reports. A
second bounded title run on `fb3daaa4` independently reached `INIT-SUPPRESS #1024` and ended with the
exact final line:

```text
FORGE-DECISION-TOTALS candidates=20 suppressed=20 landed=0 mode=all
```

That run presented frames 0 through 52, did not hit the `0x30016000` terminal chain, and instead
faulted at the other known sibling site:

```text
WORKER-THREAD FAULT: sig=11 addr=(nil) rip=eboot+0x117811f
r14=0 r15=0x2420e48230
```

The result settles only the intervention's narrow necessity question: the specific
`0x2000000001 -> 0x30016000` terminal chain does **not** survive when neither known label write
lands. Suppressing both writes is not a title fix. The all-mode arm deliberately drops live fences,
so it neither isolates the init write from the fence write nor attributes the sibling null-object
fault.

Durable run records are in the #1226 comments for the
[CPU seam](https://github.com/mattias800/prosper/issues/1226#issuecomment-5164678099),
[provisional first arm](https://github.com/mattias800/prosper/issues/1226#issuecomment-5164771700),
[census correction](https://github.com/mattias800/prosper/issues/1226#issuecomment-5164799214), and
[valid corrected arm](https://github.com/mattias800/prosper/issues/1226#issuecomment-5164904408).

## What the `addr=(nil)` fault actually dereferences

Measured on current master (`c9e2588e`), ordinary and unsuppressed
(`PROSPER_GUEST_ARGS= PROSPER_RENDER=1`), with the new `PROSPER_LAZY_COMMIT_STRICT`
discriminator. **Three terminal paths compete for the same run**, so a single run per arm cannot A/B
anything on this title — see the `## Ruled out` row.

The `addr=(nil)` class is a **masked wild read of one corrupt table entry**, and the guest instruction
pair is now named exactly. Disassembly of the eboot (`tools/re/edis.py` over a `prx_to_elf.py`
flattening) at the site the lazy-commit line reports:

```text
117e201:  mov  eax, DWORD PTR [r13+0x0]     ; a packed command dword; r13 walks the stream by 4
117e20b:  shr  ecx, 0x18                    ; top byte is a tag, compared against r15d
117e218:  bextr ecx, eax, 0x1008            ; bits 8..23 -> a table index
117e21d:  mov  rdi, QWORD PTR [r12+rcx*8]   ; rdi := table[index]         (r12 = 0x2020e381c0)
117e221:  mov  rcx, QWORD PTR [rdi+0x40]    ; <- lazy-commit fires here; rdi = 0x2100000001
117e225:  vucomisd xmm0, QWORD PTR [rcx]    ; <- the reported `addr=(nil)`: rcx is OUR zero
```

With strict mode the fault reports where it belongs, once per affected run:

```text
[lazy-commit] #1 DECLINED(strict) page=0x2100000000 addr=0x2100000041 access=read rip=0x41117e221
WORKER-THREAD FAULT: sig=11 addr=0x2100000041 rip=0x41117e221 (image+0x117e221)
```

`0x2100000041` is `rdi + 0x40` with the low bit **unmasked**, so `rdi` is used as a plain pointer:
`0x2100000001` is garbage, not a tagged pointer. Two consequences, both of which retire earlier
framings:

- The repeated page is **not** evidence of a lost or aliased commit. Every `0x21000000xx` pointer
  lands in page `0x2100000000`, so "two different guest sites first-touch the same page" is a
  property of the *value*, not of the mapping. See the `## Ruled out` row and [#1944](https://github.com/mattias800/prosper/issues/1944).
- The three recorded terminal faults all carry one value shape — `{nonzero high dword, low
  dword ≤ 1}`: `0x2000000001` (the allocator chain), `0x2100000001` (`rdi` here), and `0x400000001`
  (`rax` at the `AudioMixerRende` `rip=0x0` jump). Only the first is attributable to prosper's
  init+fence composition; see the next section for the measurement that excluded the other two.

## The forge census was measuring its own predicate — and now does not

`forges_freelist_ptr()` gates on `ptr_like()`, whose window is `[0x1000000000,0x1200000000)` ∪
`[0x2000000000,0x2100000000)`. ArcRunner's arena is `[0x2000000000,0xa000000000)` (live:
`reserve ENTRY hint=0x1000000000 len=0x8000000000` → `reserve -> 0x2000000000`), so the guard window
covers **4 GiB of 512 GiB**, and the terminal fault's `0x2100000001` sits exactly one byte above its
upper bound. The INIT-side census already used the wide predicate, so the two sides were never
comparable.

The tripwire is now split into a *decision* predicate (unchanged; what the default guards use) and a
*report* predicate over the whole guest-VA window, with `window=narrow|wide-only` on every
`FORGE-STOMP` line and a running `FORGE-TRIP-TOTALS`. That makes the following a **real negative**
rather than an unobtainable one — one bounded run on `c9e2588e` + this change,
`PROSPER_FORGE_TRIP=1`:

```text
FORGE-TRIP-TOTALS seen=256 narrow=256 wide_only=0
```

Read that as exactly what it says: **none of the 256 reported candidates** is wide-only, and every
one has `pre=0x2000000000`. The totals ride a dense every-256 schedule, so the *absence* of a `#512`
line additionally bounds the run's whole population below 512 — the claim is over a counted
population, not over a sparse sample of it.

That covers only the forge branch (`pre`'s low dword **zero**). The sibling branch — `REL1-LIVE`, a
fence landing on a pre whose low dword is a **real pointer half**, which would turn a live
`0x2100e05140` into exactly `0x2100000001` — is gated on the same stale `ptr_like()` and was equally
unmeasurable. `PROSPER_PTRLIKE_WIDE=1` arms **both** guards over the wide window, and announces
itself so the arm cannot be confused with a no-op:

```text
[agc] PTRLIKE-WIDE ARMED: guard window [0x1000000000,0xa000000000) (narrow default was …)
```

Two bounded runs with the lever witnessed: `SUSPECT-REL1-LIVE` count **0**, `FORGE-TRIP-TOTALS
seen=256 narrow=256 wide_only=0`, terminal fault unchanged. **So prosper's ReleaseMem/WriteData label
writes are excluded as the author of `0x2100000001` on both branches.** The blind spot is
structurally real and worth removing so the question can be asked at all, but it is empty on this
title. Caveat on the strength of the `REL1-LIVE` zero: `report_suspect_write()` has emitted **no**
line of any kind on ArcRunner across nine runs, so its zero carries no in-run positive control — it
is consistent with the observed REL1 population being homogeneous (`pre=0x2000000000` on every
candidate), not independently proven. The lever is default OFF because `rel1_stomp_guard()` is
default ON.

## 2026-08-06: ArcRunner renders real game graphics — the blocker is a TIMING race

**Rung 1 is demonstrated, under a diagnostic throttle.** With `PROSPER_SUBMIT_STALL_US=1500` (a
`nanosleep` in the submit fold — documented in `hle_agc.cpp` as *"Never a fix — it throttles the guest
unconditionally"*), ArcRunner runs its full intro cinematic with real 4K content and no fault at all.

| arm | terminal fault | video frames delivered | presented content |
| --- | --- | --- | --- |
| default | **17 of 17 runs faulted** at 8–14 s | 31–54 | RGB 0 |
| `PROSPER_SUBMIT_STALL_US=1500` | **0 of 2** — both ran to the 120 s timeout | **1,901 of 1,908 succeeded** | ~26.4 M of 33.2 M bytes nonzero at 3840x2160 |
| `PROSPER_SUBMIT_STALL_US=3000` | **0 of 2** — both ran to the timeout | ~1,905 | — |

Zero `WORKER-THREAD FAULT`, zero `FMallocBinned3` fatals, zero canary aborts across the stalled runs.

Frames were **opened, not merely measured** — `assets/screenshots/arcrunner-intro-space-station.png`
(a nebula, the ringed station, engine glow, and the legible caption *TITAN-CLASS SPACE STATION "THE
ARC"*) and `assets/screenshots/arcrunner-intro-city.png` (a rainy neon street, a character in
silhouette, signage, wet-ground reflections). A third retained frame is a mid-typewriter text card
reading *POPULATION: 10 MIL*. All three carry the game's own *PRESS ANY BUTTON TO SKIP* prompt.

**Reproduce it.** The headline arm, and the frame capture that produced the screenshots:

```bash
# the arm itself -- no fault, ~1,900 delivered video frames; give it a LONG bound
PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_SUBMIT_STALL_US=1500 PROSPER_AVPLOG=1 \
  timeout 120 ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0

# the same arm, writing BMPs -- frames 100 and 236 are the two committed screenshots
PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_SUBMIT_STALL_US=1500 PROSPER_AVPLOG=1 \
PROSPER_FRAME_DIR=~/arc-work/frames PROSPER_FRAME_DUMP_EVERY=25 PROSPER_DUMP_CONTENT=200000 \
  timeout 100 ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0
```

`rc=124` (timeout) is the SUCCESS condition here — it means the process was still alive. `rc=90` is the
worker-fault exit. Count `video-ex` lines, not seconds, and pass `PROSPER_AVPLOG` (not
`PROSPER_AVP_LOG`, which nothing reads — see the `## Ruled out` row).

**What this settles.** The renderer, the recompiler, the AvPlayer path, the resource layer and the
composite were never the blocker: they produce a correct 4K cinematic the moment the guest is not
racing prosper. Everything in the sections below about *what* corrupts the pointer remains accurate
and is now framed correctly — the label writes compose the fault **value**, but the **race** is what
decides whether corruption happens at all.

**This reproduces master's Crisis Core result on a second title.** The `## Ruled out` row added by
#1945/#1894 measured a dose-response over the same lever on `PPSA07809` — 0/3 survive at no stall,
0/3 at 500 us, 3/3 at 1500 us, 3/3 at 3000 us — and found pend-queue residency peaked at 3 ms, so
"our completion writes land late" is false there too. ArcRunner gives the same lever the same **answer**, at the
only two doses tried here (1500 us and 3000 us, both 2/2). **Its threshold was NOT measured when this
section was written** — with no 500 us arm the bracket was only `(0, 1500]`, and the tighter
`(500, 1500]` figure was Crisis Core's, imported rather than reproduced. **That gap is now closed: the
full twelve-arm dose-response was run on this title (#2084) and its own bracket is `(500, 1500]`, the
same one. See § *2026-08-06: the submit-duration dose-response*.** It also agrees with this document's own
`PROSPER_EOP_WRITE_SYNC` null: moving *when* our writes land changes nothing, because the gap that
matters is inside the guest's own build-to-submit interval.

**What is still owed for a real rung 1.** The throttle is not a fix and must not be presented as one —
it slows every submit unconditionally. The honest statement is: *the title's graphics are complete and
correct; the default route still faults.* Closing that gap means making prosper's submit fold behave
correctly without the artificial delay, which is now the whole of #1226 and #1945. The next lane
should start from the dose-response threshold — **measured on this title as `(500, 1500]`**, see the
following section — and ask what the guest observes during that window, not what value ends up in the
pointer. One candidate answer is already eliminated there: honouring the guest's `WAIT_REG_MEM`
barriers (`PROSPER_WAIT_DEFER=1`) does not rescue the title, 3 of 3 with the lever witnessed. A
second — that the throttle shortens the guest's build-to-submit interval — is **unmeasured rather
than eliminated**: the sample that appeared to settle it was selected for the condition under test.

**Two defects visible in the frames, both new and neither blocking rung 1:**

1. The cinematic has a heavy **green/magenta cast** — the nebula is pure green, the city is magenta.
   Green-vs-magenta is the classic chroma-plane signature (a U/V swap or a plane-order error), and the
   movie is decoded through the software libavcodec path (`yuv420p -> NV12`). Filed separately.
2. The default-route composite question in the rung-1 section below is answered: it was never losing
   content. It composited a black movie frame correctly, exactly as that section concluded.

## 2026-08-06: the submit-duration dose-response — the threshold transfers exactly

The section above left this title's threshold bracketed only at `(0, 1500]` and imported the tighter
`(500, 1500]` figure from *Crisis Core*. **The 500 us arm has now been run here, and ArcRunner's own
bracket is `(500, 1500]` — the same one.** [#2084](https://github.com/mattias800/prosper/issues/2084).

Twelve arms, four doses, three arms per dose, **one binary** (built from `f080fc23`; the same
`sha256` recorded in every arm's log header), ArcRunner's own `boot_trace` route, a 120 s bound, and
one endpoint scored the same way for every arm. Two design points beyond the method in #2084. The
doses were **interleaved** — 0, 500, 1500, 3000, repeated three times — rather than run in per-dose
blocks, so drift in machine load cannot align with dose. And **no passive observer was armed**:
instrument trap 104 records that a read-only probe's *duration* can decide whether this failure
happens, so the only diagnostics in the path are the ones a default build prints unconditionally.
Each arm's log header records a `pgrep` census of peer `boot_trace`/`prosper-app`/`screenshot`
processes taken immediately before and after it; every one of the twenty-four censuses is zero, so
no peer lane shared the GPU during the matrix.

**Endpoint, fixed before the first arm ran.** SURVIVED = the process was still alive at the bound
(`rc=124`) **and** the log carries zero `WORKER-THREAD FAULT` lines **and** zero
`FMallocBinned3`/`LowLevelFatalError` fatals. `rc=124` on its own is *not* survival — the
2026-08-05 section records an arm where the timeout killed a process that had already faulted and
wedged, and an earlier draft of that section read it as a clean run.

| stall per submit | survived the bound | died |
| --- | --- | --- |
| none (default) | **0 / 3** | 3 / 3 |
| 500 us | **0 / 3** | 3 / 3 |
| 1500 us | **3 / 3** | 0 / 3 |
| 3000 us | **3 / 3** | 0 / 3 |

**0 of 6 below the threshold, 6 of 6 above** (Fisher exact, two-tailed, p ≈ 0.002), monotone, with
the threshold between 500 us and 1500 us. Per arm:

| dose | rep | rc | wall | video frames | verdict | terminal fault | faulting thread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 90 | 17.3 s | 138 | DIED | `addr=0x30016000 rip=eboot+0x127e8eb` | `RHIThread` |
| 0 | 2 | 90 | 14.0 s | 38 | DIED | `addr=0x30016000 rip=eboot+0x127e751` | `RenderThread 1` |
| 0 | 3 | 90 | 10.9 s | 36 | DIED | `addr=(nil) rip=0x0` | `AudioMixerRende` |
| 500 | 1 | 90 | 17.9 s | 31 | DIED | `addr=0x30016000 rip=eboot+0x127e8eb` | `RHIThread` |
| 500 | 2 | 90 | 14.2 s | 36 | DIED | `addr=(nil) rip=0x0` | `AudioMixerRende` |
| 500 | 3 | 90 | 11.6 s | 37 | DIED | `addr=0x30016000 rip=eboot+0x127e8eb` | `RHIThread` |
| 1500 | 1 | 124 | 123 s | 1,906 | SURVIVED | none | — |
| 1500 | 2 | 124 | 133 s | 1,986 | SURVIVED | none | — |
| 1500 | 3 | 124 | 121 s | 1,988 | SURVIVED | none | — |
| 3000 | 1 | 124 | 124 s | 1,986 | SURVIVED | none | — |
| 3000 | 2 | 124 | 122 s | 1,983 | SURVIVED | none | — |
| 3000 | 3 | 124 | 121 s | 1,924 | SURVIVED | none | — |

Every fault is in the already-recorded family — four in the `0x30016000` allocator-pop class, two in
the `AudioMixerRende` `rip=0x0` class — and no arm produced a new one.

**Read the transfer claim narrowly.** What replicates is that *this title's* #1945 is decided by
submit duration, at the same bracket. It is still not a general law, and #2084 says why the temptation
to make it one is the error to avoid: one env var makes the arm cheap. The right reading is that two
independently-brought-up UE4 titles put the boundary in the same 500–1500 us window, which makes it a
property of prosper's submit timing rather than of either guest — but a third title still owes its own
twelve arms. `CRISIS_CORE_STATUS.md` states the same limit from the other side, and is the doc to read
for that title's arms.

### What the arms say beyond the survival table

All four observations below come from diagnostics that print **unconditionally** in a default build,
so they cost the arms nothing and cannot have confounded the timing.

**1. The `SUSPECT-REL1-OVERLAP` tripwire selects the allocator class exactly, and is empty in every
surviving arm.** The tripwire (`command_processor.cpp`, `label_rel_overlap`) fires when prosper
executes a `RELEASE_MEM` fence for a label while the guest has already **built** a newer init for the
same label that prosper has not executed — two fence generations in flight together, which is the
condition under which the guest can free the block on the first `1` and prosper's later pair then
lands in it. Its ordinals partition the matrix cleanly:

| arm class | `SUSPECT-REL1-OVERLAP` population |
| --- | --- |
| died at `0x30016000` (4 arms) | 11, 22, 27, 24 |
| died at `rip=0x0` (2 arms) | 0, 0 |
| survived (6 arms) | 0, 0, 0, 0, 0, 0 |

The counts are exact, not capped: `diag_should_print` prints the first 64 ordinals and every power of
two after, and the largest ordinal seen is 27. **The zero is positively controlled from inside the
same runs** — the counters it differences (`dma_built_n`, `dma_exec_n`) are demonstrably live in the
surviving arms, whose `WaitRegMem` reports carry 28–40 populated label event rings with both `dmaB`
and `dmaX` events. So this is "armed and nothing matched", not "never armed".

**And the reporter is not reachable-but-bypassed either**, which is the sharper form of the same
worry: the overlap check sits at the *end* of `case 1:` of `rel_data_sel`, behind seven earlier
`return`s, and a zero would be worthless if the surviving arms had simply been exiting sooner. All
seven are accounted for on a default build:

| early return | why it cannot explain the zero |
| --- | --- |
| `!c.rel_value_valid` | a short-decoded packet. `report_short_fold` is unconditional and reports **0** in all 15 arms — nothing was truncated. |
| `mb3_suppress_release` | needs `PROSPER_MB3_FREELIST_GUARD`, default OFF, not armed |
| `stale_release_generation` | needs `PROSPER_GENERATION_GUARD`, default OFF, not armed |
| `waf_guard` | `PROSPER_REL1_WAF_GUARD`, default OFF, not armed |
| `REL1-LIVE` branch | **reports before it returns**; zero `REL1-LIVE` lines in all 15 arms |
| `rel1_forge_suppress_candidate` | `rel1_forge_decision_mode()` is 0 unless armed, so it returns false |
| `REL1-FORGE` branch | **reports before it returns**; zero `REL1-FORGE` lines in all 15 arms |

Two of the seven announce themselves and did not fire; four are default-OFF levers nobody armed; the
one genuinely silent path is contradicted by an unconditional counter.

**And the positive form closes it outright, which is the evidence to quote rather than the table
above.** The success path records `LE_REL_EXEC` at `label_hist_rel_exec` — *downstream* of the overlap
check — while the three suppression branches record theirs upstream at their own `return`s. So a
`relX` event in a label ring is a witness that control passed the tripwire. The surviving arms carry
**53, 74, 82, 94, 109 and 119** of them. With zero `REL1-LIVE` and zero `REL1-FORGE` lines (both
branches report before returning) and `PROSPER_NONHEAP_PTR_GUARD` unarmed, every one of those `relX`
events was recorded on the success path. The tripwire was reached and returned zero.

It is also a much sharper statement than the survival table alone: the tripwire is a **selector for one
of the two terminal classes**, present in 4 of 4 allocator-pop arms and absent from 2 of 2 null-jump
arms — while the throttle removes *both*.

**2. No evidence that the throttle shortens the guest's build-to-submit gap — and the obvious way to
measure it is confounded.** Pairing each label ring's last `dmaB` (the guest's own AGC builder call)
with the following `dmaX` (prosper executing that packet at fold time) gives the interval the
2026-08-06 sections identify as the exposure window. **Do not take that sample over all ring-bearing
lines**, which is how the first revision of this section did it and got a much stronger claim than the
data supports. Rings are only visible inside diagnostics that embed `label_hist_report`, and the two
arm classes do not have the same ones: a dying arm carries 1–4 `WaitRegMem` reports and 11–27
`SUSPECT-REL1-OVERLAP` reports, a surviving arm carries 28–40 `WaitRegMem` reports and **zero**
overlap reports. Pooling them draws the dying arms' pairs almost entirely from a population that
exists *only* when a second generation is in flight — i.e. selected for the condition under test.

Restricted to the one population both classes have, the `WaitRegMem` rings:

| | median build→exec gap | pairs |
| --- | --- | --- |
| dying arms (6) | 282, 350, 352, 364, 409, 413 ms | 2–8 each, **29 total** |
| surviving arms (6) | 347, 400, 424, 435, 437, 584 ms | 38–80 each, **363 total** |

The surviving arms are certainly not *shorter*, and if anything longer — but at 2–8 pairs per dying
arm this supports **no** falsification, only the absence of evidence for the shortening story. The
strong form of this claim ("the throttle lengthens the interval") is **withdrawn**; it was an artifact
of the pooled sample. `CONFIDENCE: LOW`, and it needs a per-fold instrument rather than whatever
rings a diagnostic happens to print.

**3. Unsatisfied `WAIT_REG_MEM` barriers are not the discriminator.** They occur in every arm,
including all six surviving ones, with recorded packet ages from 3 ms to 445 ms. Their line count is
print-capped at 40 (`ln < 40 || (ln & 1023) == 0`), so it is a ceiling and not a rate. Four of the six
surviving arms sit exactly on it (40) and two are below (36, 28), so only the four are censored; in
no case is the number a frequency.

**4. prosper's deferred-stream barrier model is not in the default path at all.** `PROSPER_WAIT_DEFER`
is **opt-in and default OFF** (`command_processor.cpp`), so on the default route prosper barrels
through an unsatisfied wait with the "dependency violated" log rather than pausing the stream. The
count of `dependency violated` equals the count of unsatisfied waits in all twelve arms, which
confirms it directly. This matters because the ~250–580 ms / 3–7 fold build-to-exec gap in
observation 2 is therefore **entirely the guest's own build-ahead**, with no prosper deferral in it —
do not reason about that gap as though prosper were holding the packet.

### The obvious non-throttle lever was run, and it does not work

Since observation 4 says prosper barrels through the guest's own barriers on the default route, the
natural candidate fix is to stop doing that. **`PROSPER_WAIT_DEFER=1` was run here — three arms, same
binary, same route, no throttle — and it does not rescue the title.** 3 of 3 faulted, at 11.4 s,
13.6 s and 11.4 s, delivering 61, 34 and 35 video frames; all three took the `addr=(nil) rip=0x0`
`AudioMixerRende` class, and `SUSPECT-REL1-OVERLAP` still reached ordinals 25 and 22.

The lever is witnessed, not assumed: every defer arm carries
`WaitRegMem #0 … — pausing queue (deferred effects)` where a default arm carries `dependency
violated`, one arm additionally logs `WaitRegMem DEFER TIMEOUT #1 after 1000ms`, and the default arms
carry the marker `pausing queue (deferred effects)` **0 times against 3 of 3** here. (Use that exact
marker, not a bare `defer` grep: the surviving dose arms each contain one unrelated
`layered image deferred to #657` line, so a loose grep does not separate the arms.) So the model was
on, the barriers were honoured, and the title
still died — sooner, on median, than the unthrottled control arms. That reproduces on ArcRunner the
verdict `command_processor.cpp` already records from ~20 instrumented DOLL runs: honouring the
barriers removes the ordering-violation leg and leaves a second, wait-order-independent leg that
dominates, while deferral latency shifts the guest into it earlier.

One incidental corroboration worth keeping: in `defer` rep 3 the label the guest is waiting on reads
`0x1700f1e8` — the low dword of `0x41700f1e8`, the module-image vtable named in the section below. The
guest was polling a "label" that had become a C++ object.

### One correction to this document's own census, which cost nothing to find

Earlier sections record **zero** `recompile-reject` / compute-skip / unsupported-format lines across
20+ runs, and read that as this title's shader coverage being complete. It is not: **every one of
those censuses was taken on a run that died at 8–14 s.** The six surviving arms here each reach a
`[compute] skip unsupported program` line for **three programs reproducibly** (`0x3005330000`,
`0x3007780000`, `0x30094d0000`), plus a run-varying fourth, plus one
`layered image deferred to #657 -> dispatch skipped (#590)`. All of them are past the point where a
default run dies. The clean census is a fact about the first fourteen seconds, not about ArcRunner.
Filed as [#2090](https://github.com/mattias800/prosper/issues/2090). Generalise it before quoting any
"zero X across N runs" figure on a title whose runs are terminated by a fault: **a coverage census is
bounded by how far the run got.**

### Reproduction

```bash
# per arm, on ArcRunner's own established route -- ONE binary for all twelve.
# No PROSPER_GUEST_FS here on purpose: it is read ONLY on macOS (`guest_tls.cpp:46`); Linux and
# Windows take the `:58` branch, where guest TLS is default-ON and the variable that exists is
# the opt-OUT `PROSPER_NO_GUEST_FS`. The arms did set it and it did nothing. #2095.
PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_PROGRESS=20 PROSPER_AVPLOG=1 \
  PROSPER_SUBMIT_STALL_US=<unset|500|1500|3000> \
  timeout 120 ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0

# score (the same expression as the endpoint above): rc=124 AND zero matches for
#   grep -cE 'WORKER-THREAD FAULT'      AND zero for
#   grep -cE 'unrecognized block|LowLevelFatalError'
```

The control arm leaves `PROSPER_SUBMIT_STALL_US` **unset** rather than setting it to `0`; both are
inert (`submit_stall()` returns early on 0), but unset is the default route exactly.

## 2026-08-06: all three terminal values are prosper's own two label writes over a LIVE pointer

Measured on `66eaf77b` plus the two instruments this section introduces, nine ordinary unsuppressed
`boot_trace` runs (`PROSPER_GUEST_ARGS= PROSPER_RENDER=1`), exact zero pre/post process censuses, no
suppression, skip or shim. See [#1226](https://github.com/mattias800/prosper/issues/1226).

**The `{small nonzero high dword, low dword 1}` shape is not three racing causes. It is one
composition rule, and prosper performs both halves of it.** The section above had this for
`0x2000000001` as an inference from `pre=0x2000000000`; the label ring at the *faulting register*
now shows the whole sequence, for three different faults on three different threads:

| faulting thread | the pointer that was there | after our 4-byte init | after our 4-byte fence |
| --- | --- | --- | --- |
| `AudioMixerRende` | `0x41700f1e8` — a **vtable pointer**, `eboot+0x700f1e8` | `0x400000000` | **`0x400000001`** |
| `RHIThread` | `0x21c1388890` — a live heap pointer | `0x2100000000` | `0x2100000001` |
| (`rdi` at fault) | `0x21c0e182d0` — a live heap pointer | `0x2100000000` | **`0x2100000001`** |

The `AudioMixerRende` row is the complete chain end to end. The guest instruction is named by
disassembly (`tools/re/edis.py` over a `prx_to_elf.py` flattening):

```text
32b61bb:  mov  rax, QWORD PTR [rdi]     ; rax := the object's VPTR
32b61be:  call QWORD PTR [rax+0x20]     ; <- jumps to 0; rip=0x0, return address 32b61c1
```

and the label ring for that exact `rdi` reads:

```text
[labelhist] rdi=0x2020e32080 events(total=25): … dmaB@7587/f31 relB@7587/f31 waitB@7587/f31
            dmaX(D)@7895/f37:0x41700f1e8   relX(D)@7895/f37:0x400000000
```

So the object at `0x2020e32080` held vtable `eboot+0x700f1e8`; prosper's 4-byte immediate-zero DMA
init destroyed its low dword; prosper's paired 4-byte value-1 fence set that dword to 1; the guest's
next virtual call read `0x400000001` and jumped through `[0x400000021]`. **In this run, `rax` is
prosper's own arithmetic, not a guest value.**

Scope that claim exactly, because an earlier revision of this section did not. The `AudioMixerRende`
class occurred in four runs; **one** carries the `[labelhist]` ring above. For the other three the
attribution rests on the *value shape* alone — and the `{small nonzero high dword, low dword ≤ 1}`
row in `## Ruled out` says in terms that value shape alone cannot attribute such a qword to a GPU
write, because a live guest table legitimately holds `0x0000000400000002`. Those three are
**consistent with** the mechanism; they are not witnessed instances of it. One witnessed ring plus two
independently witnessed heap instances is what the finding rests on.

**Why no existing guard, tripwire or census could see it.** Every shape predicate in
`command_processor.cpp` filters `pre` through `ptr_like()`/`heap_ptr_like()`, whose widest window is
`[0x1000000000,0xa000000000)` — the guest arena. A C++ vtable pointer is not in the arena: it points
into a loaded **module image**, and ArcRunner's eboot base is `0x410000000`, which sits below that
floor. (By a factor of about four — an earlier revision of this section said "three orders of
magnitude", which is wrong by ~250x. The structural fact is what matters: below the floor, outside
every window.) `PROSPER_PTRLIKE_WIDE=1` does not reach it either; the wide window has the same
floor. So the `FORGE-TRIP-TOTALS wide_only=0` and `SUSPECT-REL1-LIVE=0` negatives recorded in the
section above are **correct and irrelevant** to this population — they were taken with an instrument
that cannot represent it. This is not a tuning gap: an arena bound is exactly what a heap predicate
is for.

`PROSPER_LIVEPTR_TRIP=1` (default OFF, log-only) asks the shape question with **no address window**:
`pre` counts as a live pointer when its high dword is nonzero and its value addresses mapped memory
(`guest_readable`, a real `write(2)` EFAULT probe, so module images are in scope). One bounded run:

```text
[agc] LIVEPTR-TOTALS(hit) examined=5091 shape=1280 live=1280 t=11607ms | hi=0x20:1280
[agc] LIVEPTR-STOMP hi=0x21#1 kind=DMA-imm dst=0x2020e33080 pre=0x21c1388890 (mapped, heap)
      val=0x0 -> 0x2100000000 after-mapped=0 width=4 pkt=0x300310fa18 t=12006ms | …
```

Read `examined` beside every zero: it counts every sub-qword write offered to the trip and prints on
its own schedule even when `live` is 0, so "armed and nothing matched" is distinguishable from
"never armed". **`after-mapped=0` is the fault preannounced** — the corrupted qword no longer
addresses anything, and the guest dereferences it next.

**The damage is done by the INIT, not the fence, and the init has no guard of this kind at all.**
`rel1_stomp_guard()` already declines a *fence* over `ptr_like(pre) && pre_low != 0` and ships
default ON; by the time it runs, the init has already replaced the low dword with 0, so `pre_low` is
0 and the branch cannot fire. There is no equivalent check on the 4-byte DMA immediate.

**And content alone cannot fix the heap half of it.** A label the guest legitimately popped from its
own pool free list carries a stale `NextFreeBlock`, which is a heap pointer with a nonzero low dword —
byte-identical to a live object pointer. That is the population the `hi=0x20:1280` bucket is: a
free-list walk, where each write's `dst` is the previous write's `pre`. Declining those would repeat
the #1245 regression at 1,280 events per run. The **module-image** half has no such twin: nothing in
a label protocol and nothing in an allocator free list ever holds a code-image pointer, so
`PROSPER_NONHEAP_PTR_GUARD=1` (default OFF, announces itself, counts its declines) is a sound A/B arm
for exactly that class. It is **not yet evidence of anything**: three armed runs recorded
`declines=0`, because the image-vptr class did not occur in them — non-discriminating, not negative.

**The deferred completion-write model is not what puts the write in reallocated memory.**
`PROSPER_EOP_WRITE_SYNC=1` makes completion writes land inside the submit call instead of through the
post-submit worker, and it now announces `[agc] EOP-WRITE-SYNC ARMED` so the arm is witnessable — the
earlier result from this lever was recorded as non-discriminating partly because nothing in the log
separated an armed run from an unarmed one. Five armed runs, lever witnessed on every one:

| | live-pointer stomps / sub-qword writes examined | module-image (`hi=0x4`) stomps |
| --- | --- | --- |
| default (deferred) | 1280 / 5091 = **25.1%** | 1 run in 6 |
| `PROSPER_EOP_WRITE_SYNC=1` | 512/1975, 512/1971, 256/965 = **25.9–26.5%** | **4 runs in 5** |

The rate does not move, and the vptr class occurs *inside* the armed arm — so the arm carries its own
positive control and the null is readable. Two of the sync runs stomped the **same** vtable
`0x4170033e8` (`eboot+0x70033e8`) at different destinations, and a third stomped `0x41700f1e8`, the
same vtable as the baseline `AudioMixerRende` fault: this is one recurring class of C++ object, not a
scattered accident.

The sharpest form of what remains comes from the cleanest of those hits, whose label had **no prior
generation at all**:

```text
[agc] LIVEPTR-STOMP hi=0x4#1 kind=DMA-imm dst=0x2020e31e40 pre=0x41700f1e8 (NON-HEAP) val=0x0
      -> 0x400000000 after-mapped=0 t=8745ms | events(total=3): dmaB@8408/f31 relB@8408/f31 waitB@8408/f31
```

The guest itself named `0x2020e31e40` as a label — `dmaB` is recorded from `hle_agc.cpp` at the
guest's own AGC builder call, so this is not a prosper decode error — and 337 ms later, at submit, the
block held a vtable pointer. **The guest builds a label into a 0x20-byte block and then frees and
reallocates that block before the command buffer referencing it is submitted.** Whatever gates that
free on real hardware is the thing prosper is modelling wrongly; the label write is the messenger.

**The stomped object is named, from the binary alone.** `tools/re/xref.py` over the flattened eboot
finds the vtable `eboot+0x700f1e8` referenced by exactly three `lea`s in two functions and by **no**
data relocation. Both construction sites have the same shape:

```text
12507db:  mov  edi, 0x18                   ; size = 24 bytes
12507e0:  call 0x116a0f0                   ; FMemory::Malloc wrapper: if(!size) size=1; align=0; jmp 0x1284120
12507e5:  lea  rcx, [rip+0x5dbe9fc]        ; # 0x700f1e8  <- the vtable
12507fa:  mov  QWORD PTR [rax], rcx        ; install the vptr
```

So it is a **24-byte (`0x18`) `new` allocation** with layout `{ vptr, u32 @+8, u32 @+0xc, ptr @+0x10 }`,
built in the RHI/graphics layer at `eboot+0x1250720` and `eboot+0x128e4b0`. `FMallocBinned3`'s
small-block bins step 16/32/48/…, so a 24-byte request lands in the **32-byte bin** — the same bin the
guest's 0x20-byte GPU labels come from. The 24-byte allocation size and both construction sites are
**measured from the binary**; the *bin-step* itself is not measured here and is `CONFIDENCE: MED`,
corroborated independently within this document by the provenance census — 1,314 writes at
32-byte-aligned addresses over 282 distinct slots is a 32-byte-stride pool on its own evidence.
**So this all but closes the collision statically**: the interleaving of
label destinations and this object at 32-byte stride in one 64 KiB page is two consumers of one
allocator bin, not an address-decode accident or a page-mapping coincidence, and the finding carries
no run-local addresses.

**Suppressing every label-write class prosper can currently detect does not extend the run.** The
maximal-protection arm is `PROSPER_MB3_FREELIST_GUARD=1 PROSPER_GENERATION_GUARD=1
PROSPER_REL1_WAF_GUARD=1 PROSPER_NONHEAP_PTR_GUARD=1`, and each lever is independently witnessed in
it: **26** `MB3-` suppressions, **3–10** `REL-GENERATION-CHANGED-STALE-SUPPRESS`, and **1**
`NONHEAP-PTR-DECLINE`. The run still faults on `RHIThread` at about the same time and delivers
**31** video frames with 31 successes — the same count as an ordinary unguarded run. So the label
writes are the *mechanism that composes the fault value*, and removing every detectable one of them
still leaves the title dying in the same window. That is consistent with the earlier fence-suppression
arm moving the fault two pops along the same walk rather than removing the corruption, and it means a
fix has to address the block-lifetime seam rather than any individual write.

> **The `PROSPER_GENERATION_GUARD` hits above are all the RELEASE_MEM leg.** Its DMA leg is inert on
> current builds — #1756 shrank `DMA_DATA` to its hardware 7 dwords and removed the slot carrying the
> build snapshot, and `dma_build_pre_changed()` says so once when armed. Since the DMA init is the
> write that destroys the pointer, the one generation check that could catch it is exactly the one
> that cannot run.

**Two instrument facts that shaped this and will bite the next arm.**

- The trip's detail schedule is **per `pre >> 32` bucket**, not global. On a shared ordinal the one
  image-vptr hit lands somewhere around #300 — past the first-64 window, not a power of two — and is
  counted but never printed, among hundreds of free-list-residue hits. That happened on the first arm
  here and cost a run. A rare class sharing a rate limit with a common one is invisible.
- **`nullpage_deep_dump`'s per-register memory windows print nothing** — 9 of 9 runs, all 16
  registers, including `rsp`, which is unquestionably readable. `[rdi]` would have answered this in
  one run instead of five. The `[labelhist]` line added here does not depend on `probe_readable` and
  is what recovered the evidence. Filed as
  [#2078](https://github.com/mattias800/prosper/issues/2078).
- **`PROSPER_PRESENT_NZLOG=1` on its own is inert under `boot_trace`**, and the `## Reproduction`
  block below prescribes it. A run with it set produced **zero** `[render-nz]` lines across ~100
  presented frames. The line is gated on `!px.empty()` (`frontends/shared/live_renderer.cpp`), `px`
  comes from `selected_pixels`, and the registrar announced `dump=0` — the readback that fills it is
  opt-in and this switch does not request it. It needs a companion that turns readback on. **Do not
  read a silent `PROSPER_PRESENT_NZLOG` run as "every frame was black"** — the absence of a line is
  the absence of a measurement. This does **not** impugn the `nz=0` / `nz=8294400` figures in the
  rung-1 section above: those runs reported lines, so their readback was evidently on (a companion
  such as `PROSPER_DUMP_RTGROUPS=1` or `PROSPER_FRAME_DIR` was in the same arm). The correction is to
  the *recipe*, which prescribes the switch alone.

## How far the movie actually is, and why the fault is only half the arithmetic

`sceAvPlayerGetVideoDataEx` advances the movie **one frame per guest call** — `next_video()` pops the
decode queue with no media-clock test. One bounded run: **31 calls, 31 successes** (`result=1` on
every one), and the guest makes no other AvPlayer call at all — no `IsActive` poll, no `CurrentTime`
poll. So the queue is never starved and the movie's playback rate is exactly the guest's poll rate,
measured here at ~2.6 Hz against ~8.3 presented frames/s.

The consequence for rung 1: full picture is video frame 150, so the run must survive **~150 guest
polls**, which at the observed rate is roughly a minute of wall time — not the ~13 s the fault
currently allows. Fixing the terminal fault is necessary and, on this arithmetic, close to
sufficient (the movie is 63 s long), but a rung-1 arm must be given a **long** bound and must count
`video-ex` calls, not seconds. Do not read "43-54 video frames" from the section above as a rate.

## 2026-08-06: the init-side generation census — the question #1756 made unanswerable, answered

The previous handoff named one concrete next step and recorded it as blocked: *"restoring a
build-time snapshot out of band … is the concrete next step, and it needs no content predicate"*,
blocked because #1756 shrank `DMA_DATA` to its hardware 7 dwords and removed the slot
`dma_build_pre_changed()` reads. **It was not blocked.** The fence build journal
(`prosper_fence_journal_record`, keyed by the packet's guest address) already stored a per-packet
build-time target and content snapshot for the `RELEASE_MEM`, `WRITE_DATA` and `WAIT_REG_MEM` legs —
the DMA-init builder simply never called it. It does now, and `PROSPER_DMA_INIT_GEN=1` (default OFF,
log-only, gates nothing) reports, at the instant prosper executes a 4-byte label init:

* **generation depth** — `dma_built_n - dma_exec_n`, read before this exec's own bump. `>= 2` means
  the guest has already built a LATER generation of this label's protocol while this one is still
  unexecuted. This is the init-side twin of `label_rel_overlap()`, which only sees the fence side.
* **build→exec drift** — did the target's qword change between the guest building this packet and
  prosper executing it. A consumed-marker label is deliberately uninitialised at build time, so
  residue proves nothing; residue *changing* does, because nothing may legitimately write the block
  in that window.

### The result: both are properties of the dying arm and absent from the surviving one

One binary, ordinary `boot_trace` (`PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_AVPLOG=1`), exact
zero peer-process censuses (`pgrep -x` over `prosper-app`/`screenshot`/`boot_trace`/`gpu_replay`)
immediately before and after every arm.

| arm | reps | terminal fault | video frames | examined inits | depth >= 2 | drift | age max / mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| default | 3 | **3 of 3** | 34 / 34 / 36 | 960 each | **19 / 11 / 17** | **14 / 26 / 25** of ~524 journal-matched | ~838 / ~345 ms |
| `PROSPER_SUBMIT_STALL_US=1500` | 1 | **0 of 1**, ran to the 60 s bound | 1,327 | 19,520 | **0** | **0** of 10,079 | 902 / 354 ms |

`SUSPECT-REL1-OVERLAP`, which a default build prints for free, agrees exactly: in the run where the
census reported `depth>=2=21` the log carries **21** `SUSPECT-REL1-OVERLAP` lines. The init-side and
fence-side instruments are counting the same events from opposite ends of the pair, which is the
cross-check neither had on its own.

Three things follow, and the third is the load-bearing one:

1. **The overlap and the drift are real, measurable at the init, and specific to the failing arm.**
   Roughly 2% of label inits in a dying run execute with a later generation already built, and
   ~4% execute into a block whose content has changed since the guest built the packet.
2. **The raw build→exec lag is NOT what the throttle removes.** It is essentially the same in both
   arms (~345 ms mean, ~840–900 ms max). Whatever the throttle fixes, it is not "prosper catches up".
3. **The throttle changes neither how fast the guest runs nor how fast prosper executes — only how
   long the run lives.** A first revision of this section claimed the opposite (a sixfold speed-up)
   from rates taken over **whole-run** duration. That is a denominator artifact and the row is
   corrected below: both arms spend an identical **~6.8 s** boot prefix before the first
   `sceAvPlayerGetVideoDataEx`, which dilutes a 9-second arm's average and barely touches a
   59-second one's. Measured over the movie window itself — first delivered video frame to last —
   the arms are indistinguishable, and if anything the **default** arms are marginally faster:

   | arm | first video frame | movie window | video frames/s | label inits/s |
   | --- | --- | --- | --- | --- |
   | default | 6.77 / 6.73 / 7.00 s | 1.25 / 1.24 / 1.26 s | **27.2 / 27.4 / 28.5** | **437 / 440 / 434** |
   | `PROSPER_SUBMIT_STALL_US=1500` | 6.82 / 6.81 s | 52.56 / 52.49 s | **26.3 / 26.3** | **414 / 406** |

   Both arms decode at essentially the movie's nominal 30 fps. So **both** "the throttle slows the
   guest down" and "the throttle speeds the guest up" are false, and the lever's entire observable
   effect on this title is **survival**. That is a much tighter constraint on any remaining
   hypothesis than the wrong one it replaces. It also condemns one figure this document already
   quotes: the "~2.6 Hz" poll rate in § *How far the movie actually is* is the same whole-run
   average over the same ~6.8 s prefix, and the true rate there is ~27 Hz.

### The pend queue is not the late-write path on this title

`PROSPER_PEND_AGE=1` over a full default faulting run: **peak residency 5 ms** across 4,096 applied
completion writes. Against a build→exec age of ~345 ms mean and ~840 ms max, the deferred-write
model contributes about 1.5% of the window. The entire gap is the **guest's own build-to-submit
interval** — prosper folds a submit synchronously inside the caller's HLE handler and the pend queue
adds single-digit milliseconds on top. This reproduces on ArcRunner the 3 ms figure Crisis Core
already recorded, and it closes `PROSPER_EOP_WRITE_SYNC` as a lead here for a second, independent
reason (the first is in `## Ruled out`).

### Acting on the drift does not fix the title

`PROSPER_DMA_INIT_DRIFT_GUARD` (default OFF) is the A/B arm that declines on that evidence, so this
null stays reproducible. It has two levels because the init decline and the paired-fence decline are
**not the same experiment**:

* **Level 1 — decline the drifted init.** 3 of 3 runs faulted, delivering **35 / 34 / 35** video
  frames against the baseline's 34 / 34 / 36, with the lever witnessed on every arm (**21 / 13 / 15**
  `DMA-INIT-DRIFT-DECLINE` lines). A readable null, not a silent one.
* **Level 2 — also retire that generation's paired fence** through the dma-free debt. Level 2 exists
  because the first head of this work assumed the default-ON `rel1_stomp_guard()` would decline the
  fence for free once the init no longer zeroed the low dword. **It does not**, and the terminal
  fault of that arm named the reason exactly: the surviving pre-content was `0x21c0f88b70`, above
  `ptr_like_narrow()`'s `[0x2000000000, 0x2100000000)` ceiling, so the fence guard could not see it
  and wrote `1` over the pointer's low dword anyway — producing `0x2100000001`, the value in `rdi`
  at the fault. **Declining half a pair is worse than declining neither**: the fence alone composes
  the forge without any help from the init.

**Level 2 does not stop the fault either — but it is the first intervention on this title that moves
the progress metric at all, and it moves it a long way.** Five runs, same binary and route, against
seven baseline runs whose spread is remarkably tight:

| arm | delivered video frames, per run | max |
| --- | --- | --- |
| baseline (7 runs, two binaries) | 31, 33, 34, 34, 34, 36, 37 | **37** |
| `PROSPER_DMA_INIT_DRIFT_GUARD=1` (3) | 34, 35, 35 | 35 |
| `PROSPER_DMA_INIT_DRIFT_GUARD=2` (5) | 31, 39, 147, 203, 509 | **509** |
| `PROSPER_SUBMIT_STALL_US=1500` (2) | 1,380, 1,383 — both ran to the 60 s bound | — |

**Four of five level-2 runs exceed the maximum of every baseline run**, by up to 14x, and **5 of 5
still fault**. Read it as a distribution, not a fix: level 1 is indistinguishable from baseline,
level 2 is not, and neither survives. The variance is real and large — one level-2 run delivered 31
frames — so no single level-2 run is evidence of anything on its own (see the standing `## Ruled
out` row about one-run-per-arm comparisons on this title). Peer-process censuses were exact zero
before every run and after all but one, which is recorded because a peer sharing the GPU slows
prosper and, per the throughput inversion above, slowing prosper is not obviously a bias in the
conservative direction on this title.

So the earlier `## Ruled out` verdict — *suppressing the label writes prosper can detect is not
enough* — survives, and now covers the one detectable class it had not been tested against. What is
new is that the class is detectable exactly where the previous handoff predicted, and that acting on
it as a **pair** buys real progression while acting on it by halves buys none.

### Real game graphics on the un-throttled route

![ArcRunner — the intro's TrickJump, PQube and Unreal Engine logos, captured on the default route with no submit throttle](../../assets/screenshots/arcrunner-default-route-logos.png)

`assets/screenshots/arcrunner-default-route-logos.png` is the **unmodified** `tools/screenshot`
frontend capture, 3840x2160, taken with `PROSPER_GUEST_ARGS= PROSPER_RENDER=1
PROSPER_RENDER_SCALE=1 PROSPER_RENDER_EVERY=1 PROSPER_DMA_INIT_DRIFT_GUARD=2` and **no**
`PROSPER_SUBMIT_STALL_US`. It carries the guest's *TrickJump* and *PQube* logos, the *Unreal Engine*
logo, and the game's own *PRESS ANY BUTTON TO SKIP…* prompt: 426,965 non-black RGB pixels of
8,294,400 and 47 distinct colours. It is very dark because the movie is still fading in at this
point; the image was **opened and read**, not merely measured.

Three things this is and is not.

* It **is** the first real ArcRunner game imagery produced without the diagnostic throttle. Baseline
  cannot reach it structurally, not merely by luck: the movie's first non-black frame is video frame
  60 and seven baseline runs deliver 31–37.
* It is **not** rung 1, and the guard is **not** a candidate correctness change. An earlier
  revision of this section called it one — "it declines a write into a block whose content proves
  the guest has repurposed it" — and that is wrong on the decisive point. **It is a lever, in the
  same evidentiary class as the throttle**, for five reasons, in weight order:
  1. It **discards** two writes the guest asked for and hardware performs — the `DMA_DATA` init, and
     via the dma-free debt the paired `RELEASE_MEM` fence. It does not re-time, re-target or
     re-derive either one. A fix for *"prosper executes this too late"* executes it at the right
     time.
  2. **The predicate measures prosper's schedule, not the title.** Drift is 14–26 per dying run and
     **0 in 10,079** under the throttle. Nothing about ArcRunner differs between those arms — only
     prosper's timing does.
  3. The section above already says it in other words: the declined writes are *the messenger*, and
     only the throttle, *which changes the schedule rather than the writes*, removes the fault.
  4. **The 5/5-still-fault spread is a widened race window, not a removed defect.** 31 / 39 / 147 /
     203 / 509 spans 16x with one run at the baseline floor. Removing a defect looks like the
     throttled arm: 1,380 / 1,383, tight, 0 of 2 faulting.
  5. It is **strictly broader** than the default-ON guards it resembles. `rel1_stomp_guard` and
     `mb3_suppress_release` decline only the *fence*, on a *content-shape* predicate. Level 2 also
     declines the *init*, on a *temporal* predicate, and forces the fence through the debt even
     where the fence's own predicate would have allowed it.

  The precedent is already in this document: with the throttle the title renders its whole intro
  cinematic in 4K and is still held at rung 0. Guard-dependent output is the same class of evidence
  — about prosper's renderer, not about the title's progression. **Do not make this guard default-ON
  on the strength of the progression it buys.**
* It is **not** a `MODE=2` artifact. That failure mode is a single-colour blanket, and this frame has
  47 distinct colours and legible geometry. One sample in the same session *was* one — a uniform
  `distinct_rgb_colors=1` frame with all 8,294,400 pixels non-black at 30 delivered video frames —
  and it is excluded on exactly that test. Keep using it: `distinct_rgb_colors` separates the two in
  the `screenshot` frontend's own JSONL with no extra instrument.

**Sampling cadence changed the subject.** Two capture batches, same build and route: at
`--seconds 3` the runs delivered 140 video frames and two of five reached picture; at `--seconds 1`
they delivered 30–31 and none did. The 4K readback per sample is not free. Instrument trap 104
again — take the widest sampling interval that can still catch the window.

**Two limits on the drift predicate itself, both measurable with what this census already prints.**

1. **The journal is keyed by packet guest address and replaces on write.** If a later generation of
   the same label rebuilds at the same address while the older packet is still unexecuted, the
   census compares the older exec against the *newer* build's snapshot — the wrong baseline — and
   that population is exactly `depth>=2`, exactly where the guard fires. The throttled control
   cannot discriminate it, because `depth>=2=0` and `drift=0` co-occur there. What bounds it is the
   census's own `target-changed` count, which is **0, 0, 0** across the three default arms (~524
   journal hits each) and 1,249 / 1,208 across the two 22,000-init throttled arms. So no slot was
   reused *at all* in the window where the guard fires, and slot reuse landing on a **different**
   label is far likelier than reuse landing on the same one — several hundred distinct label
   addresses share a monotonically advancing packet ring. Zero of the likelier kind bounds the
   rarer kind; it does not eliminate it. Settling it needs a build generation in the journal record,
   not another arm. `CONFIDENCE: MED` on the drift predicate; it is one more reason the guard is a
   lever.
2. **Roughly 45% of label inits are invisible to both the census and the guard**, and the rate is
   **structural rather than collision**: 436 / 434 / 436 of 960 in the default arms and 9,290 /
   9,199 of ~22,000 in the throttled ones — 45.4% against 42.2% across a 23x population change. A
   hash-collision miss would fall with population; this does not. The likely cause is one queue
   whose builder records the journal against a packet address the fold does not execute from.
   Tracked as [#2126](https://github.com/mattias800/prosper/issues/2126).

**One switch interaction to know before quoting a level-1 arm.** The declined init records
dma-free debt, and **two** paths consume that debt: `declines_drifted_pair_release` (level 2) and
`mb3_suppress_release`, which is gated on `PROSPER_MB3_FREELIST_GUARD`. So
`PROSPER_DMA_INIT_DRIFT_GUARD=1` **together with** `PROSPER_MB3_FREELIST_GUARD=1` behaves as level 2,
because the MB3 release leg retires the paired fence that level 1 deliberately leaves alone. The
level-1 arms recorded above did not arm the MB3 guard, so they are level 1. Anyone combining the two
must read the arm as level 2 or the comparison is between two labels for one condition.

### What this leaves

The mechanism is now stated as narrowly as the evidence allows: **the guest rebuilds a
consumed-marker label roughly one frame after building the previous generation's packet, and prosper
executes that packet at the very end of the same frame; the margin is under one submit.** In the
label rings the pattern is exact and repeatable — build at fold *f*, exec at fold *f+5* or *f+6*,
rebuild at fold *f+6*. When the exec lands on *f+6* rather than *f+5* the two collide, and which of
them reaches the block first decides whether the run survives. That is why every intervention that
merely suppresses one of prosper's two writes moves the fault without removing it, and why the
throttle — which changes the schedule rather than the writes — removes it completely.

The open question is therefore **what sets the guest's recycle margin**, not what writes the value.
Nothing in this pass measured that, and the throughput inversion above says the cheap answers
("prosper is late", "the guest outruns us") are the wrong shape.

## 2026-08-06 (arc6): the title screen is reached, and the throttle rescues by DELAY, not by lock hold

Three results, on `8ab70b74` plus the discriminator this section adds. All three runs of every arm
used `PROSPER_GUEST_ARGS= PROSPER_RENDER=1` with `PROSPER_RENDER_SCALE=1` and
`PROSPER_RENDER_EVERY=1` at their defaults, and no suppression, skip, guard or shim.

### 1. Past the cinematic is the title screen, and it holds

![ArcRunner — the title screen, captured on the PROSPER_SUBMIT_STALL_US=1500 throttled route](../../assets/screenshots/arcrunner-title-screen.png)

`assets/screenshots/arcrunner-title-screen.png` is an **unmodified** `tools/screenshot` frontend
capture at 3840x2160 — the *ArcRunner* logo, `PRESS ✕ TO START`, the TrickJump and PQube logos and
`VERSION 1.0.1 RELEASE`. One 288 s run, `--seconds 8 --count 36`, route
`PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_SUBMIT_STALL_US=1500 PROSPER_AVPLOG=1`. The run
delivered **1,926** `sceAvPlayerGetVideoDataEx` frames — past the movie's 1,908 — reached the title
screen at sample 13 (t = 112 s) and **held it to the end of the run**: samples 13 through 35 are all
6,880 distinct RGB colours, and the last four share one CRC across 96 s. Every image was opened.

Two things this settles and one it does not.

* It is **not** a `MODE=2` blanket. That failure mode presents as `distinct_rgb_colors=1`; these
  samples are 6,880 with legible glyphs. The metric is shown to separate the two **inside this run**:
  sample 0 reads `distinct_rgb_colors=1` and is uniform `(0,0,0)` over all 8,294,400 pixels (an empty
  first frame, not a blanket — a blanket is a non-black uniform colour).
* **Nothing other than #1226 is now known to stand between this title and rung 2.** Before this run
  the throttled route was only known to reach the end of the intro cinematic, so "what is behind the
  movie" was open. It is a title screen.
* It is **not** rung 2, and the rung is deliberately not raised. This is the same evidentiary class
  as the cinematic — output that exists only under a diagnostic lever — and the tracker already
  refuses rung 1 on exactly that ground. Fix #1226 and this becomes rung 2 on the default route.

No peer-process census was taken for this run; it is a progression observation rather than a timing
measurement, and contention cannot manufacture a title screen.

### 2. The throttle's mechanism is the DELAY, not the submit mutex

`submit_stall()` is called **inside** `g_agc_state_mu`, so a surviving throttled run has had two
candidate mechanisms all along and the dose-response cannot separate them: the guest's submit call
takes 1.5 ms longer (**delay**), or prosper serialises the `SubmitDcb` and `SubmitAcb` entry points
for 1.5 ms per submit and thereby changes the **interleaving** of the two queues' folds (**lock
hold**). The second is not idle speculation — `hle_agc.cpp` already records for #305 that prosper's
fold order "is a property of lock acquisition, not of the guest's program order", and the label rings
on this title carry both `(D)` and `(A)` origins.

`PROSPER_SUBMIT_STALL_OUTSIDE=1` (added here, default OFF) runs the **same** sleep, of the **same**
length, on the **same** thread, after the mutex is released instead of while it is held. Only the
mutex differs.

| arm | stall | mutex during the sleep | runs | faulted | delivered video frames |
| --- | --- | --- | --- | --- | --- |
| control | none | — | 3 | **3 of 3**, exit 90 | 36 / 35 / 36 |
| A | 1500 µs | held (current behaviour) | 3 | **0 of 3**, ran to the 70 s bound | 1,629 / 1,635 / 1,628 |
| B | 1500 µs | released first | 3 | **0 of 3**, ran to the 70 s bound | 1,640 / 1,627 / 1,642 |

**B is indistinguishable from A, so the rescue is the delay.** Moving the sleep out from under the
submit mutex changes nothing — not survival, not the delivered-frame count, not `SUSPECT-REL1-OVERLAP`
(0 in all six throttled arms). Any hypothesis that the defect is prosper serialising two hardware
queues through one mutex has to explain why releasing that mutex for the whole stall costs nothing.

The arms were **alternated** A, B, A, B, A, B so a time-varying confound lands on both — which
matters here, because a peer lane held the GPU for five of the six arms (`peers_pre`/`peers_post` of
1, disclosed rather than averaged away). Alternation is what makes the A-versus-B comparison sound
under that contention; it does **not** license comparing either arm against a historical default one.

Both levers are witnessed per run, and the control doubles as the discriminator's **mutation test**:
`PROSPER_SUBMIT_STALL_OUTSIDE=1` with no `PROSPER_SUBMIT_STALL_US` prints
`STALL_OUTSIDE=1 NOT ARMED — … there is no stall to move` (1 line in each of the three control runs)
rather than reading as an armed arm that found nothing. Those same three runs fault 3 of 3 at
`eboot+0x117e225`, which is the control that matters for the code: the `lock_guard` → `unique_lock`
change this discriminator needed is inert when unarmed.

```bash
# A — current behaviour           # B — same sleep, mutex released first
PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_AVPLOG=1 PROSPER_SUBMIT_STALL_US=1500 \
  [PROSPER_SUBMIT_STALL_OUTSIDE=1] ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0
```

### 3. No false-success NID is in this title's failing path

ArcRunner statically imports **all three** of the `sceAgc*GetSize` functions #1756 deliberately left
unregistered (`sceAgcDcbWriteDataGetSize`, `sceAgcCbSetShRegisterRangeDirectGetSize`,
`sceAgcCbSetShRegistersDirectGetSize`), and also `sceKernelWaitCommandBufferCompletion` — a set that
reads like a ready-made explanation, since a `GetSize` answered with 0 makes the guest reserve
nothing (`docs/AGC_PACKET_SIZES.md`) and a wait answered with 0 tells a guest that work completed.
`tools/nid_census` lists them because it is a **static** census of imports.

**None of them is called.** A full default faulting run with the runtime unimplemented-call census
enumerates **12** distinct NIDs — `libSceCoredump`, `libScePosix`, `libSceHttp`, `libSceNpWebApi2`,
`libSceVoiceQoS`, `libSceNet`, `libSceAjm` and four `EOSSDK-PS5-Shipping` — and **not one** is in
`libSceAgc`, `libSceAgcDriver` or `libkernel`. Bounded honestly: that run reached 34 video frames
before faulting, so this covers the boot and intro-movie window, which is the window the fault lives
in. Generalise the method rather than the result: `nid_census` ranks what a title *could* call, and
only the boot log says what it *did*.

```bash
./build-linux/nid_census <DUMP_ROOT>/PPSA21406-app0 --names <PS5_LIBS_DIR>   # static: what could
PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_PROGRESS=1 PROSPER_PROGRESS_UNIMPL=1 \
  ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0                        # runtime: what did
```

## 2026-08-07 (arc7): the per-fold account — the guest's builder is released MID-FOLD, and the contract that prevents it is version-gated off

arc6 left one thing owed: *"the next instrument owes a **per-fold** account of what 1.5 ms of wall
time per submit changes — not another whole-run rate, which has now produced two retracted rows on
this title."* This section is that account, and it ends with a candidate fix that survives the
default route.

`PROSPER_FOLD_MARGIN` (new, default OFF, log-only, gates nothing) has two halves that arm together:

* a **per-submit ledger** in `hle_agc.cpp` splitting every fold into `gap` (the guest's own time
  between prosper's return and its next submit entry), `lock` (waiting for `g_agc_state_mu`), `work`
  (decode + fold + execute) and `stall` (the diagnostic throttle, *measured* rather than assumed);
* a **fold-margin census** in `command_processor.cpp` counting the recycle race in FOLDS rather than
  milliseconds — the unit the mechanism is stated in. The build journal now records the fold a packet
  was built in, so build->exec distance is expressible in folds; `REBUILD-BEFORE-EXEC` fires at the
  guest's own builder call when it rebuilds a label whose previous generation prosper has not
  executed.

At level 2 it prints one line per fold. **Level 2 is not timing-neutral** — a control run lives
~9 s at level 1 and ~13 s at level 2 — so compare level-2 arms only against level-2 arms. Every arm
below still faults or survives as its level-1 twin does.

### 1. prosper's fold is the bottleneck, and the throttle is 1.7% of it

Steady state, both arms, from the ledger: **`work` = 64–67 ms per submit**, `gap` = **5–50 µs**.
Excluding the single ~5.3 s boot-prefix gap, the guest spends essentially no time outside prosper —
its RHI thread returns from one submit and calls the next within tens of microseconds, so
**prosper's synchronous fold is the pacer for this title and the guest is inside it ~84% of wall
time.** A 1500 µs stall is 1.7% of one fold and drops the fold rate ~8% (15.0 -> 13.8 folds/s after
boot). Whatever the throttle does, it is not a large re-timing.

The steady state is a repeating three-fold cycle, one per frame:

| fold | entry point | `work` | label builds | label inits executed |
| --- | --- | --- | --- | --- |
| consume | `SubmitDcb` | ~110 ms | 0 | 58–59 |
| build | `SubmitDcb` | ~85 ms | 46 | 26–30 |
| compute | `SubmitAcb` | ~0.6 ms | 0 | 0–1 |

### 2. The exposure window is LONGER under the throttle, not shorter

Build->exec distance in folds, from the journal, over the population both arms have:

| arm | age-folds histogram | mode | mean |
| --- | --- | --- | --- |
| default (n = 506) | 1:11 2:42 3:53 4:48 **5:239** 6:75 7:38 | 5 | 4.66 |
| `SUBMIT_STALL_US=1500` (n = 7,854) | 1:18 2:197 3:37 4:1129 5:375 **6:4664** 7:818 32+:616 | 6 | 20.4 |

In milliseconds it is longer too (~5 x 66 ms vs ~6 x 72 ms). So the throttle does **not** shorten the
build-to-exec window — it lengthens it, and the corruption still vanishes. This is the third and
strongest form of that falsification: the two earlier attempts were retracted for sampling from a
population only one arm has, and this one is drawn from every journal-matched init in both arms.
It also kills the reading the arc5 mechanism sentence invites — *"exec at f+5 is safe, exec at f+6
collides"* — because f+6 is the surviving arm's **mode**.

### 3. `REBUILD-BEFORE-EXEC` is the discriminator, and it is a single-fold burst

| arm | runs | faulted | delivered video frames | `REBUILD-BEFORE-EXEC` |
| --- | --- | --- | --- | --- |
| default | 3 | **3 of 3**, exit 90 | 29 / 30 / 32 | **16 / 31 / 14** of ~380 builds-with-exec |
| `SUBMIT_STALL_US=1500` | 3 | 0 of 3, to the 40 s bound | 861 / 858 / 856 | **0 / 0 / 0** of ~6,700 |

The three instruments agree exactly and independently: in every default arm the
`REBUILD-BEFORE-EXEC` count (build side), the `depth>=2` exec count (init side) and the
`SUSPECT-REL1-OVERLAP` population (fence side) are the **same number**.

It is not a trickle. Every event of a run lands in **one fold**, at one millisecond, across 13–21
*different* labels of one command-buffer chunk, each exactly one generation behind:

```text
[agc] FOLD-MARGIN-REBUILD-BEFORE-EXEC #1  [0x2020f395e0] fold=37 built=5 exec=4
      last-build-fold=31 last-exec-fold=30 t=7593ms
... 15 more, same fold, same millisecond, 15 other addresses ...
```

### 4. What the failing fold actually looks like — the guest builds INTO the consuming fold

The per-fold trace localises it to one deviation from the three-fold cycle. Reproduced in 3 of 3
default arms and absent from 3 of 3 throttled ones:

```text
default (dies)                                  throttled (survives)
#35 Dcb  work=74ms  built=46 ex=26              #35 Dcb  work=73ms  built=46 ex=26
#36 Acb  work=0.7ms built=0  ex=0               #36 Acb  work=0.6ms built=0  ex=1
#37 Dcb  work=91ms  built=46 ex=57  rbe=16  <<  #37 Dcb  work=87ms  built=0  ex=56
#38 Dcb  work=97ms  built=0  ex=28              #38 Dcb  work=100ms built=46 ex=28
```

`built` is drained per fold, so it counts what the guest's **builder thread** did while prosper was
inside that fold. Normally the 46 rebuilds are concurrent with the *build* fold, which executes 26–30
inits. In the failing arm they are concurrent with the *consume* fold, which executes 57 — including
the previous generation of those same 46 labels — and the builder beats prosper to 16 of them.

**So the guest's builder thread is released mid-fold.** prosper applies a submit's completion writes
while it is still executing the rest of the same command buffer, the guest's chunk recycler sees its
consumed markers, and it rebuilds labels whose init packets prosper has not reached yet. The race is
not between frames; it is *inside one fold*.

### 5. The contract that forbids exactly this exists, and ArcRunner is version-gated out of it

`command_processor.cpp`'s post-submit visibility model holds completion writes private until the
submit scope closes, with its own comment recording *"CONFIDENCE: HIGH on the invariant (completion
is post-submit by construction on real HW …)"*. It is armed from `agc_reg_defaults.cpp` by
`if (version >= 13)`. **ArcRunner requests SDK version 10**, so on this title fence writes become
visible in the middle of the fold that produced them — which is the mechanism in §4.

`PROSPER_POST_SUBMIT_VISIBILITY=1` (new, default OFF) forces the model on regardless of version.
Three arms, **default route, no throttle**, lever witnessed on every one:

| arm | runs | faulted | delivered video frames | `REBUILD-BEFORE-EXEC` | mean fold `work` |
| --- | --- | --- | --- | --- | --- |
| default | 3 | **3 of 3** | 29 / 30 / 32 | 16 / 31 / 14 | 64–66 ms |
| `POST_SUBMIT_VISIBILITY=1` | 3 | **0 of 3**, to the 40 s bound | **788 / 782 / 838** | **0 / 0 / 0** | **14.7–15.0 ms** |

Zero `WORKER-THREAD FAULT`, zero `FMallocBinned3`/`LowLevelFatalError` in all three. The fold also
gets **4.4x cheaper**, because the synchronous completion drains the pre-13 path performs inside the
fold move to the worker.

**Those three arms had one peer `boot_trace` present, and on this title that is not a neutral bias** —
the throttle is protective, so a peer slowing prosper could in principle manufacture the survival.
Re-run alternated control/armed with `pgrep -x` censuses of **zero before and after all six**:

| arm | rc | faults | fatals | delivered video frames | `REBUILD-BEFORE-EXEC` |
| --- | --- | --- | --- | --- | --- |
| control 1 | 90 | 1 | 0 | 139 | 0 |
| control 2 | 90 | 1 | 0 | 30 | 20 |
| control 3 | 90 | 1 | 0 | 31 | 33 |
| `POST_SUBMIT_VISIBILITY=1` 1 | 124 | 0 | 0 | **924** | 0 |
| `POST_SUBMIT_VISIBILITY=1` 2 | 124 | 0 | 0 | **919** | 0 |
| `POST_SUBMIT_VISIBILITY=1` 3 | 124 | 0 | 0 | **935** | 0 |

Control 1 is worth reading rather than averaging away: it lived to 139 video frames and recorded
`REBUILD-BEFORE-EXEC=0`, i.e. it died on one of the other two terminal paths this title races. That is
the standing reason a single run cannot A/B here, restated by a control that behaved exactly as the
`## Ruled out` row predicts.

**This is a candidate correctness change, not a lever in the throttle's class**, and the distinction
is the one the throttle's own comment draws: the throttle discards nothing and models nothing, it
just delays the guest. This arms a contract prosper already implements, already believes is
hardware-true, and already applies to every SDK-13 title.

### 5a. It reaches the title screen on the default route

![ArcRunner — the title screen at 3840x2160 on the DEFAULT route with no submit throttle, under PROSPER_POST_SUBMIT_VISIBILITY=1](../../assets/screenshots/arcrunner-title-screen-default-route.png)

`assets/screenshots/arcrunner-title-screen-default-route.png` is an unmodified 3840x2160 presented
frame from `boot_trace`'s live renderer (`PROSPER_FRAME_DIR`), route
`PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_AVPLOG=1 PROSPER_POST_SUBMIT_VISIBILITY=1` with
`PROSPER_RENDER_SCALE` and `PROSPER_RENDER_EVERY` at their defaults and **no**
`PROSPER_SUBMIT_STALL_US`, no suppression, no guard and no skip. The 260 s run delivered **1,977**
`sceAvPlayerGetVideoDataEx` frames — past the movie's 1,908 — with **zero** `WORKER-THREAD FAULT`
and **zero** `FMallocBinned3`/`LowLevelFatalError` lines, and ended by timing out (`rc=124`) rather
than dying. It carries the *ArcRunner* logo, the TrickJump and PQube logos and
`VERSION 1.0.1 RELEASE`. The image was opened, not merely measured.

A 220 s companion run on the same route delivered 1,177 video frames with zero faults and rendered
the intro cinematic — the rainy neon street with the game's own *PRESS ANY BUTTON TO SKIP…* prompt,
which baseline cannot reach structurally (the movie's first non-black frame is video frame 60 and a
default arm delivers 29–32). The green/magenta cast is the already-filed chroma-plane fault
([#2094](https://github.com/mattias800/prosper/issues/2094)), unrelated to this change.

**This is not rung 2 yet**, and the rung is deliberately not raised: it needs an env var, so it is
one gate condition away from a default launch rather than at one.

### 5b. The prediction transfers to Crisis Core, the other title in this family

`CRISIS_CORE_STATUS.md` records the identical `(500, 1500]` dose-response on `PPSA07809`. That title
**also requests SDK version 10**, so the same contract is unarmed for it — which makes "forcing it on
rescues Crisis Core too" a prediction rather than a second search. Same binary, same census method,
alternated control/armed:

| arm | runs | faulted | note |
| --- | --- | --- | --- |
| default | 3 | **3 of 3**, exit 90 | one arm also carries 4 `FMallocBinned3`/fatal lines |
| `POST_SUBMIT_VISIBILITY=1` | 3 | **0 of 3**, to the 40 s bound | lever witnessed on each; zero fatals |

Two independently brought-up titles, one unarmed contract, the same rescue. Peer-process censuses
were exact zero before and after all six arms.

### 5c. Why the gate is NOT removed in this pass

**Removing it is not a no-op for the matrix.** A census of the SDK version each title requests
(`[agc] register defaults requested for SDK version N`, printed unconditionally on every boot):

| SDK version | titles |
| --- | --- |
| 13 (already armed) | *The Messenger* `PPSA24651`, *Blue Prince* `PPSA25009`, *Little Nightmares III* `PPSA05143`, *Dragon Quest VII* `PPSA17942` |
| 12 | *The Oregon Trail* `PPSA19244` |
| 10 | *Dead Cells* `PPSA15552`, *Blasphemous 2* `PPSA13579`, *Sonic Frontiers* `PPSA03831`, *Crisis Core* `PPSA07809`, *ArcRunner* `PPSA21406` |
| 8 | *Alex Kidd in Miracle World DX* `PPSA02664` |

Three of the pre-13 titles are **rung-6, snapshot-guarded** (Dead Cells, Blasphemous 2, Alex Kidd),
so the gate is load-bearing for the guarded matrix and removing it must be scored against it. The
gate's own provenance carries no recorded evidence — it arrived inside `474af058`, a large DQ7 commit
with a one-line message, and `#2031` kept it while explicitly moving the *register table* off the
same gate — but a numeric improvement is not evidence of a correct model. **The next step is a
cross-title snapshot pass with the gate removed, plus review; that is what stands between this title
and rung 2**, and it is now the only thing. Tracked as
[#2220](https://github.com/mattias800/prosper/issues/2220).

### 6. One hypothesis this pass killed with its own instrument

The per-fold trace shows the ACB queue's single label init landing **one fold earlier** in a default
arm than in a throttled one (27 of 33 ACB folds carry zero against 156 of 156 carrying exactly one),
which reads as a sub-fold phase shift of prosper's own writes — the ~1 ms pend-drain latency against
the measured `(500, 1500]` threshold. `PROSPER_EOP_WRITE_SYNC=1` reproduces the shift exactly (ACB
folds carry their init 33/33, 11/11, 12/12; the big Dcb fold drops 59 -> 58, the same one init moving)
and **still faults 3 of 3**, with `REBUILD-BEFORE-EXEC` at 8 / 0 / 21. The shift is real and not
causal. See the `## Ruled out` row.

### Reproduction

```bash
# level 1: totals only, timing-neutral enough to compare against a default arm
PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_AVPLOG=1 PROSPER_FOLD_MARGIN=1 \
  timeout 40 ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0

# level 2: one line per fold. NOT timing-neutral -- level-2 arms compare only to level-2 arms.
PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_AVPLOG=1 PROSPER_FOLD_MARGIN=2 \
  timeout 40 ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0

# the candidate: default route, no throttle, post-submit visibility forced on
PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_AVPLOG=1 PROSPER_POST_SUBMIT_VISIBILITY=1 \
  timeout 40 ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0
```

Score exactly as the dose-response does: SURVIVED = `rc=124` **and** zero `WORKER-THREAD FAULT`
**and** zero `unrecognized block|LowLevelFatalError`. Peer-process censuses (`pgrep -x` over
`boot_trace`/`prosper-app`/`screenshot`/`gpu_replay`) were exact zero before and after every arm in
§1–§4 and §6; the three `POST_SUBMIT_VISIBILITY` arms in §5 ran with **one** peer `boot_trace`
present, disclosed rather than averaged away — it bears on their fold timings, not on whether they
faulted.

## Ruled out

Do not re-derive these without contradictory new evidence.

| Hypothesis | Evidence that killed it | Ref |
| --- | --- | --- |
| The throttle rescues the title by shortening the guest's build-to-submit exposure window — the reading the arc5 mechanism sentence invites (*build at f, exec at f+5 or f+6, rebuild at f+6; when the exec lands on f+6 the two collide*), which makes "exec at f+5 is safe, f+6 collides" the thing a fix has to restore | **Falsified for the third time, and this time on a population both arms share and in the unit the mechanism is stated in.** With the build journal recording the fold a packet was built in (`PROSPER_FOLD_MARGIN`), the build->exec distance in FOLDS is **longer** under the throttle, not shorter: default mode **5** (239 of 506 journal-matched inits, mean 4.66), throttled mode **6** (4,664 of 7,854, mean 20.4) — and longer in milliseconds too (~5 x 66 ms against ~6 x 72 ms). So f+6 is the **surviving** arm's mode. The two earlier attempts at this question were retracted for sampling only from diagnostics a dying arm prints; this one is every journal-matched init in both arms, at n = 506 against 7,854. | #1226 |
| prosper's own writes landing a fold late is what the throttle fixes — the sub-fold phase shift the per-fold trace shows directly, with the ~1 ms pend-drain latency sitting inside the measured `(500, 1500]` threshold | **Real, reproduced, and not causal.** The per-fold trace shows the ACB queue's single label init executing one fold EARLIER in a default arm than in a throttled one (27 of 33 ACB folds carry zero inits against 156 of 156 carrying exactly one; the big `SubmitDcb` fold carries 59 against 58 — the same one init moving between adjacent folds). `PROSPER_EOP_WRITE_SYNC=1`, which removes the pend queue entirely, **reproduces the shift exactly** — ACB folds carry their init 33/33, 11/11, 12/12 and the big Dcb fold drops to 58 — and **still faults 3 of 3** (`REBUILD-BEFORE-EXEC` 8 / 0 / 21, lever witnessed by `EOP-WRITE-SYNC ARMED` in each). A discriminator whose lever is visible in the same census as its endpoint, returning a witnessed null. | #1226 |
| The submit throttle rescues this title by holding `g_agc_state_mu` across the sleep — i.e. the real defect is prosper serialising the `SubmitDcb` and `SubmitAcb` entry points through one mutex, so the stall works by re-interleaving the two queues' folds rather than by delaying anything | `PROSPER_SUBMIT_STALL_OUTSIDE=1` runs the identical 1500 µs sleep on the identical thread with the mutex **released** first. Three arms each, alternated A/B/A/B/A/B: **0 of 3 faulted with the mutex held, 0 of 3 with it released**, delivering 1,629/1,635/1,628 against 1,640/1,627/1,642 video frames, with `SUSPECT-REL1-OVERLAP` 0 in all six. The rescue is therefore the **delay**. Positive control on the same binary: three unthrottled runs fault **3 of 3** at `eboot+0x117e225` with 35–36 video frames, so the `lock_guard` → `unique_lock` change the lever needed is inert unarmed; and the lever's own mutation arm (`OUTSIDE=1` with no `STALL_US`) prints `NOT ARMED` rather than passing as an armed null. A peer lane held the GPU for five of the six throttled arms — disclosed, and the reason the arms were alternated; it bears on any comparison against a *historical* arm, not on A versus B. | #1226 |
| A `*GetSize` or wait NID that falls to the dispatcher's return-0 default is what makes this guest recycle a label block early — the FALSE SUCCESS class of #2081, and ArcRunner imports every candidate that would fit (all three `sceAgc*GetSize` gaps #1756 left open, plus `sceKernelWaitCommandBufferCompletion`) | It calls none of them. `tools/nid_census` is a **static** import census; the runtime unimplemented-call census over a full default faulting run lists **12** distinct NIDs, all in `libSceCoredump` / `libScePosix` / `libSceHttp` / `libSceNpWebApi2` / `libSceVoiceQoS` / `libSceNet` / `libSceAjm` / `EOSSDK-PS5-Shipping`, and **none** in `libSceAgc`, `libSceAgcDriver` or `libkernel`. Bounded to the boot and intro-movie window (34 delivered video frames), which is the window the fault occurs in. **The general lesson is the gap between the two censuses**: an import is what a title *may* call, and only the boot log says what it *did*. | #2081, #1226 |
| The `addr=(nil)` / `rip=0` worker-fault class on this title is a defect distinct from the `0x30016000` free-list class (the split this doc and #1944 have tracked separately) | **Falsified — both are one forged qword, and the low-bit read artifact is only how one of them presents.** The `addr=(nil)` site disassembles (offset `0x117e21d`, via `tools/il2cpp/prx_to_elf.py` + `objdump -b binary`) to `mov (%r12,%rcx,8),%rdi` / `mov 0x40(%rdi),%rcx` / `vucomisd (%rcx),%xmm0`, and at the fault `rdi = 0x2100000001` — the same **low-dword-replaced-by-1** shape as `0x2000000001`. The intervening `[lazy-commit] mapped page=0x2100000000 rip=0x41117e221` is prosper *backing the forged pointer's page*, so the corrupt dereference succeeds, returns zeros, and the SIGSEGV surfaces one instruction later as `addr=(nil)` with the corrupt value nowhere in the report. #1944's handler is therefore **downstream** of this bug and destroys its attribution; it is still not the writer (that row stands). | #1945, #1226, #1944 |
| The corrupt qword is computed by the guest, or is a constant prosper hands it — "a value bit-identical across four unrelated UE4 builds is more likely something prosper hands the guest" | **Falsified in both halves, by exact-address attribution.** (1) prosper's own write ring, read at the fault with the widened probe set, names the author byte-for-byte: `seq=548 kind=4 addr=0x2020e39da0 size=4 value=0x0 pkt=0x30030d5b90` immediately followed by `seq=549 kind=1 addr=0x2020e39da0 size=4 value=0x1 pkt=0x30030d5bac` — a `DMA_DATA` (`size=4 value=0x0`, i.e. the immediate form) and a `RELEASE_MEM` fence `1`, both 4 bytes at the *same* guest address — and at the fault the qword there reads `0x0000000400000001`. **The low dword is prosper's fence value, written by prosper at that exact address**; the high dword `0x00000004` is pre-existing content prosper did not write, pointer-plausible for the guest-image range (`0x4………`) though `4` is also an ordinary adjacent `int32`. So the corrupt value is **not guest-computed** — prosper authored its low dword — and the faulting `RHIThread` held that exact address in `rdi`. What this does **not** settle is *ownership*: an immediate-zero `DMA_DATA` followed by a `RELEASE_MEM` fence `1` at one address is also prosper's normal label init-then-signal sequence, so "prosper stomped a live guest pointer" and "prosper correctly fenced a label the guest had recycled under it" are both still open, and distinguishing them needs a per-address write history for that label (#1995); n=1. `CONFIDENCE: MED` on the composition, `LOW` on ownership. **A fourth arm, at the canonical `addr=0x30016000` / `eboot+0x127e751` site, returned ZERO ring hits** over `probe-pages=5/5` and zero clock-fence targets — a *partial negative*, bounded two ways: the ring retains only prosper's last 16,384 writes (#1995), and the forged node the pop misread lives at `0x2000000001`, whose page no register held, so it was never probed. That arm did yield new structure: the per-thread bin array is `{head, count}` pairs at 0x20 stride (`rax=rdi=0x3152350000`, `rdx` = the bad entry at `+0x20`), and **two independent bin arrays** (`0x3152350000+0x20` and `r14=0x300d400000+0x20`) carried head `0x30016000` at once, with counts `1` and `0x31`. (2) The constant `0x30016000` is authored by nobody: `PROSPER_FORGE_TRIP=1` on PPSA19244 reports `pre=0x2000000000 val=0x1 -> 0x2000000001` with `misread(*forged)=0x30016000` on every hit — it is the misaligned read of the qword at the arena base, which is why it is identical across titles. | #1945, #1226 |
| The `0x30016000` "POOLSHIFT byte-shifted pool pointer" is a **structurally different artifact** from the `0x2000000001` free-list value — the two shapes this issue tracked as separate legs since #1249 | They are the same shape one dereference apart. A qword that is a pointer with its **low bit set** misreads, when dereferenced, to the target qword shifted down by 8: `*(0x2000000001) = 0x30016000`, byte-for-byte the value the terminal fault dereferences. The pop at `eboot+0x127e751` both **returns** that node — the guest's own `FMallocBinned3 Attempt to free/realloc an unrecognized block 2000000001` fatal — and **stores** the misread as the next head, which is the SIGSEGV. Confirms the session-10 "the byte-shift is a READ ARTIFACT" note with a measurement. **This is about the shape, not the author** — see the next row. | #1226, #1754 |
| **The paired 4-byte `DMA_DATA` immediate-zero init is the "first half" of the damage**, systematically destroying live `FFreeBlock::NextFreeBlock` links (#1754 left this at `CONFIDENCE: MED`) | Not systematic. Whole-run totals with `PROSPER_INIT_TRIP=1`: **n=1024, overptr=926, member=10, both=1** — 90% of these inits do overwrite pointer-shaped content, but only ~1% of destinations are on an idx=1 chain the walk models, and exactly **one** init in the run is both. The overwritten pointer is almost always the label pool's own stale link in a block the guest holds. Two cautions: `member` positives appear **late**, so any single sample before ordinal ~512 reads 0 and must not be quoted as the run (this is how the first draft of this row got it wrong); and the scope is *not on the idx=1 chains the walk models*, not *on no list anywhere*. The walk is positive-controlled per sample by `mb3_freelist_selftest()` for both arming and traversal. `CONFIDENCE: LOW` on any causal role for the init. | #1226 |
| A prosper **PM4 write path** stores `0x30016000` into guest memory | `PROSPER_WRITE_TRAP=0x30016000` (checks `RELEASE_MEM` sel 1/2/3, `EVENT_WRITE`, `WRITE_DATA`, both `DMA_DATA` forms): **0 hits** across a full faulting run, positive-controlled by arming `0x1` alongside it (that control reaches ordinal 2,048 in the same run). The value is produced by the guest's own misaligned read, above. Three limits on how far this negative reaches: PM4 paths only (compute writeback, the Vulkan backend and the HLEs are outside it); the scan is 4-byte-strided from each payload base, so an *unaligned* poison inside a `DMA_DATA` **copy** is not seen; and a copy is scanned only to its first 4 KiB. | #1226, #1754 |
| A prosper GPU write targets the `0x3001600000` page the poison decodes to | `PROSPER_PROVENANCE_ADDR=0x3001600000:0x10000` reports zero overlapping writes across a full faulting run; `PROSPER_POOLSHIFT=1` is also 0. The page is an ordinary 64 KiB guest `sceKernelBatchMap` mapping, consecutive in the same series as the earlier-reported `0x30015f0000`. | #1226, #1754 |
| The renderer's fold latency (the guest outrunning our deferred label writes) is causal | `PROSPER_RENDER=0` faults at the same site with the same value, about 6 seconds in instead of about 21 seconds. | #1226, #1754 |
| Suppressing the forging fence — or both known label writes — fixes the underlying allocator corruption | The fence-only arm removes the terminal `0x30016000` and moves the fault to `0x2400100024001`, two pops farther along the same walk. The first combined run at `dfd89f3f` was **PROVISIONAL/VOID for causality**: its forge population was at least 64 but its only totals snapshot was candidate 1. After the terminal census was fixed and mutation-tested, the corrected arm at `fb3daaa4` independently reached `INIT-SUPPRESS #1024` and ended with the exact final line **`candidates=20 suppressed=20 landed=0`**. It presented frames 0–52, then faulted at the other already-known sibling site `eboot+0x117811f` with `r14=0` (`addr=(nil)`), not at the `0x30016000` pop. Thus removing both writes is not a title fix, but the valid arm settles its narrow necessity question: the specific terminal `0x2000000001 -> 0x30016000` chain does **not** survive when neither known write lands. It does not isolate init from fence, nor attribute the sibling fault, because all-mode deliberately drops live fences. Content-selective capture retained exactly two non-alpha-only images; both were inspected and are single-colour clears (frame 12 solid yellow, frame 50 solid white), with no game imagery. | #1226, #1754 |
| A resource completed by the mapped normal construction path can have `+0x68`, `+0x78`, or `+0x88` populated while its `+0x98` side-command descriptor was never created | `eboot+0x1186a40`, reached by all ten mapped resource-creation calls, uses the same populated-field condition as the allocation-page appender, allocates the 16-byte side-command descriptor, fills `{target, dword count}`, and stores it at `+0x98` before the wrapper is published. The common teardown at `eboot+0x1187300` is a confirmed later writer of null. This rules out omission by that normal completed path; it does not rule out an unlocated foreign/incomplete constructor, teardown-after-retention, or stale/reused memory. | #1226 |
| The `eboot+0x117811f` sibling fault is a **guest resource-lifetime bug** — a retained command item whose `+0x98` side-command descriptor was freed by teardown, or was never constructed, or belongs to reused memory. This framing drove the vptr-classification discriminator, the `PROSPER_HWBP_R14=0` probe, and the whole producer/teardown static map | The null `r14` is **prosper's own zero**. `[lazy-commit] mapped page=0x2100000000 rip=eboot+0x1178064` fires exactly once in the retained combined arm, at exactly the documented `mov r14,[rax+0x98]`, and the `addr=(nil)` fault at `eboot+0x117811f` is the next instruction. An independent ordinary run here reproduced the pattern at a **different** guest site: `[lazy-commit] ... rip=eboot+0x117e221` followed by `sig=11 addr=(nil) rip=eboot+0x117e225` — **+4 bytes**. A third ordinary run had **no** lazy-commit line and **no** null-pointer fault, terminating on the allocator chain instead. So: lazy-commit present → null deref within a few bytes (2/2); absent → no null deref (1/1). The reserved-range handler `mmap`s 64 KiB of anonymous zeros with `MAP_FIXED` and resumes, so a load of a pointer field returns 0. Both guest sites first-touch the **same** page `0x2100000000`, a round 4 GiB above the arena base — a wild pointer would not repeat that. Do not resume vptr classification, teardown-after-retention, or `r14`-gated probing of this site: they classify a value prosper wrote. The static producer/teardown map remains correct as disassembly; it is simply not what this fault is about. | #1944, #1226 |
| The two retained non-alpha-only frames (solid yellow, solid white) are **guest clears**, and therefore evidence that the guest reaches a clear-only phase | They are `ELIMINATE_FAST_CLEAR` artifacts. Across two independent runs — the retained combined-suppression arm and an ordinary unsuppressed run here — **every** coloured frame is presented immediately after a `CB_COLOR_CONTROL.MODE=2` draw, and **no** coloured frame occurs without one (3 for 3; the combined arm's MODE=2 count reached exactly 2 for its 2 coloured frames, the ordinary run's count reached 1 for its 1). prosper runs MODE=2 as an ordinary colour draw, which blankets the target with the fast-clear colour. Do not read either frame as a phase marker or as progress. | #1588, #1226 |
| ~~The rung-1 blocker is the terminal render-thread fault, so bring-up should continue by attributing that fault~~ — **THIS ROW IS WITHDRAWN (#2011). The fault IS the rung-1 blocker.** | The original entry read: "Every recorded terminal fault occurs **after** dozens of presented frames — 19 in an ordinary arm here, 53 in the retained combined arm — and all of those frames carry RGB 0. The fault cannot be what prevents a first frame." The inference had a hole: it never asked **why** those frames are black. They are black because ArcRunner's intro movie is black until video frame 60 (independently measured on the file with `ffmpeg`, and prosper stages it byte-correctly), and every run is killed by the fault at 43-54 delivered video frames — i.e. **before** the picture starts. Fixing #1226 does not yield "more black frames"; past frame 60 it yields picture. The row's *second* sentence stands unchanged and was always separate: prosper realizes 456 of 468 submitted draws with zero recompile-reject/compute-skip/unsupported-format lines, so the black output is not a dropped-draw problem. | #2011, #1226 |
| The black composite needs a rendering explanation at all, in the window the guest currently reaches | **Not falsified — made unnecessary and untestable here, which is a weaker statement and the honest one.** Every input to the composite is legitimately black in this window, so a rendering defect, if one existed, would be invisible rather than absent. ArcRunner plays `arcrunner_intro_a_1080p_ps5_30fps.mp4`; prosper opens it, decodes it in the real libavcodec software fallback, and stages byte-correct BT.601 black (`Y=0x10`, `U=V=0x80`, exactly `1920x1080` non-zero bytes with the pitch padding cleared, measured with `PROSPER_DUMP_RAWTEX`). The **independent** control is the file: decoded on the host with `ffmpeg`, the mp4 has mean luma 0.0 / max 0 at t=0 s and t=1 s and no picture until t=2 s. Three runs delivered 43/53/54 video frames, so every run stops before the movie's first picture. **Re-open this question once the guest reaches a frame the movie is not black in** — do not read this row as "rendering is fine". | #2011 |
| The narrow claim that content is **lost between the scene-colour target and the scanout** | Falsified on the **default live-target path**: the 4K scene-colour target `0x312aee0000` measures `rgb_nonblack=0/8294400` **and** `raw_nonzero_bytes=0/66355200` under `PROSPER_DUMP_PERSISTENT`, so nothing is present there to lose. Zero `[agc] PUBLISH DROPPED`, zero recompile-rejects and zero compute-skips across four runs, so the composite is neither dropped nor degraded. | #2011 |
| The fully-zero bloom pyramids and the "step that loses the scene colour between `0x315f4f0000` and `0x3160700000`" are a defect worth a capture bundle (the previous discriminator 1) | A black scene thresholds to zero bloom; that is UE4 behaving correctly, not content being dropped. The 4K scene-colour target measures `raw_nonzero_bytes=0/66355200` on the **default live-target path** (`PROSPER_DUMP_PERSISTENT`), which is what an empty scene behind a black movie frame looks like. Do not spend a bundle on this until the guest reaches a frame the movie is not black in. | #2011 |
| The magenta 1x1 auto-exposure targets (the previous discriminator 2) are garbage feeding a tonemapper (`(255,92,255)`) | Not a defect worth chasing: those targets are `1x1 VK_FORMAT_R32G32B32A32_SFLOAT` (`resolved-format=109`), and a 16-byte float texel converted for display is not readable as an exposure value. With the scene legitimately black there is nothing for the tonemapper to lose. | #2011 |
| The per-target readback table in this document describes prosper's **default** rendering path | It does not. It was taken with `PROSPER_DUMP_RTGROUPS`, and `frontends/shared/live_renderer.cpp:1008-1015` lists that variable (with `_RGBA`, `PROSPER_DUMP_DRAWSTEPS`, `PROSPER_DUMP_SAMPLED_RTT`, `PROSPER_RTTLOG`, `PROSPER_RESOURCE_HASH_DIM`, `PROSPER_TARGET_STEP_HASH_DIM`, `PROSPER_GPU_REPLAY_*`) in the `live_gpu_targets` **disable** list — so the whole table describes the CPU-readback path. `PROSPER_DUMP_PERSISTENT` is the census that does observe the normal persistent-GPU-target path, and it agrees on the black scanout. Re-take any localisation on that variable before building on it. | #2011 |
| The first three current-master ordinary diagnostic arms prove the sibling absent or identify its producer | The first title run exited through an unrelated `AudioMixerRende` null jump. The second lost to the usual allocator chain at `eboot+0x127e8eb`, with the guest reporting `Attempt to realloc an unrecognized block 2000000001`. The third used the address-filtered hardware probe, which armed on primary and worker guest boundaries, but lost to the allocator predecessor at `eboot+0x127e751` before recording any target hit. None reached `eboot+0x117811f`; the first two stack peeks sampled unrelated frames and the third correctly emitted no probe. All three arms are **void/non-discriminating**, not negative reproductions. A nearby D-queue unsatisfied wait is retained as co-occurring history only; it does not attribute any competing failure. | #1226 |
| A prosper GPU write lands on the guest's pointer table slot at `0x2020e381c0` — the co-location lead that made "which writer touches that 8-byte slot" the open question | `PROSPER_PROVENANCE_ADDR=0x2020e30000:0x10000` over a full faulting run records **zero** writes overlapping `0x2020e381c0`, with **1,314** writes elsewhere in the same 64 KiB page as the in-run positive control. Every one of those 1,314 is `kind=dma-data size=4` at a 32-byte-aligned address (282 distinct slots) — so the **`DMA_DATA` copy** candidate named as the other half of that question is also empty here: the population is 4-byte **immediates**, not copies. The `0x2020e3xxxx` page is a 0x20-byte label pool, and the table shares it. | #1226 |
| The three terminal faults (`0x2000000001`, `0x2100000001`, `0x400000001`) are three racing causes that need separate attribution | One composition rule, performed by prosper. The label ring at the faulting register shows the pointer that was in the destination, our 4-byte immediate-zero init clearing its low dword, and our paired 4-byte value-1 fence setting that dword to 1 — measured for a module-image vptr (`0x41700f1e8` → `0x400000001`, with the guest's `mov rax,[rdi]; call [rax+0x20]` at `eboot+0x32b61be` disassembled) and twice for live heap pointers (`0x21c1388890` and `0x21c0e182d0` → `0x2100000001`). See the 2026-08-06 section. | #1226 |
| prosper's label writes are excluded as the author of `0x2100000001` / `0x400000001` (recorded from `FORGE-TRIP-TOTALS seen=256 narrow=256 wide_only=0` and `SUSPECT-REL1-LIVE=0`, both levers witnessed) | **Withdrawn.** Those censuses are correct and structurally blind to this population: every shape predicate filters `pre` through `ptr_like()`/`heap_ptr_like()`, whose widest window floor is `0x1000000000`, and a vtable pointer lives in the module image at `0x41xxxxxxx`. `PROSPER_PTRLIKE_WIDE=1` shares that floor, so it does not reach it either. The windowless `PROSPER_LIVEPTR_TRIP` census reports `examined=5091 shape=1280 live=1280` in one run. A negative from a predicate that cannot represent the case is not a negative. | #1226 |
| The `AudioMixerRende` `rip=0x0` jump is an unrelated audio failure, retained only as a competing terminal path | It is the same defect. Census of nine runs: **4** `AudioMixerRende`, **3** `RHIThread`, **1** `AgcCleanupThrea`, and **1** whose log is truncated mid-line and carries no `[fault] thread=` record — 4+3+1+1, where an earlier revision of this row listed only eight. Restricting to the **six default (unarmed) runs**, `AudioMixerRende` is 4 of 6. `rax=0x400000001` is the object's **vptr**, read by `mov rax,[rdi]` one instruction before the faulting `call [rax+0x20]`. **Attribution to prosper's two writes is witnessed for ONE of the four** — the run whose `[labelhist]` ring records `dmaX(D):0x41700f1e8` then `relX(D):0x400000000`. For the other three it rests on value shape, and the `{small nonzero high dword, low dword ≤ 1}` row below says in terms that value shape alone cannot attribute such a qword; those three are *consistent with* the mechanism, not instances of it. | #1226 |
| A rung-1 arm can be bounded like the fault arms, because the fault is what stops the movie | The fault is necessary but not the whole arithmetic. `sceAvPlayerGetVideoDataEx` advances the movie one frame per guest call (`next_video()` pops the decode queue with no media-clock test), and a bounded run measured **31 calls / 31 successes** with **no** other AvPlayer call — so the queue is never starved and the rate is the guest's poll rate, ~2.6 Hz. Reaching video frame 150 needs ~150 polls, about a minute of wall time. Count `video-ex` calls, never seconds. | #1226 |
| `PROSPER_NONHEAP_PTR_GUARD=1` (decline a sub-qword label write over a mapped non-heap pointer) does or does not remove the fault | Neither, **on those three runs**. They recorded `declines=0` because the image-vptr class did not occur in them, so that arm is **non-discriminating**, exactly like the three void arms above. Do not read this row as the current state: the max-guard row below witnesses **1** `NONHEAP-PTR-DECLINE` — on the same vtable `0x41700f1e8` — and the run still faulted. The lever announces itself and counts, so this is a readable null rather than a silent one. | #1226 |
| prosper's **deferred** completion-write model (#312's post-submit worker) is what makes a label write land in memory the guest has reallocated — so `PROSPER_EOP_WRITE_SYNC=1` should remove it | It does not. Five armed runs with the lever now witnessed (`[agc] EOP-WRITE-SYNC ARMED`, added for this): the live-pointer stomp rate is **25.9–26.5%** of examined sub-qword writes against the default's **25.1%** (both measured before `liveptr_trip`'s WRITE_DATA call moved inside the `wd_num <= 4` branch, which narrowed the `examined` denominator by excluding packets that could never have matched — a re-run at a later head will not reproduce these exact percentages, and that is a denominator change, not a subject change), and the module-image vptr class occurs in **4 of 5** armed runs against 1 of 6 default runs. Read those two facts separately. The **aggregate rate is unmoved**, which is the result. The class-frequency difference is large and in the *opposite* direction from a null, and it is **unexplained at n = 5/6** — it is as consistent with the synchronous lever increasing exposure to the class as with small-sample noise, and it must not be folded into the null as though it were merely a positive control. What it does establish is that the class occurred inside the armed arm, so the arm is not a silent one. The cleanest hit has `events(total=3): dmaB@8408/f31 relB@8408/f31 waitB@8408/f31` and executes at 8745 ms, i.e. the guest named the block as a label and freed/reallocated it 337 ms later, before the referencing submit. Synchronous writes cannot help: the build→submit gap is the guest's own. | #1226 |
| Suppressing the label writes prosper can detect is enough to get the title past the fault | It is not. The maximal arm `PROSPER_MB3_FREELIST_GUARD=1 PROSPER_GENERATION_GUARD=1 PROSPER_REL1_WAF_GUARD=1 PROSPER_NONHEAP_PTR_GUARD=1`, with every lever independently witnessed (**26** `MB3-` suppressions, **3–10** `REL-GENERATION-CHANGED-STALE-SUPPRESS`, **1** `NONHEAP-PTR-DECLINE`), still faults on `RHIThread` in the same window and still delivers **31** video frames with 31 successes — the same as an unguarded run. The label writes compose the fault *value*; they are not the whole blocker. A fix has to address the block-lifetime seam. | #1226 |
| The rung-1 blocker is a rendering, recompiler, AvPlayer or composite defect somewhere in prosper's graphics path | It is none of those. With `PROSPER_SUBMIT_STALL_US=1500` the title renders its **entire intro cinematic** in 4K — a nebula and the ringed station captioned *TITAN-CLASS SPACE STATION "THE ARC"*, a rainy neon street with a character and reflections, and a *POPULATION: 10 MIL* text card — with **0 of 4** stalled runs faulting against **17 of 17** default runs, 1,901 of 1,908 `GetVideoDataEx` calls succeeding against 31, and ~26.4 M of 33.2 M bytes nonzero per presented frame against RGB 0. Frames opened, not just measured: `assets/screenshots/arcrunner-intro-space-station.png`, `assets/screenshots/arcrunner-intro-city.png`. Every graphics subsystem needed to produce this cinematic runs — geometry, text, composition and presentation are all correct — and the blocker is a submit-timing race. Not a claim that the graphics path is defect-free: the same frames carry a chroma-plane colour fault (#2085). | #1226, #1945 |
| The ArcRunner corruption is title-specific, so it needs a title-specific fix | The same lever gives the same answer on **Crisis Core** (`PPSA07809`), whose dose-response is 0/3 at no stall, 0/3 at 500 us, 3/3 at 1500 us, 3/3 at 3000 us. **The shared finding was originally the direction only, because ArcRunner had been run at 1500 and 3000 us alone — that caveat is now retired: this title's own twelve arms give 0/3, 0/3, 3/3, 3/3 at the same four doses, so its bracket is `(500, 1500]`, identical to Crisis Core's** (#2084; see § *2026-08-06: the submit-duration dose-response*). Pend-queue residency on Crisis Core peaks at 3 ms, so "our completion writes land late" is false on both titles. Two titles, one lever, one measured threshold: this is a property of prosper's submit timing, not of either guest. It also agrees with this document's own `PROSPER_EOP_WRITE_SYNC` null. It is still not a general law — a third title owes its own dose-response. | #1945, #1894, #1226, #2084 |
| ArcRunner's `(500, 1500]` threshold is Crisis Core's, imported rather than reproduced, and must be treated as an untested assumption until someone runs the 500 us arm here | The 500 us arm was run: **0 of 3 survived**, all three faulting in the already-recorded family (two at `0x30016000`, one at `rip=0x0`) at 11.6–17.9 s with 31–37 video frames. With 3/3 surviving at 1500 us, this title's bracket is measured, not inherited. Twelve arms, four doses, one binary from `f080fc23`, doses interleaved so load drift cannot align with dose, no passive observer armed (instrument trap 104), and a peer-process census of zero before and after each arm. | #2084, #1226 |
| ~~The throttle rescues the title by giving prosper time to catch up inside the guest's build-to-submit interval~~ — **NOT ESTABLISHED; this row is withdrawn as filed** | The first revision claimed the opposite (that the interval is *longer* when the arm survives) from a pooled sample of every ring-bearing diagnostic line. That sample is **selected for the condition under test**: rings are only printed inside diagnostics that embed `label_hist_report`, and a dying arm's pairs come almost entirely from its 11–27 `SUSPECT-REL1-OVERLAP` lines, a population a surviving arm has **zero** of. Restricted to the `WaitRegMem` rings both classes have, the medians are 282–413 ms over **29** pairs in the dying arms against 347–584 ms over **363** in the surviving ones — not shorter, but 2–8 pairs per dying arm cannot falsify anything. What stands is only the absence of evidence for the shortening story. `CONFIDENCE: LOW`; needs a per-fold instrument. Caught in review of #2091, not by the author. | #2084, #1226 |
| prosper's deferred-stream barrier model (#312) is in the default path, so the build-to-exec gap on this title is prosper holding the packet | `PROSPER_WAIT_DEFER` is **opt-in and default OFF** (`src/gpu/command_processor.cpp`). On the default route prosper barrels through an unsatisfied `WAIT_REG_MEM` with the "dependency violated" log and defers nothing — confirmed directly in all twelve dose arms, where the `dependency violated` count equals the unsatisfied-wait count exactly (3/3, 4/4, 2/2, 1/1, 2/2, 4/4, 40/40, 36/36, 28/28, 40/40, 40/40, 40/40) and the barrier model's own marker `pausing queue (deferred effects)` appears **0 times in all twelve**, against 1 in each of the three `PROSPER_WAIT_DEFER=1` arms. The ~250–580 ms build-to-exec gap is therefore entirely the **guest's own** build-ahead. | #2084, #1226 |
| Honouring the guest's `WAIT_REG_MEM` barriers instead of barrelling through them fixes the title — the obvious non-throttle candidate, and the one #312's original 5/5 evidence points at | `PROSPER_WAIT_DEFER=1`, three arms on the same binary and route with no throttle: **3 of 3 faulted**, at 11.4/13.6/11.4 s with 61/34/35 video frames, all in the `addr=(nil) rip=0x0` `AudioMixerRende` class, with `SUSPECT-REL1-OVERLAP` still reaching ordinals 25 and 22. The lever is witnessed rather than assumed — every arm logs `WaitRegMem … — pausing queue (deferred effects)` and one logs `DEFER TIMEOUT #1 after 1000ms`, while no dose arm carries the `pausing queue (deferred effects)` marker at all — 0 of 12 against 3 of 3. Do not substitute a bare `defer` grep for that marker: the six surviving dose arms each contain one unrelated `layered image deferred to #657` line. This reproduces on ArcRunner the verdict `command_processor.cpp` already records from ~20 DOLL runs: the model removes the ordering-violation leg and a wait-order-independent leg dominates, and deferral latency makes the title die sooner. | #2084, #312, #1226 |
| The zero `recompile-reject` / compute-skip / unsupported-format census recorded in the sections above describes this title's shader coverage | It describes the **first 8–14 seconds**, which is all any of those runs got. Every one of the six surviving dose arms reaches `[compute] skip unsupported program` for three programs reproducibly (`0x3005330000`, `0x3007780000`, `0x30094d0000`), plus a run-varying fourth, plus one `layered image deferred to #657 -> dispatch skipped (#590)` — all past the point where a default run dies. Generalise before quoting any "zero X across N runs" figure on a title whose runs end in a fault: a coverage census is bounded by how far the run got. | #2090, #2084, #1226 |
| `PROSPER_AVP_LOG=1` enables the AvPlayer log, so a run with it set and zero `video-ex` lines means the movie never started | `avp_log()` reads **`PROSPER_AVPLOG`** (`src/hle/hle_service.cpp:978`), not `PROSPER_AVP_LOG`. A max-guard arm here reported `video-ex calls: 0` purely because the wrong switch was passed; re-run with `PROSPER_AVPLOG=1` it reported **31**. Two arms were void this way before it was caught. Pass `PROSPER_AVPLOG=1`. | #1226 |
| `PROSPER_PRESENT_NZLOG=1` on its own reports presented-frame content under `boot_trace` | It reports nothing. A run with it set produced **zero** `[render-nz]` lines over ~100 presented frames: the line is gated on `!px.empty()` in `frontends/shared/live_renderer.cpp`, `px` comes from `selected_pixels`, and the registrar announced `dump=0` — the readback that fills it is opt-in and this switch does not request it. A silent run is **not** "every frame was black". | #1226 |
| The `addr=(nil)` faults show that page `0x2100000000` is a **legitimately committed guest region whose contents prosper lost** — [#1944](https://github.com/mattias800/prosper/issues/1944) reading 1, argued from "two *different* guest code sites first-touch the same 64 KiB page; a wild pointer would not repeat like that" | It is reading 2 (a wild read masked), and the repetition is a property of the value, not the mapping. `PROSPER_LAZY_COMMIT_STRICT=1` moves the fault to the loading instruction: `[lazy-commit] #1 DECLINED(strict) page=0x2100000000 addr=0x2100000041 access=read rip=0x41117e221`, then `sig=11 addr=0x2100000041 rip=eboot+0x117e221`. The load is `mov rdi,[r12+rcx*8]` / `mov rcx,[rdi+0x40]` with `rdi=0x2100000001`, and `0x40` is added **without masking the low bit**, so `rdi` is a plain corrupt pointer. **Any** `0x21000000xx` value lands in page `0x2100000000` — which is exactly the shape a heap pointer takes when its low dword is lost — so both recorded sites hitting that page is expected, not anomalous. Exactly one lazy-commit event occurs per affected run. Do not spend another arm treating the page as a commit-protocol gap on this title. | #1944, #1226 |
| The `0x2100000001` that the terminal `addr=(nil)` fault dereferences is composed by prosper's own `RELEASE_MEM`/`WRITE_DATA` fence, the way `0x2000000001` is — and `ptr_like()`'s stale upper bound (exactly `0x2100000000`) is hiding that population from the tripwire and from both guards | The blind spot is **real but empty on this title**, on both branches. With the tripwire's report predicate widened to prosper's whole guest-VA window, one bounded run on `c9e2588e` ends at `FORGE-TRIP-TOTALS seen=256 narrow=256 wide_only=0`: none of the 256 reported candidates is wide-only, every one has `pre=0x2000000000`, and the absence of a `#512` totals line on the dense every-256 schedule bounds the run's population below 512. The sibling `REL1-LIVE` branch — a fence over a pre whose low dword is a real pointer half, which is what would turn a live `0x2100e05140` into exactly `0x2100000001` — was checked separately with `PROSPER_PTRLIKE_WIDE=1`, which arms both guards over the wide window and prints `PTRLIKE-WIDE ARMED` so the lever is witnessed: two runs, `SUSPECT-REL1-LIVE` count **0**, terminal fault unchanged. So prosper's label writes are excluded as the author of the value that terminates the run. Two limits: `report_suspect_write()` has emitted no line of any kind on this title across nine runs, so the `REL1-LIVE` zero has no in-run positive control (it is consistent with the REL1 population being homogeneous, not independently proven); and this does **not** clear the narrow window for other titles — it is still stale, and the lever exists to A/B flipping it. | #1226 |
| `{small nonzero high dword, low dword ≤ 1}` is by itself a prosper-forge signature, so a value of that shape found in guest memory attributes the write to us | It is also **ordinary guest data**. A fault-time dump of a live guest table (`PROSPER_FAULTMEM=1`, register `r12`) shows 16-byte `{pointer, metadata}` entries whose metadata qword is a constant `0x0000000400000002`, interleaved with live pointers `0x2000e03840` / `0x2000e03830` / `0x2100e05140`. `0x400000001` — the value at the `AudioMixerRende` `rip=0x0` jump, and the one #1945's brief quotes — has that same `{4, n}` shape. Value-shape alone therefore cannot attribute a `<high>_0000000n` qword to a GPU write; only a write-side record (the attribution ring, or the forge tripwire's `pre`) can. | #1226, #1945 |
| A one-run-per-arm comparison can A/B a candidate fix on this title | **Three terminal paths compete for every run** and which one wins is a race: measured on `c9e2588e`, the allocator chain (`addr=0x30016000 rip=eboot+0x127e751`), the lazy-commit-masked null (`rip=eboot+0x117e225`, one lazy-commit event), and an `AudioMixerRende` jump to `rip=0x0` with `rax=0x400000001` each terminated at least one of six runs, and one run produced two faults. Consequently the `PROSPER_EOP_WRITE_SYNC=1` and `PROSPER_REL1_WAF_GUARD=1` arms run here are **non-discriminating, not negative**: each was a single run, each ended on the same `AudioMixerRende` null jump as its baseline at ~16 presented frames. Any future arm needs ≥3 runs and a quantitative progress metric (presented frames, delivered video frames, time to first fault), not the identity of the terminal fault. | #1226 |
| The `PROSPER_MB3WATCH` per-thread head watch reports nothing, so the bin head is not being stomped | A DIFFERENT instrument from the `ptr_like()` upper bound two rows above — that one is `command_processor.cpp`'s report predicate, this one is the watchpoint's arm hook, and widening either says nothing about the other. The arm hook (`exec_image_linux.cpp`) filtered on a stale DOLL-era `[0x20_0000_0000, 0x21_0000_0000)` window while every current title's per-thread cache base is `0x30_xxxx_xxxx` — ArcRunner's is `0x3152350000`, Crisis Core's `0x3001af0000`. It therefore **armed nothing at all**, printed nothing, and read exactly like "armed and saw no write" — the same class as the #1998 finding for `PROSPER_WATCH_LABEL`/`PROSPER_WATCH_HOT`. Window widened to `[0x20..0x40)`, and its report trigger widened from the byte-shift value shape to the structural "a head must be 0 or a 0x20-aligned mapped node". Any null quoted from this instrument before 2026-08-06 is **void, not negative**. | #1945, #1998 |
| Both prosper-authored halves of the forged pointer are jointly necessary for the corruption (the open question left by the *Suppressing the forging fence* row above; that ArcRunner arm was read as settling it) | Run on **Crisis Core** (`PPSA07809`), a cheaper reproducer for the same family, with the same lever and a valid census (`candidates=127 suppressed=127 landed=0`, `INIT-SUPPRESS` past #4096): the guest allocator is **still** corrupted and still faults at its bundle-list pop, with `0x0002400100024001` in the bin head. Separately, the "our completion writes land late in a recycled block" premise underneath this whole line was measured on that title and is **false**: pend-queue residency peaked at **3 ms** over 5,632 writes, none above 20 ms (`PROSPER_PEND_AGE`). What *does* decide the outcome there is the DURATION of the guest's own submit call, measured as a dose-response over a plain `nanosleep` in the submit fold: **0/3 survive at no stall, 0/3 at 500 us, 3/3 at 1500 us, 3/3 at 3000 us** (0/6 vs 6/6, Fisher two-tailed p ≈ 0.002), threshold between 500 and 1500 us. That is the controlled form of the timing-not-code conclusion `OREGON_TRAIL_STATUS.md` reached from nine arms of PPSA19244. | #1945, #1894 |
| The DMA-init generation question is **blocked** because #1756 removed the packet slot `dma_build_pre_changed()` reads — recorded as the concrete next step and as unreachable in the same paragraph | It was never blocked. The build snapshot the check needs already existed **out of band**: `prosper_fence_journal_record()` stores a per-packet build-time target and content, keyed by the packet's guest address, and the `RELEASE_MEM` / `WRITE_DATA` / `WAIT_REG_MEM` legs all called it — the DMA-init builder was the only one that did not. Adding that one call, plus `PROSPER_DMA_INIT_GEN=1` to read it back at exec time, answers both halves (generation depth and build→exec drift) with no packet-format change and no content predicate. Generalise: before recording a question as blocked on a removed field, check whether a sibling path already records the same fact for its own reasons. | #1226, #2084 |
| prosper's completion writes land late on **ArcRunner**, so the deferred pend queue is what puts a label write into a recycled block (the premise the Crisis Core measurement left open for this title) | `PROSPER_PEND_AGE=1` over a full default faulting run: **peak residency 5 ms** across 4,096 applied completion writes. The build→exec age measured in the same arm is ~345 ms mean and ~840 ms max, so the deferred model contributes about **1.5%** of the window and the remaining 98.5% is the **guest's own build-to-submit interval**. The 3 ms Crisis Core figure therefore reproduces here, on a second title, and `PROSPER_EOP_WRITE_SYNC` is closed as a lead for a second independent reason. | #1226, #1945 |
| Declining the label init whose target **drifted** between the guest's build and prosper's exec fixes the title — the check `dma_build_pre_changed()` was written for, re-armed out of band | It does not, at either level. `PROSPER_DMA_INIT_DRIFT_GUARD=1` (init only): **3 of 3 faulted**, delivering **34 / 35 / 35** video frames against seven baseline runs' **31–37**, with the lever witnessed on every arm (**21 / 13 / 15** `DMA-INIT-DRIFT-DECLINE` lines) — a readable null, not the silent-non-arm class. `=2` (init **and** its paired fence): **5 of 5 faulted**, at **31 / 39 / 147 / 203 / 509** video frames — four of five above every baseline run, so this is the first intervention here that moves the progress metric, and it still does not remove the fault. The predicate is sound and specific (14–26 drifts per dying run against **0 in 10,079** journal-matched inits in a surviving throttled arm), which makes the null stronger rather than weaker: the writes prosper can detect and decline are the messenger. | #1226 |
| Declining a drifted init is enough on its own, because the default-ON `rel1_stomp_guard()` will decline the paired fence once the low dword is no longer zeroed | It will not, and the arm that assumed it named the reason at its own terminal fault: the surviving pre-content was `0x21c0f88b70`, which is above `ptr_like_narrow()`'s `[0x2000000000, 0x2100000000)` ceiling, so the fence guard could not represent it and wrote `1` over the pointer's low dword — producing exactly the `0x2100000001` in `rdi` at the fault. **Declining half a pair is worse than declining neither**: the 4-byte value-1 fence composes the `{stale high dword, 1}` forge on its own, with no help from the init. Any future arm that suppresses one half must carry the decision across the pair explicitly (`PROSPER_DMA_INIT_DRIFT_GUARD=2` does, through the dma-free debt). | #1226 |
| The submit throttle changes **how fast the guest runs** — either slowing it down (the reading its name invites, and every "the guest outruns prosper" framing built on it) or, as the first revision of this row claimed, speeding it up about sixfold | **Neither. It changes only how long the run lives.** The sixfold claim was a denominator artifact and is retracted here rather than merely softened: both figures behind it were totals divided by **whole-run** duration, and both arms spend an identical **~6.8 s** boot prefix before the first `sceAvPlayerGetVideoDataEx` — which dilutes a 9-second arm's average and barely touches a 59-second one's. Time-to-first-video-frame is **6.77 / 6.73 / 7.00 s** default against **6.82 / 6.81 s** throttled, i.e. the same. Over the movie window itself the rates are indistinguishable and the **default** arms are marginally *faster*: **27.2 / 27.4 / 28.5** video frames/s against **26.3 / 26.3**, and **437 / 440 / 434** label-init executions/s against **414 / 406**. Both decode at about the movie's nominal 30 fps. The build→exec lag is also unchanged (~345 ms mean either way), so the throttle is not letting prosper catch up either. The lever's entire observable effect on this title is **survival** — a tighter constraint on any remaining hypothesis than the inversion it replaces. Two consequences: a rate quoted over whole-run duration on a title whose arms differ in lifetime by 6x is not a rate, and this document's own "~2.6 Hz" poll figure has the same defect (true rate ~27 Hz). | #1226, #2084 |

## Next discriminators

Ordered for **rung 1** (any real graphics), which the terminal faults do not block.

1. **Fix #1226's terminal fault — that is now the whole of rung 1.** The movie needs 60 delivered
   video frames for its first (barely) non-black frame and 150 for full picture; every run is killed
   by the fault at 43-54. Nothing else stands between this title and real visible graphics, and the
   ownership question in #1945 is the live part of it. Note this **reverses** the standing guidance
   that redirected the lane away from the fault — see the withdrawn `## Ruled out` row.
2. ~~Re-check the movie surface extent after the `AvPlayerVideoEx` crop fix.~~ **CONFIRMED on `b97d0bb3`
   — do not re-run this.** The measurement and its numbers are in
   § *The movie surface extent, measured before and after the `AvPlayerVideoEx` fix* above: the
   `1792x1080` movie surfaces go to **zero** and `1920x1080` ones appear, so the guest does use the
   `width`-based spelling and the coded-extent contract is the right one. Nothing further is owed here.
3. **Resolve #1944 before spending any further arm on an `addr=(nil)` fault.** Add the default-off
   fail-visible lazy-commit mode described there, so the fault reports at the *loading* instruction with
   the real faulting address, and decide whether page `0x2100000000` is a legitimately committed guest
   region whose contents prosper lost, or a wild read the handler is masking. Until then, no null-pointer
   fault site in this title should be treated as guest evidence.
4. Establish whether the `mode=0` depth prepass into `0x310fea0000` should have depth writes enabled, and
   if so whether prosper's decode or the guest drops them. See the depth paragraph above; `CONFIDENCE: LOW`
   on any role in the black output.
5. Isolate the init and REL1 fence interventions only with arms whose independent lever witnesses and
   terminal populations are complete. Do not infer authorship from a moved terminal fault alone.
6. Revisit the per-queue barrier model and intro-movie path from #1226 only after checking their
   current master reproductions; neither is settled by the allocator arm.
7. Capture and inspect a real game image before advancing the tracker or adding an ArcRunner
   screenshot to `COMPATIBILITY.md`. A frame presented right after a `MODE=2` draw does not qualify —
   see the `## Ruled out` row.

**Where the terminal-fault hunt stands after 2026-08-06** (added without renumbering the list above,
which a concurrent lane also edits — the paragraph below it is the previous standing summary and is
now superseded by this one):

The **author** of every terminal value is settled: prosper's own 4-byte immediate-zero label init,
followed by its paired 4-byte value-1 fence, over a destination that still held a live pointer. Three
instances are measured at the faulting register, one of them a module-image vtable pointer with the
guest's faulting virtual call disassembled. What remains is **ownership**, not authorship:

1. **Why is a built label packet still pointing at a block the guest has reallocated?** The label
   ring shows the same address rebuilt every few frames (`dmaB@11450/f94` → `dmaX@11786/f100`, 336 ms
   and six frames apart) and, at the last generation, holding an object. The existing content-free
   answer to exactly this — `dma_build_pre_changed()`, which compares the destination at build time
   against exec time — is **inert on current builds**: #1756 shrank `DMA_DATA` to its hardware
   7 dwords and removed the slot that carried the build snapshot, and the guard says so once when
   armed. Restoring a build-time snapshot out of band (the label ring already records the init event
   via `prosper_label_hist_dma_built`) is the concrete next step, and it needs no content predicate.
2. **The heap half cannot be closed by content and should not be attempted.** Stale `NextFreeBlock`
   residue in a popped label block is byte-identical to a live object pointer; the `hi=0x20:1280`
   bucket is a free-list walk (each write's `dst` is the previous write's `pre`). Suppressing it is
   #1245 again at 1,280 events per run.
3. **The module-image half is safely separable** and `PROSPER_NONHEAP_PTR_GUARD=1` exists for it, but
   it needs runs in which the class actually occurs — three armed runs declined nothing.
4. **Give a rung-1 arm a long bound and count `video-ex` calls.** See the `## Ruled out` row.

**Added 2026-08-06 by the #2084 dose-response lane, without renumbering the list above:**

5. **The threshold is measured on this title now — `(500, 1500]`, the same bracket as Crisis Core** —
   and the two cheapest explanations of *why* the throttle works are both dead. It does not shorten the
   guest's barriers — `PROSPER_WAIT_DEFER=1` faults 3/3, sooner, with the lever witnessed, and that one
   is in `## Ruled out`. The build-to-submit-interval story is **still open**: the measurement that
   looked like it settled it was drawn from a population only dying arms have, and its row is
   withdrawn. Re-ask it with a per-fold instrument, not with whatever rings a diagnostic prints.
6. **`SUSPECT-REL1-OVERLAP` is the sharpest live discriminator this title has**, and it is free — a
   default build prints it. Population 11/22/27/24 in the four arms that die at `0x30016000`, **0** in
   the two that die at `rip=0x0`, and **0** in all six that survive, with the counters it differences
   positively controlled inside the surviving arms. It says the fault happens when prosper executes a
   fence for label generation *k* while the guest has already built generation *k+1*. Any candidate fix
   should be scored against this number, not only against survival: it separates the two terminal
   classes, which survival does not.
7. **Do not quote a "zero X across N runs" coverage census on this title without saying how far the
   runs got.** Six surviving arms reach three `[compute] skip unsupported program` lines that no
   14-second run can see (#2090).

**Added 2026-08-06 by the arc6 lane, without renumbering the list above:**

8. **#1226 is now the whole of rung 2, not merely rung 1.** The throttled route reaches the title
   screen and holds it for 176 s (§ *2026-08-06 (arc6)*), so the question "what is behind the intro
   movie" is answered and nothing else is known to be in the way. Score a candidate fix against
   reaching that screen on the **default** route.
9. **The rescue is the delay — stop looking for a lock, an ordering, or a queue-interleaving story.**
   Moving the whole stall out from under `g_agc_state_mu` costs nothing (0/3 either way, with an
   unthrottled 3/3 control on the same binary). Combined with the withdrawn "it changes how fast the
   guest runs" row, what survives is narrow and awkward: the lever's only observable effect is
   survival, it is not the mutex, and it is not the guest's rate. The next instrument owes a
   **per-fold** account of what 1.5 ms of wall time per submit changes — not another whole-run rate,
   which has now produced two retracted rows on this title.

**Added 2026-08-07 by the arc7 lane, without renumbering the list above:**

10. **The next step is a cross-title snapshot pass with the `version >= 13` gate on post-submit
    visibility removed** (`agc_reg_defaults.cpp`), tracked as
    [#2220](https://github.com/mattias800/prosper/issues/2220). That single condition is what stands between this
    title and rung 2: with the model forced on, the default route survives 3 of 3 with no throttle
    and delivers 782–838 video frames (§ *2026-08-07 (arc7)* §5). Removing it changes behaviour for
    every pre-13 title in the matrix, and the gate's own provenance carries no recorded evidence, so
    it needs the matrix and a review — not another ArcRunner arm.
11. **`tools/screenshot` is NOT interchangeable with `boot_trace` on this lever.** Two 300 s
    `screenshot` runs with `PROSPER_POST_SUBMIT_VISIBILITY=1` and no throttle died at 35 and 42
    delivered video frames, both `AgcCleanupThrea` at `rip=0x5c00048e0`, *before* their first sample
    interval elapsed — so it is not the 4K readback. The same lever on the same head survives 3 of 3
    under `boot_trace`. Establish which frontend a claim is about before quoting it; filed as
    [#2217](https://github.com/mattias800/prosper/issues/2217).
12. **Score a candidate against `REBUILD-BEFORE-EXEC`, not only against survival.** It is the
    build-side twin of `SUSPECT-REL1-OVERLAP` and of DMA-INIT-GEN's `depth>=2`, all three report the
    same number in every default arm, and it fires at the guest's own action rather than at one of
    prosper's — so it says whether a change moved the *guest's* margin or only prosper's writes.

*Superseded — the previous summary, kept because its provenance census is still the record:*

The `addr=(nil)` class is now attributed down to one slot. The guest loads
`rdi = table[index]` from a pointer table at `r12 = 0x2020e381c0` and gets `0x2100000001` — a
`0x21xxxxxxxx` heap pointer that lost its low dword. **prosper's ReleaseMem/WriteData label writes are
excluded as the author** (both guard branches, lever witnessed — see `## Ruled out`), so the next
question is not "which label write" but **which writer touches that 8-byte slot at all**.

The co-location is the lead: prosper's fence destinations on this title are `0x2020e35f40 …
0x2020e3d540`, and the guest's live pointer table is at `0x2020e381c0` — **the same 64 KiB page**. So
the guest is holding a live object-pointer table inside the block region it also allocates its 0x20-byte
GPU labels from. Two writers that have never been censused against that exact span:

- `DMA_DATA` **copies** (`dd_bytes` > 4). `forge_trip` covers `REL1` and `WRITE_DATA` only, and
  `PROSPER_WRITE_TRAP`'s scan is 4-byte-strided from the payload base and stops at the first 4 KiB of
  a copy, so an unaligned or late poison inside a copy is a documented blind spot.
- The guest itself, writing through a stale index. `bextr ecx, eax, 0x1008` takes the index from a
  packed command dword, so a corrupt *stream* (`r13`) produces an out-of-range `rcx` and reads past
  the table — which would make the table innocent and move the hunt one level up.

`PROSPER_PROVENANCE_ADDR=0x2020e30000:0x10000` (whole-page overlap census) discriminates the first
against "no prosper write lands there at all"; the second needs the table's own extent, which is
recoverable from the allocation-page builder already mapped in `## Current state`. Either way,
**do not spend another arm on the label-write path** — it is now excluded with a witnessed lever.
