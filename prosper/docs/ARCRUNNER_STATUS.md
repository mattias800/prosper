# ArcRunner (`PPSA21406`) — status and evidence

Unreal Engine 4.27-plus. **Rung 0 — renderer bring-up, with no real visible game graphics yet.**
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
(`PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS= PROSPER_RENDER=1`), each with exact zero pre/post process
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
(`PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS= PROSPER_RENDER=1`), three runs, exact zero pre/post process
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
(`PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_COLORSTATETRACE=all`), exact zero
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
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS= PROSPER_RENDER=1 \
  ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0
```

The rung-1 census arms above are, in order, the realization check, the per-draw colour/depth state
census, and the rendered-target content readback. Write every artifact under `~/`, never `/tmp` or
`/var/tmp`:

```bash
# 1. submitted-vs-realized draws, plus any recompile/skip lines
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_DRAWLOG=1 \
  ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0

# 2. one raw-to-resolved colour/depth record per draw (runs BEFORE the no-effect fast path,
#    so a dropped draw is still visible)
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_COLORSTATETRACE=all \
PROSPER_PRESENT_NZLOG=1 ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0

# 3. a BMP per rendered target group with >=1 nonzero byte; a DRAWN target with NO file is
#    fully zero. ~1 GB per 14 s -- keep the run short and delete afterwards.
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS= PROSPER_RENDER=1 PROSPER_DUMP_RTGROUPS=1 \
PROSPER_FRAME_DIR=~/arc-work/rtt ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0
```

The combined diagnostic arm used for the most recent narrow experiment is:

```bash
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS= PROSPER_RENDER=1 \
PROSPER_INIT_SUPPRESS=ptr PROSPER_REL1_FORGE_SUPPRESS_ALL=1 \
PROSPER_FORGE_TRIP=1 PROSPER_PRESENT_NZLOG=1 \
  ./build-linux/boot_trace <DUMP_ROOT>/PPSA21406-app0
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
(`PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS= PROSPER_RENDER=1`), with the new `PROSPER_LAZY_COMMIT_STRICT`
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

## Ruled out

Do not re-derive these without contradictory new evidence.

| Hypothesis | Evidence that killed it | Ref |
| --- | --- | --- |
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
| The `addr=(nil)` faults show that page `0x2100000000` is a **legitimately committed guest region whose contents prosper lost** — [#1944](https://github.com/mattias800/prosper/issues/1944) reading 1, argued from "two *different* guest code sites first-touch the same 64 KiB page; a wild pointer would not repeat like that" | It is reading 2 (a wild read masked), and the repetition is a property of the value, not the mapping. `PROSPER_LAZY_COMMIT_STRICT=1` moves the fault to the loading instruction: `[lazy-commit] #1 DECLINED(strict) page=0x2100000000 addr=0x2100000041 access=read rip=0x41117e221`, then `sig=11 addr=0x2100000041 rip=eboot+0x117e221`. The load is `mov rdi,[r12+rcx*8]` / `mov rcx,[rdi+0x40]` with `rdi=0x2100000001`, and `0x40` is added **without masking the low bit**, so `rdi` is a plain corrupt pointer. **Any** `0x21000000xx` value lands in page `0x2100000000` — which is exactly the shape a heap pointer takes when its low dword is lost — so both recorded sites hitting that page is expected, not anomalous. Exactly one lazy-commit event occurs per affected run. Do not spend another arm treating the page as a commit-protocol gap on this title. | #1944, #1226 |
| The `0x2100000001` that the terminal `addr=(nil)` fault dereferences is composed by prosper's own `RELEASE_MEM`/`WRITE_DATA` fence, the way `0x2000000001` is — and `ptr_like()`'s stale upper bound (exactly `0x2100000000`) is hiding that population from the tripwire and from both guards | The blind spot is **real but empty on this title**, on both branches. With the tripwire's report predicate widened to prosper's whole guest-VA window, one bounded run on `c9e2588e` ends at `FORGE-TRIP-TOTALS seen=256 narrow=256 wide_only=0`: none of the 256 reported candidates is wide-only, every one has `pre=0x2000000000`, and the absence of a `#512` totals line on the dense every-256 schedule bounds the run's population below 512. The sibling `REL1-LIVE` branch — a fence over a pre whose low dword is a real pointer half, which is what would turn a live `0x2100e05140` into exactly `0x2100000001` — was checked separately with `PROSPER_PTRLIKE_WIDE=1`, which arms both guards over the wide window and prints `PTRLIKE-WIDE ARMED` so the lever is witnessed: two runs, `SUSPECT-REL1-LIVE` count **0**, terminal fault unchanged. So prosper's label writes are excluded as the author of the value that terminates the run. Two limits: `report_suspect_write()` has emitted no line of any kind on this title across nine runs, so the `REL1-LIVE` zero has no in-run positive control (it is consistent with the REL1 population being homogeneous, not independently proven); and this does **not** clear the narrow window for other titles — it is still stale, and the lever exists to A/B flipping it. | #1226 |
| `{small nonzero high dword, low dword ≤ 1}` is by itself a prosper-forge signature, so a value of that shape found in guest memory attributes the write to us | It is also **ordinary guest data**. A fault-time dump of a live guest table (`PROSPER_FAULTMEM=1`, register `r12`) shows 16-byte `{pointer, metadata}` entries whose metadata qword is a constant `0x0000000400000002`, interleaved with live pointers `0x2000e03840` / `0x2000e03830` / `0x2100e05140`. `0x400000001` — the value at the `AudioMixerRende` `rip=0x0` jump, and the one #1945's brief quotes — has that same `{4, n}` shape. Value-shape alone therefore cannot attribute a `<high>_0000000n` qword to a GPU write; only a write-side record (the attribution ring, or the forge tripwire's `pre`) can. | #1226, #1945 |
| A one-run-per-arm comparison can A/B a candidate fix on this title | **Three terminal paths compete for every run** and which one wins is a race: measured on `c9e2588e`, the allocator chain (`addr=0x30016000 rip=eboot+0x127e751`), the lazy-commit-masked null (`rip=eboot+0x117e225`, one lazy-commit event), and an `AudioMixerRende` jump to `rip=0x0` with `rax=0x400000001` each terminated at least one of six runs, and one run produced two faults. Consequently the `PROSPER_EOP_WRITE_SYNC=1` and `PROSPER_REL1_WAF_GUARD=1` arms run here are **non-discriminating, not negative**: each was a single run, each ended on the same `AudioMixerRende` null jump as its baseline at ~16 presented frames. Any future arm needs ≥3 runs and a quantitative progress metric (presented frames, delivered video frames, time to first fault), not the identity of the terminal fault. | #1226 |
| The `PROSPER_MB3WATCH` per-thread head watch reports nothing, so the bin head is not being stomped | A DIFFERENT instrument from the `ptr_like()` upper bound two rows above — that one is `command_processor.cpp`'s report predicate, this one is the watchpoint's arm hook, and widening either says nothing about the other. The arm hook (`exec_image_linux.cpp`) filtered on a stale DOLL-era `[0x20_0000_0000, 0x21_0000_0000)` window while every current title's per-thread cache base is `0x30_xxxx_xxxx` — ArcRunner's is `0x3152350000`, Crisis Core's `0x3001af0000`. It therefore **armed nothing at all**, printed nothing, and read exactly like "armed and saw no write" — the same class as the #1998 finding for `PROSPER_WATCH_LABEL`/`PROSPER_WATCH_HOT`. Window widened to `[0x20..0x40)`, and its report trigger widened from the byte-shift value shape to the structural "a head must be 0 or a 0x20-aligned mapped node". Any null quoted from this instrument before 2026-08-06 is **void, not negative**. | #1945, #1998 |
| Both prosper-authored halves of the forged pointer are jointly necessary for the corruption (the open question left by the *Suppressing the forging fence* row above; that ArcRunner arm was read as settling it) | Run on **Crisis Core** (`PPSA07809`), a cheaper reproducer for the same family, with the same lever and a valid census (`candidates=127 suppressed=127 landed=0`, `INIT-SUPPRESS` past #4096): the guest allocator is **still** corrupted and still faults at its bundle-list pop, with `0x0002400100024001` in the bin head. Separately, the "our completion writes land late in a recycled block" premise underneath this whole line was measured on that title and is **false**: pend-queue residency peaked at **3 ms** over 5,632 writes, none above 20 ms (`PROSPER_PEND_AGE`). What *does* decide the outcome there is the DURATION of the guest's own submit call, measured as a dose-response over a plain `nanosleep` in the submit fold: **0/3 survive at no stall, 0/3 at 500 us, 3/3 at 1500 us, 3/3 at 3000 us** (0/6 vs 6/6, Fisher two-tailed p ≈ 0.002), threshold between 500 and 1500 us. That is the controlled form of the timing-not-code conclusion `OREGON_TRAIL_STATUS.md` reached from nine arms of PPSA19244. | #1945, #1894 |

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

**Where the terminal-fault hunt actually stands after the instrument repair** (added without
renumbering the list above, which a concurrent lane also edits):

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
