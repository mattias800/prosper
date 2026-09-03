# Stray (`PPSA02101`) — status

Tracker: [#2883](https://github.com/mattias800/prosper/issues/2883). Engine: Unreal Engine 4.
Route: [`prosper/scripts/stray-PPSA02101/`](../scripts/stray-PPSA02101/README.md).

**Rung 2.** A Cross-only route accepts the brightness-calibration screen and reaches the first map
load (`hk_project_mainstart`, t ≈ 37 s, absent from every default run). Two separate things are wrong
after that, and they are worth keeping apart because they have different evidence:

| what | issue |
| --- | --- |
| the world composites as a flat **letterboxed** clear — no scene renders | [#2932](https://github.com/mattias800/prosper/issues/2932) |
| the **title screen** renders its menu but not its background | [#3126](https://github.com/mattias800/prosper/issues/3126) |

## READ THIS BEFORE QUOTING A NUMBER FROM THIS DOC

**`max_nonblack = 0.1140` is the brightness-calibration screen, NOT the title screen.** The title
screen is **0.0069**. Both are held steady states and both look like "the run settled", so the two are
easy to confuse — and every live A/B in the section below was originally run against the wrong one.

| screen | `max_nonblack` | what it looks like |
| --- | --- | --- |
| brightness calibration | **0.1140** | **correct and complete** — three cat-head silhouettes at increasing brightness, the instruction text, the 16-step slider, `△ Defaults` / `✕ Accept` |
| title screen | **0.0069** | menu only — `START GAME` / `SETTINGS` / `CREDITS`, the version string and `✕ Select`, everything else black at 8× brightness |

The calibration screen rendering correctly is real, previously unrecorded progress: it is a full
screen of the title's own art and UI, drawn right.

**Reaching the title screen at all needs `PROSPER_NULL_PAGE=1`.** Without it a run either faults
(`rc=90`, a `[nullpage]` report) or survives and renders pure black after Accept. With it the route
still only lands roughly one run in three, so **repeat any A/B here** rather than trusting a single
arm.

**`reach-title-flip.pad` does not reach the title screen.** Measured over a 320 s run: 16 of 16
captured frames are the calibration screen at 0.1140. The route README's claim that it stops at the
title is wrong and needs correcting.

## Three routes, and which to use

| route | reaches | notes |
| --- | --- | --- |
| `reach-title-hold.pad` | the **title screen** (0.0069) | needs `PROSPER_NULL_PAGE=1`; lands about one run in three |
| `reach-title-flip.pad` | the **calibration** screen (0.1140), held | flip-anchored and deterministic — a stable oracle for that screen, and *not* a title route despite its name |
| `reach-first-map.pad` | past calibration to the first map load | |

`reach-title-flip.pad` was landed as a title route and is not one; the banner above records the
measurement. It is kept because holding calibration deterministically is genuinely useful — that
screen renders correctly — and because renaming a committed route would break every citation of it.

## Performance: this title is not GPU-bound

Measured 2026-08-29 with an F8 `.prperf` capture over a 5.02 s window at the title screen, read with
`tools/perf/performance_capture_report.py`:

| bucket | ms |
| --- | --- |
| `gpu-device` | 221.7 (≈ 4% of the window) |
| compute | 2023.8 |
| texture materialisation | 1137.4 |
| buffer copy | 602.0 |

One compute program, `0x3011300000`, accounts for ~605 ms of that at roughly 32 ms per dispatch at
3840×2160. So the cost is on the host side of the boundary, and pointing a GPU profiler at this title
answers a question it does not have. Detail on [#3126](https://github.com/mattias800/prosper/issues/3126).

### Inside the texture leaf (2026-09-04, `fc21d46ca`, 5.02 s at the title screen)

The 1110 ms above is **one class of reference and two surfaces**. Read from the same `.prperf` with
the frontend cache-outcome classes the report now prints:

| class | ms | note |
| --- | --- | --- |
| rtt | 77.3 | |
| compute | 59.9 | |
| persist_hit | 25.0 | |
| local | 17.1 | |
| persist_reuse / persist_miss | 0.0 / 0.0 | |
| **unclassified** | **930.8** | **84% of the leaf** |

The slowest single unclassified reference is 59.6 ms, and its identity is the whole story:
**3840×2160 RGBA16F, 63.8 MiB of tiled source, DCC on, a compute *and* a persistent candidate**, at
one of exactly **two alternating addresses** (`0x30784e0000` / `0x30d10f0000`) — a double-buffered 4K
HDR intermediate, re-decoded on every callback that samples it.

**Two thirds of that decode was the allocator and the kernel, not the decode.** Each reference built
two fresh value-initialised `std::vector<uint8_t>` intermediates — 66,846,720 tiled bytes and
66,355,200 linear — both past glibc's 32 MiB mmap threshold, so each was an `mmap`, 16,000 page
faults, a memset and a `munmap`, per reference. Measured standalone at this exact shape:

| | ms |
| --- | --- |
| fresh `traw`+`hlin`, copy + detile-equivalent traffic | 26.00 |
| **the two fresh zero-initialised allocations alone** | **17.84** |
| pooled `traw`+`hlin`, same copy and traffic | 6.00 |
| pooled linear only, tiled source read in place | 2.83 |

This is the same cost [#3149](https://github.com/mattias800/prosper/issues/3149) saw from outside as
`__memmove_avx512` at 14.5% self and `clear_highpages_kasan_tagged` / `map_anon_folio_pte_nopf` /
`zap_present_ptes` / `__free_one_page` under a 22.2% `do_syscall_64` — and attributed to the compute
path, because that is where it looked. It is the graphics frontend materializer.

Two more allocations of the same family sit on the same reference. The persistent cache re-stored a
fresh 63.8 MiB `source_prefix` on every invalidation (13.00 ms at this shape) — it now inherits the
allocation from the entry it replaces. The remaining one is **not** addressed:
`cached.pixels = std::move(texture_pixels)` steals the pooled `texstore` slot, so the next decode
allocates its 33 MiB output buffer fresh (~4.4 ms) — [#3310](https://github.com/mattias800/prosper/issues/3310).
Fixing it needs a decision about buffer ownership between `texstore` and the persistent cache, which
hands its buffer out as a `shared_ptr<const std::vector>` that never comes back.

### The lever that is worth more than all of this

The witness says `compute_cand=1`: the surface is a **compute image candidate whose import misses
every frame**. `import_live_compute_storage_image` finds nothing under its key, so graphics falls
through to the guest-byte decode. If that import hit, the 63.8 MiB writeback → detile → convert
disappears **on both sides of the boundary** — the graphics decode and the compute writeback that
feeds it. That is a bigger lever than every allocation above put together, and it lives in the
compute cache's admission, not in the materializer. Both halves of that round trip are read
together in [#3307](https://github.com/mattias800/prosper/issues/3307): the compute side's re-tile
into guest memory, and the graphics side's read of those same bytes back out.

## The unresolved image ops — established on CALIBRATION

> **The five-op census below was read on the calibration screen**, like every other live census above
> § *The title screen's REAL numbers*. The resource-table finding it rests on is a statement about
> shader stages and holds regardless of screen; the count of five does not.

`[mimg-unresolved]` reports five image ops on one routed boot whose descriptor resolves to nothing;
every draw using those shaders is discarded. What is **established**: for the shader stages that were
dumped, the resource table is complete and correct — each declared read-only T# is present, the
SGPR-resident one under DIRECT (`sgpr_base`) provenance and the EUD-resident ones under INDIRECT
(`srt_offset = (offset_dw - num_user_sgprs) * 4`) provenance, which is exactly what the table shows.

What was **not** established at the time of that census is which stage the five failures belong to.
Every line reported `program=0x0`, because `recompile_fragment_impl` hardcoded a zero program address
([#3130](https://github.com/mattias800/prosper/issues/3130)) — so no fragment-shader recompile failure
on any title could be attributed to a shader. **That is fixed** (#3132): fragment diagnostics now carry
the real guest program address, which turns this from an inference into a lookup. The five failures
above predate the fix and were never re-attributed, so they remain unassigned to a stage — re-running
the census is what would assign them.

Separately, four stages declare a *writable* 8-dword T# that reaches no resource table at all
([#3128](https://github.com/mattias800/prosper/issues/3128)); one of them has four image ops against a
completely empty table. Whether that is what its image ops want is untested — see Ruled out.

## The dropped-draw census — measured on CALIBRATION

> **VOID as title-screen evidence, and this warning covers BOTH censuses below it.** Every number in
> this section was read on the brightness-calibration screen (`max_nonblack` 0.1140), not on the
> title screen (0.0069). The section is kept because these numbers are cited elsewhere and need
> somewhere to point — **not** because anything in it survives as title-screen evidence. An earlier
> version of this banner claimed two *mechanism* findings did survive: that no `CB_TARGET_MASK`
> register is being lost (so this is not the *Oregon Trail* defect #1946), and that the
> colour-masked-off draws come from one shader. Both are marked VOID in `## Ruled out`, and the
> banner cannot exempt them: a trace that ran only on calibration says nothing about which registers
> reach the GPU on the title screen, mechanism or quantity. Every number here describes the wrong
> screen. Do not
> quote the 1024, the 7, the 32,649 or the 92% as facts about the title screen; § *The title
> screen's REAL numbers* has that screen's own census, and it disagrees.

`PROSPER_DROPPED_DRAW_CENSUS=1` on the **calibration** route, at 1024 discarded draws:

| reason | count | targets |
| --- | --- | --- |
| `no-effect(early)` | 1015 | `0x9fc0000000` (511), `0x9fc2000000` (504), two others |
| `shader-recompile` | 7 | `0x9fc2000000` (4), `0x9fc0000000` (3) |

So **on calibration** the unresolved image ops cost 7 draws, and nearly everything discarded is
dropped because every colour target's write mask is zero with no depth/stencil side effect. That
conclusion does **not** transfer: the same census on the title screen reads ~3800 `shader-recompile`
of 8192 discarded.

**But the dropped draws are not where the missing picture is either.** `PROSPER_COLORSTATETRACE=1`
over a shorter window traced **32,649** draws, and every single one carries a *decoded*
`CB_TARGET_MASK` — the presence flag is 1 on all of them, so no register is being lost, which is what
would have made this the *Oregon Trail* defect (#1946):

| `CB_TARGET_MASK` | draws |
| --- | --- |
| `0x0f` | 26,774 |
| `0xff` | 1,655 |
| `0x07` / `0x0c` | 884 / 838 |
| **`0x00`** | **2,498** |

Roughly **92% of draws write colour and execute**. The colour-masked-off draws all come from **one**
fragment shader, `0x3010660000`, which exports a full `0xf` colour mask and runs with
`DB_DEPTH_CONTROL` `0x70`/`0x62` — Z_WRITE off, no stencil — so under the RDNA rule (the masks are
upstream hardware gates that an export can narrow but not enable) they genuinely write nothing, and
`PROSPER_FORCE_COLORWRITE=1` admitting them makes the composite worse.

**So the title screen is missing content while the overwhelming majority of its draws execute** — but
see the frame dissection above before concluding, as I did at first, that the discarded draws
therefore do not matter. They are few and each one is full-screen. The one thread still worth pulling on the dropped side is whether `CB_TARGET_MASK=0` is
*stale* for that shader rather than current — #1706 established that prosper's decoded
`CB_COLOR_CONTROL.MODE` is not per-draw-trustworthy because a utility sequence's bits stay latched
onto later ordinary draws, and the same shape would explain one shader consistently reading zero.

## The title-screen frame, dissected offline

A **captured F9 bundle of the title screen already exists** and reproduces the defect deterministically
without booting the game — the fastest loop this title has. Its submit-14072 capsule inspects as:

```
submit=14072 3840x2160 draws=92 computes=53 operations=156 shaders=113 failed=11
```

**Eleven of 156 operations never realize** — ten draws and one dispatch — and every one of the
`shader-recompile` drops is a **full-screen 3840×2160 pass** with `CB_TARGET_MASK=0xf`, scissor
`0,0..3840,2160`, alternating between the two swap targets `0x9fc0000000` / `0x9fc2000000`. That is
why the count looked negligible and the picture is not: seven-to-eleven draws is nothing by count and
the whole composite by area. **Do not dismiss these by draw count** — that mistake is what sent this
investigation down the `no-effect` path.

`gpu_replay --retry-failed-stage <failure>:<stage>` with `PROSPER_DBG=1` names each cause exactly:

| failure | stage | reject |
| --- | --- | --- |
| 1, 2 | vertex `0x300f190000` | `v_mbcnt_lo/hi_u32_b32` cross-lane, rejected at **pc=4** -- see the correction below |
| 3 | fragment `0x30be800000` | MIMG `op=0x1`, `recompile-reject-mimg-address extra=1` at pc=134 |
| 6-9 | fragment `0x300c010000` | `s_mov_b32 s0, m0` — `scalar-data-reject pc=37 special=s124 tracked=0` |
| 10 | compute `0x300e390000` | the same `s_mov_b32 sX, m0` at pc=157, behind a `nested-backedge-in-body` loop reject at pc=688 |

Two more fragment programs fail in other draws of the same frame: `0x300e500000` (`unsupported=29`,
first reject pc=20) and `0x3011560000` (`unsupported=1`, first reject pc=383).

The vertex one is the deepest, and **two things recorded here were wrong**; both were re-measured on
2026-09-01 against the same capsule (#3135).

`0x300f190000` contains **both** MBCNT forms: the canonical all-ones lane-id pair at pc 4/7 *and* four
general **SGPR-mask** MBCNTs at pc 277-286. The reject is at **pc=4**, not at 277-286 -- the all-ones
pair is lowerable on its own, and the presence of a non-all-ones form anywhere in the program is
exactly what sets `logical_mbcnt_invalid` and disqualifies the whole-program `ngg_logical_lane` model,
so the failure surfaces at an instruction that is not the cause. The reject line now names the
disqualifying pc (`mode=unsupported-in-stage reason=mbcnt-cross-lane stage=vertex form=all-ones
wave=none disqualified-by-general-mask-pc=277`).

**And MBCNT is not the frontier.** With the MBCNT satisfied by a throwaway probe, the live-config
`gpu_replay --retry-failed-stage 1:1` reject moves ten instructions, to `pc=14 ds_write_b32`
(`--retry-failed-chain 1`: `pc=116 ds_read2_b32`). The program is a **merged NGG ES+GS threadgroup**,
not a vertex shader: 288 instructions carrying LDS traffic at four offsets, three `s_barrier`s,
`s_bcnt1_i32_b64` x2, `s_pack_ll_b32_b16`, two `v_add_nc_u32_dpp row_shr` cross-wave scans,
`v_readlane_b32` x2, `s_sendmsg(MSG_GS_ALLOC_REQ)` and `exp prim`. It computes an NGG
vertex/primitive compaction: per-wave popcounts published to LDS, a DPP prefix sum across waves, and
`v_mbcnt` for the within-wave prefix. `rdna2_to_spirv.cpp` fails all of it closed on purpose -- its
comment says the wave approximations are "an exception for the one captured Astro wrapper, not a
property of the GS_ALLOC_REQ opcode". **The gap is a missing WAVE, not a missing lowering**: a Vulkan
vertex stage has neither shared memory nor any promise that a subgroup is the guest wave, so the
honest home for this program is a mesh shader. See `RECOMPILER_REMAINING.md` § Ruled out.

**Reproducing any of it takes seconds, not a boot:**

```bash
gpu_replay <capsule>.prgcap --inspect-only                     # the failure/operation list
PROSPER_DBG=1 gpu_replay <capsule>.prgcap --retry-failed-stage 6:1
gpu_replay <capsule>.prgcap --dump-failed-shader 1:1 vs.bin    # the raw guest program
```

## The scene targets are black AT SOURCE

The obvious next hypothesis after "most draws execute" is that the world renders into an offscreen
HDR target and the tonemap/composite that reads it is lost. **It is not.** Every render target the
title-screen frame samples can be dumped straight out of the capture — no boot, no guessing which
draw wrote what:

```bash
gpu_replay <capsule>.prgcap --dump-rtt-seed 0x3096fd0000 out.bin   # writes a BMP
```

| seeded target | format | brightest channel |
| --- | --- | --- |
| `0x3096fd0000` 3840×2160 | rgba16f | 5 |
| `0x30a4a20000` 3840×2160 | rgba16f | 0 |
| `0x3071dd0000` 1920×1080 | r11g11b10f | 6 |
| `0x300a790000` / `0x3025380000` 480×270 | r11g11b10f | 4 / 0 |
| `0x3092b10000` 3840×2160 | rgba8 | 0 |

Every HDR scene target is black at source, so the world's colour never exists to be lost downstream.
**This is not a composite defect.** The draws run and write nothing — the same family as *Grand Theft
Auto V* and *Sonic Frontiers*, not a missing pass.

**Two of the targets are traps rather than evidence.** `0x3094b60000` and `0x30a29e0000` score
`nonblack=0.2543, mean=63.8` — by far the brightest things in the frame — and both are a **white
quadrant covering the top-left 1920×1080**, i.e. prosper's own seed-miss fill. `0x30563d0000` carries
the seed-miss *gradient*. A brightness metric ranks all three above any real content in this frame,
which is the recorded hazard: open the image before believing the number.

## Measured on CALIBRATION, and only one row survives as title-screen evidence

(The one is the composite/tonemap row, which was established on the title-screen bundle. An earlier
version of this section counted two, by crediting the colour-write-mask row with a surviving
"mechanism" its own VOID marking denies.)

Every number in this table was read on the 0.1140 calibration screen. The table is kept because it is
cited, and annotated because three of its four rows were read as ruling out causes on the title
screen, which they cannot do. **The next section has the title screen's own numbers, and they
disagree.**

| candidate | measurement (calibration) | title-screen status |
| --- | --- | --- |
| dropped draws | 7 `shader-recompile`, ~30,000 draws executed | **FALSIFIED for the title screen** — ~3800 there, see below |
| skipped compute | `0x300ba70000` executed **7455**, skipped **2** (`PROSPER_COMPUTE_PROGRAM_CENSUS=1`) | not re-measured on the title screen |
| lost colour write masks | present and decoded on 32,649 of 32,649 traced draws | **VOID** — a trace run only on calibration says nothing about which registers reach the GPU on the title screen |
| composite / tonemap | the HDR sources are black before it runs | holds — established on the title-screen bundle |

## The title screen's REAL numbers (measured on a 0.0069 frame)

Everything above this section that quotes a live census was measured on calibration. Each such
section carries that warning **at its own head, above the first number it covers** — a retraction a
hundred lines below the number it retracts is one most readers never reach, and a banner placed
mid-section silently exempts whatever sits above it. These are the
title screen, `PROSPER_NULL_PAGE=1`, census read at the same cumulative total in every arm:

| | calibration (0.1140) | **title screen (0.0069)** |
| --- | --- | --- |
| draws discarded | 1024 | **8192** |
| `shader-recompile` | 7 | **~3800** |
| `no-effect` | 1015 | ~4400 |

Two targets take almost all of it — `x1260` and `x835` in one run. **That**, not the seven drops this
investigation spent hours on, is why the background is black.

`PROSPER_DBG=1` on the same route gives the remaining work list, in order of instances:

| reject | count | note |
| --- | --- | --- |
| `[vertex-recompile-reject] body or export lowering failed` | 15 | |
| MUBUF `unresolved-operand`, `fmt=12` | 14 | see below |
| MIMG `fmt=14` | 3 | #3134's family |
| `scalar-data-reject special=s124` | 4 | fixed by #3133 |

### The MUBUF class, decoded

```
sh=/77  pc=57 words=e00c2000,6a010000 op=0x3 src=0(k2),4(k1),106(k6)  stage=vertex
sh=/221 pc=66 words=e0042000,6b020900 op=0x1 src=0(k2),8(k1),107(k6)  stage=vertex
```

Both are `buffer_load_format_*` with `idxen` — **vertex attribute fetch** — in small vertex shaders,
with `VCC_LO`/`VCC_HI` as the soffset. The opcodes themselves are implemented (`0x0`-`0x3` are all
handled), so the reject is an operand, not a missing instruction. The next thing to establish is
whether the failing operand is the soffset or the SRSRC: `allow_smem` is `(rt != nullptr)` for
graphics, so **a vertex stage that arrives with no resource table fails every buffer op it has**, and
vertex fetch is a buffer op — that would take the draw with it. A probe on `rt == nullptr` at the
vertex recompile answers it in one run.

### What #3133 is worth here

Measured on this screen, census at the same total, two arms per build:

| build | `shader-recompile` @ 4096 discarded |
| --- | --- |
| baseline | 1026, 989 |
| with #3133 | **866, 886** |

A ~13% reduction, groups non-overlapping. The visible frame does not change — the remaining ~870
drops still discard the background.

## Ruled out

- **"The surfaces prosper renders into read black because the live RTT cache does not treat them as
  authoritative, so the sampler falls through to zeroed guest memory."** Falsified (#3140) — 58,569
  sample HITs to 185 misses (an instrumented-run figure), and — the part that needs no run — the
  graphics RTT sampler path is **on by default**, so a black default frame is not explained by that
  path being off. The measurement, and why the original `PROSPER_RTT=1` arm did not mean what it was
  captioned as meaning, are recorded above in the sub-bullet under the staging-buffer entry;
  this row exists so the falsification is findable from the section people actually read before
  forming a hypothesis. Do not restart from the RTT cache. #3140, #3126.

- **"The title-screen vertex stages die because the recompiler cannot lower
  `buffer_load_format_* … idxen`, or because they arrive with no resource table."** Both falsified
  (#3137). The census line that raised them names an INSTRUCTION —
  `[recompile-reject] sh=…/77 mode=unresolved-operand pc=57 words=e00c2000,6a010000 … stage=vertex` —
  and that is not where the defect is. MUBUF `0x0`–`0x3` are implemented, and the disposition line
  from the same run says which arm fired: `[buf-op] … pc=57 MUBUF op=0x3 n=4 … rt=1
  reject-unresolved`. `rt=1` with five resources kills the no-resource-table candidate outright, and
  the `VCC_LO`/`VCC_HI` SOFFSET is not it either — `operand_bits` has a vertex-stage branch for
  exactly that, and the reject happens before the address is ever formed.
  What actually fails is SRSRC provenance. Disassembled (`llvm-mc -mcpu=gfx1030`), both failing
  stages build their own descriptor:
  `s_load_dwordx4 s[4:7], s[12:13], vcc_hi` with a register offset, then a `s_and`/`s_or`/
  `s_cselect_b32 s7, …` patch of dword 3. That leaves no SRT key and rewrites the SRSRC range, so the
  direct-SGPR fallbacks are correctly suppressed and only the const-fold's per-fetch entry could
  resolve it — and it has none.
  - **The const-fold is not the defect either; it is refusing correctly.** `PROSPER_DYNTRACE_FAIL=1`
    shows the whole chain hanging off `s_load_dwordx4 s[16:19], s[14:15]`, whose base reads
    `0x0004dfac00000001` — `addr … unreadable` — after which every scalar is `ok=0` and the fetch is
    "left unresolved (not folded to 0)".
  - **This is #305, on a third title.** `PROSPER_UDPROV=1` shows the draw's own bind writing twelve
    contiguous user dwords in one direct `SET_SH_REG`, while `RSRC2_GS.USER_SGPR = 8` equals the
    shader's `user_data_range_end = 8`. The two mapped 64-bit pointers the shader wants sit at dwords
    8 and 10, outside that window; dwords 6:7 hold the `num_records`/`dword3` tail of the preceding
    V# — the `0x0004dfac…` constant family `RESOURCE_BINDING.md` already names. Do not restart this
    from the recompiler. See `RESOURCE_BINDING.md` § Ruled out, and note the row there recording why
    the `[udcand]` "implied seed" offset is *not* a safe fix.

- **"Vertex `0x300f190000` is blocked on `v_mbcnt` having no vertex-stage lowering, so giving the
  vertex shell a general SGPR-mask MBCNT path recompiles it."** **Half falsified, and the surviving
  half is not implementable in the vertex shell.** With MBCNT satisfied by a throwaway probe the
  live-config reject moves ten instructions -- `--retry-failed-stage 1:1` to `pc=14 ds_write_b32`,
  `--retry-failed-chain 1` to `pc=116 ds_read2_b32`. The program is a merged **NGG ES+GS
  threadgroup**: LDS at four offsets, three `s_barrier`s, `s_bcnt1_i32_b64` x2,
  `s_pack_ll_b32_b16`, two `v_add_nc_u32_dpp row_shr` scans, `v_readlane_b32` x2,
  `s_sendmsg(MSG_GS_ALLOC_REQ)` and `exp prim` -- MBCNT is one of eight wave/workgroup primitives it
  needs, first only in program order. And a subgroup-based lowering would not be exact anyway: the
  vertex shell's guest lane is MODELLED as the flattened invocation index, and Vulkan does not
  promise a subgroup is that set of lanes. **The gap is a missing wave, not a missing lowering.**
  #3135, and `RECOMPILER_REMAINING.md` § Ruled out for the full argument.

- **"Modelling entry-M0 as a constant is fine, because the shader only saves and restores it."**
  Falsified for the general case, and the falsification is what shaped the fix. `0x300c010000`
  itself does only save and restore (dw37/38/88/92 and dw305/306/356/359, no VINTRP, no
  `s_sendmsg`, no `s_movrel`, and every DS op runs under an M0 the shader wrote to 0), so any
  consistent value round-trips there. But the arm applies to every shader, and a constant makes
  the destination an ordinary tracked scalar: one `s_mov_b32 s0, m0` in front of
  `v_add_nc_u32 v1, s0, v0` then defeats the guard that rejects the direct form, and the restore
  makes M0 itself *tracked*, silently defeating seven guards that require it untracked. Zero is
  additionally the worst constant available: the shader's own LDS base is `s_movk_i32 m0, 0`,
  `uconst` is interned, so post-restore ADDTID slot keys alias pre-restore ones exactly and the
  restore becomes a model-level no-op. The shipped model is an opaque token instead; see
  `docs/RECOMPILER_REMAINING.md` § Ruled out for the two ways a token-with-no-value is NOT
  self-containing (CFG merges and the dispatcher spill both fabricate a zero for it). #3133, #3136.

- **"The 4K title background is black because its asset never loads, or loads late."** Falsified, and
  by three sources that share no code. The pak read delivers the bytes: `PROSPER_APR_VERIFY`
  re-reads each guest destination through `process_vm_readv` immediately after the write and
  **`memcmp`s it against the source** — 21,499 of 21,499 writes byte-identical on the title route,
  with a mutation arm (`PROSPER_APR_VERIFY_SELFTEST`) proving the comparison can report a loss. A
  `dd`/`od` read of `hk_project-ps5.pak` outside the emulator agrees exactly with the in-memory
  read-back (25 of 64 non-zero dwords at texture offset 0). And the range is nonetheless **entirely
  zero when the shader samples it**: base `0x303cd10000`, BC1_SRGB 3840×2160, read by
  `0x300c150000`, `nonzero=0/512` at +65.7 s after its verified write.
  So the pak reader is not the place to look; something zeroes these ranges between load and use, and
  the row below names it. #3142.

- **"The range is zeroed some time during the ~65 s before the shader reads it."** Falsified, and the
  65 s was an artifact of when the *shader* happened to look rather than of when the data went away.
  `PROSPER_ZEROWATCH` polls each armed destination every 50 ms and reports the transition itself: the
  watched **256-byte window** is zero **0.5–2.3 s** after its load, and the defect therefore
  **reproduces in a 2-minute run** rather than the 7-minute title route — it happens during asset
  load, long before the title screen. **This row said "the data is gone", which is a claim about the
  whole range that the instrument never tested — read the CORRECTION bullet below before quoting
  it: on most of these destinations the payload is still there.**
  At the instant of the flip the mapping is *unchanged*: still `rw-s` on the dmem memfd at the same
  file offset, so it is neither unmapped nor re-pointed. And because that mapping is `MAP_SHARED` on a
  memfd, reading the mapping **is** reading the file — there is no "stale view" case, so the content
  itself was zeroed. #3142, #3145.
  - **CORRECTION (2026-09-02, #3142): "the data is gone" is a statement about ONE 256-byte window,
    and of the payload it is false in the common case.** `PROSPER_ZEROWATCH` arms on the *first*
    non-zero window of a payload -- window 0 in every case measured -- and its WENT-ZERO verdict
    reads that single window. It now profiles **16** windows across each destination at the instant
    of the flip and again at +10 s and +45 s, against the source's own per-window census so an
    always-empty window is not counted as a loss. That separates two populations the one-window
    verdict had merged. On a 120 s boot (`PROSPER_NULL_PAGE=1`, no input): **11 of 14** destinations
    still hold most of their payload 45 s after their own WENT-ZERO (`kept` 4-15 of 16 windows),
    **3 of 14** end genuinely all-zero, and one destination's zeroed window reads **non-zero again**
    at +10 s. So what the guest clears about a second after a load is, in the common case, the
    chunk's header; the asset body survives. The 2-minute reproduction and the 0.5-2.3 s figure are
    unaffected -- they are correct about window 0. What is withdrawn is the inference from them to
    "the range", which is what made this look like a memory-lifetime defect across the board. The
    fully-cleared minority is the class this issue's own subject case belongs to, and the log now
    separates it in one line instead of requiring a second instrument.
  - **No verbatim relocation: the payload has exactly ONE guest-visible copy while it is live, and
    none afterwards.** `PROSPER_ZEROWATCH_FIND=N` searches the whole guest address space
    (`[0x100000000, kGuestAutoMapLimit)`) for the payload's own bytes -- a control scan taken while
    the data is demonstrably present, then a second scan at the flip, so an empty second result is
    falsifiable rather than void. Control finds the payload at exactly one guest address, the
    destination itself; the post-flip scan finds it at no guest address at all. Run with a **16-byte**
    needle (`PROSPER_ZEROWATCH_FIND_PREFIX=16`), the length that survives Gen5 tiling, so this is not
    merely a statement about untiled copies. That length is derived from prosper's own swizzle
    tables rather than from a generic Z-order reading, because the generic reading gets it wrong:
    `src/gpu/texture/tile.cpp:149` puts an 8-byte element (BC1) in an **8x4** micro-tile, not 8x8
    (8x8 is the 4-byte case), and `tile.cpp:275`'s bit order for 8-byte elements
    (`x0 y0 y1 x1 x2 y2 x3 y3 x4`) puts a row's first four blocks at byte offsets 0, 8, **64**, 72 --
    so two consecutive blocks stay adjacent and the third is eight elements away, not four.
    **Two conditions on that survival, both real:** the 16-byte run must start on an EVEN block (an
    odd one spans elements 1 and 8, 56 bytes apart), which is why the needle offsets are 16-aligned;
    and 16-alignment relative to the destination is 16-alignment relative to the SURFACE only if the
    surface starts 16-aligned inside the buffer, which a file read cannot tell us. **What it cannot
    see at all:** a payload the guest decompressed or re-encoded is not the same bytes at any needle
    length, so the null means "no verbatim guest copy", never "the data does not exist". Scanning
    wider *does* find these payloads in prosper's own host-side buffers, which is exactly why the
    scan is bounded to guest space -- prosper holding a copy answers nothing about where the guest
    put it. #3142, #3243.
  - **These destinations are a recycled pool, which is why "was it reused?" has to be asked
    per-address.** Over one 90 s boot, **1,147 of 9,631** distinct APR destinations receive reads
    from two or more different file offsets. #3142.
  - **The writer is the guest's own libc, and that reclassifies the whole issue.** With #3147's
    re-baseline mode the write trace observes the window it previously could not, and attributes
    **64 of 64 events to `libc.prx`** — guest code — storing sequentially from the buffer base at a
    16-byte stride through a tight `0x3d71`–`0x3dd0` RIP loop, on one thread. The guest clears its own
    staging buffer about a second after loading it, which is ordinary engine behaviour. So this is
    **not** a memory-lifetime defect and never was: the pak reader delivers correctly, the guest
    disposes of its buffer correctly, and prosper's defect is that a shader still reads that address
    afterwards. The open question is now a descriptor one — why `0x300c150000`'s texture resolves to a
    staging buffer the engine has finished with — which also fits its `rtt=MISS`. #3142.

    **And it is NOT why the background is black** — established after the above and worth stating
    here, because the reclassification reads like a lead and is not one. The graphics RTT sampler
    resolves these surfaces correctly: a landed title-screen run measures **58,569 sample HITs against
    185 misses and 5 skips**, and all five skips are 32x32/64x64 3D textures.

    **And the evidence is stronger than the `PROSPER_RTT=1` arm it was first reported as.** That arm
    was read as switching the sampler path on, and it does not: `pertarget` is
    `getenv("PROSPER_RTT_PERTARGET") != nullptr || PROSPER_ENV_VALUE("PROSPER_RTT_SINGLE_TARGET") == nullptr`,
    so with neither variable set it is **true**, and `rtt_on = getenv("PROSPER_RTT") != nullptr ||
    pertarget` is true with it. The graphics RTT sampler path is therefore **ON BY DEFAULT**, and
    that is a property of the code needing no run: a black default frame is not explained by the
    path being off. That is what excludes the cache, and it is a stronger statement than "enabling
    the path changes nothing".

    `PROSPER_RTT` is read at **three** sites, though, and the other two are not `rtt_on`. Both
    `agc_shader_layout.cpp` and `gpu_executor.cpp` declare
    `rtt_bind = getenv("PROSPER_RTT") != nullptr || getenv("PROSPER_RTT_PERTARGET") != nullptr`
    — **no `PROSPER_RTT_SINGLE_TARGET` disjunct**, so `rtt_bind` is false by default and true under
    `PROSPER_RTT=1`, the opposite default from `rtt_on`. It decides whether an unmapped Gen5 image
    format is dropped or bound as RGBA8. So `PROSPER_RTT=1` is a real change, and the "renders no
    additional pixel" null spans a binding-policy difference rather than nothing at all — it
    measured something, just not what its caption said. Read the identifier, not the variable name:
    every error made on this question, this row's first version included, came from resolving the
    line that was expected to matter and reading past the ones beside it.

    So the composite reads black because its INPUTS are black at source (see the
    section of that name), not because a sample was served from cleared guest memory. The staging
    buffer being cleared is ordinary engine behaviour with no bearing on the picture. #3140 is
    falsified; do not restart from the RTT cache.

    One caveat on the counted run, since the counts came from `PROSPER_RTTLOG`: that variable is the
    final conjunct of **`live_gpu_targets`** (not of `timeline_capture_requested`, which is its own
    one-line predicate above it), so setting it changes what the run RETAINS rather than only what
    it prints. The HIT/miss ratio is therefore an instrumented-run figure. The default-configuration
    claim above does not depend on it.
  - Four mechanisms measured and excluded, each in the same runs: **`dmem_zero`'s hole punch** (all 20
    calls occur at startup, before every one of these writes); **a re-map of the VA** (one `[physmap]`
    per VA, created ~2 ms *before* its write, none after); **a second VA aliasing the phys** (none —
    each phys range is mapped exactly once); and **the renderer writing back**
    (`guest_memory_gpu_write` fires once per run, covering none of the zeroed ranges). A fifth:
    **the address being recycled for another asset** — `0x303cd10000` receives exactly one payload
    and is never written again, though 956 of 6,574 destinations in the run genuinely are reused, so
    this had to be checked per-address rather than assumed either way.
  - Corroborated on a second base: `0x302a300000` is sampled by five different shaders across
    **152 s** and reads zero every time, the earliest only 2.0 s after its byte-verified write. One
    address going quiet could be a descriptor pointing somewhere stale; two, with one of them read by
    five separate shaders, is the range.
  - What *used* to block naming the writer, kept because it explains why this took so long and what
    the attribution above depends on: `PROSPER_DMEM_WRITE_TRACE` records writer RIPs but was
    invalidated by a **host** write, and the APR load *is* a host write (all four host writers of
    guest memory in the tree are file reads), so it could never watch a range APR loads into — it
    reported `page-faults=0` because the watch died at the load, not because no guest write happened.
    **[#3147](https://github.com/mattias800/prosper/pull/3147) lifts that** with an opt-in
    re-baseline mode (`PROSPER_DMEM_WRITE_TRACE_REBASE=1`), which is what makes the `libc.prx`
    attribution above reproducible.
    A `SIGSEGV` trap hand-rolled to get the RIP directly killed the run at asset load and was removed
    rather than debugged — extending the existing trace's lifecycle was the route, not a second
    write-watch fighting the first.
  - Two traps came out of it and are recorded in `GAME_COMPAT_ORCHESTRATION.md`: **235**, log line
    order is not happens-before (the ordering above is a shared monotonic stamp, not line distance);
    **236**, a verifier that compares populations instead of content reports MATCH on entirely
    different bytes, and passes a mutation arm while doing so.

- **"prosper performs the APR read at command-RECORD time where real hardware performs it at SUBMIT,
  so the guest's own preparation of the destination lands on top of bytes that should not be there
  yet."** Falsified by the guest's own call order, and worth recording because the code comment in
  `apr_execute_read` ("prosper serves every read eagerly") invites exactly this hypothesis.
  `PROSPER_AMPRLOG` + `PROSPER_FILELOG` show `sceAmprAprCommandBufferReadFile` followed immediately,
  on the same guest thread with no intervening guest work, by `H896Pt-yB4I(CbSetEqueue)` and
  `sceKernelAprSubmitCommandBufferAndGetResult` (`-> token=0x209f (bound)`). Record and submit are
  microseconds apart while the clear is 0.5-2.3 s later, so deferring the read to submit cannot move
  it. #3142.

- **"The biggest dropped stage is the title-screen background."** Falsified: `0x3011560000` was the
  largest single loss on this route — one instruction discarding **1536 full-screen 3840×2160 draws
  per boot** — and fixing it (#3138) left `max_nonblack` unchanged at **0.0069**, a menu-only title
  screen. Draw-loss rank does not predict which draws carry the picture, so the next candidate should
  not be chosen by that rank alone. #3138.

One line per falsified hypothesis, the evidence, and the link. Do not restart these without
contradictory new evidence.

- **"The descriptor is at the right SGPR base and mis-classified as a buffer."** Falsified. A
  whole-table dump (SRSRC set + finished table + the guest's four raw sharp arrays, per stage) shows
  the tables for the dumped stages are complete: five declared T#s, five present. #3126.
- **"The table being consulted and the code being scanned belong to different stages."** Falsified by
  the same dump — same stage, and the join is exact once the sharp arrays are printed alongside. #3126.
- **"The lost `sreg_srt` tag explains the `srsrc=s8/s16` failures."** Void, not merely wrong: the
  diagnostic's own `written=0` says the shader never wrote those SGPRs, so nothing was `s_load`ed and
  no tag could be lost. The resolver correctly falls through to `by_sgpr_base`, which is what returns
  null. #3126.
- **"Admitting the dropped writable T#s (#3128) fixes the title screen."** One arm says no: admitting
  them as `StorageImage` left both `mimg-unresolved` (5) and `max_nonblack` (0.1097) exactly
  unchanged. That arm may have used a class/provenance the MIMG resolver does not match on, so the
  question is open — but do not assume the fix moves this title. #3128.
- **RETRACTED — "A wall-clock pad route reaches the title screen." (was: *Falsified*.)** The evidence
  was 0.1095 once, then 0.0063 / 0.0000 / 0.0063 on three unmodified reruns, read at the time as "the
  timed route works once by luck". Under the corrected number map that reading **inverts**: ~0.11 is
  the *calibration* screen and ~0.006 is the *title* screen, so **two** of the three "failures" (the
  0.0063 pair) are the runs that reached the title, the "lucky" 0.1095 sample is the one that did not,
  and 0.0000 is neither screen — a black frame is still a failure. Two of four, which is consistent
  with the "roughly one run in three" this doc records elsewhere; an earlier version of this row said
  three of four by counting the black frame as a success. This doc now commits a wall-clock
  route (`reach-title-hold.pad`, Cross at 150 s/160 s) as the title route for exactly that reason. The
  row is kept rather than deleted because the mistake it records — adopting a single sample as an A/B
  baseline — was real even though its conclusion was upside down. #3127.
- **"The unresolved image ops are what is missing from the title screen."** ~~Falsified by the
  dropped-draw census~~ — **this row was wrong and is retained as a warning.** The census reading (7
  of 1024 discarded draws are `shader-recompile`) is correct, and the inference from it is not: every
  one of those draws is a **full-screen 3840×2160 pass**. Draw *count* is not area, and the failing
  shaders are the composite. #3126.

**VOID, not falsified — every row below whose bullet begins `VOID`**, and only those. The scoping is
per-row and stated in the row itself, never positional: an earlier version of this paragraph said
"the next three rows", which silently changed meaning the moment a row was inserted, and did. Every one was measured
on a route that settles on the
**brightness-calibration** screen (`max_nonblack` 0.1140), not the title screen (0.0069). They are
correct statements about calibration and say nothing about the title screen, so they are neither
evidence nor falsifications for it. Re-run each with `PROSPER_NULL_PAGE=1` and a window that reaches
0.0069 before quoting any of them. #3126.

- **VOID — "prosper is losing `CB_TARGET_MASK` for the no-effect draws" (the #1946 shape).**
  `PROSPER_COLORSTATETRACE` reported the presence flag as 1 on 32,649 of 32,649 traced draws — on
  calibration. #3126.
- **VOID — "The dropped draws are where the missing picture is."** ~92% of draws carried a non-zero
  write mask and executed, and the colour-masked-off ones were a single fragment shader
  (`0x3010660000`) — on calibration. #3126.
- **VOID — "`no-effect(early)` is discarding content that should render."** One arm said no:
  `PROSPER_NO_EARLY_NO_EFFECT=1` moves the same draws to the late `no-effect` verdict with
  `max_nonblack` unchanged at 0.1140, and `PROSPER_FORCE_COLORWRITE=1` makes the composite **worse**
  (0.1140 → 0.0296). Neither lever admits content. That does not prove the masks were decoded
  correctly, only that admitting these draws wholesale is not the fix. #3126.
- **"The `srsrc=s0` failures are a size-0 slot wrongly claimed by the V# path."** Falsified by a
  built arm: making a guest-declared 8-dword slot decline the V# claim when its eight dwords decode
  as a valid T# changed nothing — on the draw that recompiles, those eight dwords are **residue**
  (`degenerate T#`), so the guard correctly falls through and the slot is still claimed. The real
  shape is #1590's: the T# is absent on the draw that happens to compile the shader. #3126.
- **"The world renders into an HDR target that the composite then loses."** Falsified: every seeded
  scene target in the title-screen frame is black at source (brightest channel 0-6 across five HDR
  targets). #3126.
- **VOID — "A skipped compute dispatch collapses the composite" (the charter's LUT/exposure shape).**
  `PROSPER_COMPUTE_PROGRAM_CENSUS` reports the only program with any skips at **executed=7455,
  skipped=2** — but that census was read on **calibration** and has not been re-run on the title
  screen, so it is not a falsification for it. This row said `Falsified` and was corrected: the
  measurement is real, the scope was overclaimed. #3126.
- **MIMG `SRSRC` is not a user-SGPR index.** It names any SGPR, so a scan that treats every SRSRC as
  a user-data slot reports mostly false positives — scratch registers the shader loaded a descriptor
  into. An earlier list of "bases missing from the table" derived that way is discarded. #3126.
- **"Widening `rdna2_mimg_zero_mip_shape()` to recognize the rejected `IMAGE_LOAD_MIP` NSA packet
  (#3134) closes the reject, the same way it did for GTA V's `IMAGE_STORE_MIP` NSA sibling."**
  Half right, half falsified. Fragment `0x30be800000`'s reject at pc=134
  (`words=f004070a,00080500,0000052a`) is byte-verified against llvm-mc gfx1030's
  `image_load_mip v[5:7], [v0, v42, v5], s[32:39] dmask:0x7 dim:SQ_RSRC_IMG_2D`, and widening the
  shape function to admit it (UNORM=0/GLC=0, dmask=0x7, matching the plain-assembly encoding rather
  than GTA V's UNORM=1/GLC=1/dmask∈{1,0xf}) does make `rdna2_mimg_zero_mip_shape()` correctly report
  `shape=1`/`mip_vgpr=5`. **But the resource this instruction addresses is not eligible for the
  "prove the mip is zero and discard it" fast path at all**: on the live `s14072.prgcap` capture it
  declares **six real mip levels**, is **in the mip tail**, and the guest's mip VGPR is **not
  provably zero** (`[mimg-mip] shape=1 proven_zero_mip=0 img_dim=1/1 mips=6 mip_tail=1
  dataformat=7(Uint16) ncomp=2`) — a small `R16G16_UINT` pyramid, most plausibly a UE4-style
  auto-exposure luminance chain the guest computes itself via its own mip-level writes, not a
  downsampled color texture. This is the same underlying gap #2818 found on *Sonic Frontiers*'
  compute-side `IMAGE_LOAD_MIP` (12-mip 2D_ARRAY, not in the mip tail) — prosper's graphics texture
  path only ever *generates* a mip chain by blitting level 0 (`tests/fixtures/render_runner.h`,
  gated to `VK_FORMAT_R8G8B8A8_UNORM`), never uploads the guest's own per-level bytes, and an integer
  format like this one cannot even use that generation path (no meaningful linear-filtered downsample
  of texel IDs, and Vulkan does not expose `SAMPLED_IMAGE_FILTER_LINEAR` on pure-integer formats
  regardless). So the emitter's `res->declared_mip_levels != 1u` / `!res->proven_zero_mip` gates are
  correctly still rejecting it after the shape widening — reading `Lod=0` or a blit-synthesized
  chain would both render a wrong image, which per #2818's own rule is worse than the current
  visible reject. The shape widening landed anyway (decode-level groundwork, tested against the
  exact captured bytes) because it makes the CPU-side zero-mip proof accurate for this encoding and
  may unblock some OTHER draw that shares it but addresses a genuinely single-mip resource; it does
  **not** close #3134 on its own. Full detail: `docs/RECOMPILER_REMAINING.md` § Ruled out. #3134,
  #2818.
- **"The three-dword `IMAGE_LOAD_MIP` is an exotic packet that needs its own lowering."** Falsified.
  NSA is an address ENCODING, not a different operation, and the defect was the emitter's
  expected-address-count rule: `rdna2_mimg_dynamic_mip_shape` derived the mip register as
  `VADDR + n`, which is only true for the consecutive form, so it declined every NSA packet rather
  than name the wrong VGPR. Assembled by hand through llvm-mc gfx1030, the NSA and consecutive forms
  carry the SAME address vector (`[x, y, mip]` at 2D, `[x, y, slice, mip]` at 2D_ARRAY); the
  emitter's own coordinate gather already read NSA generically. It is now read as an encoding and
  lowers through the #3048 dynamic-LOD path, with an executed arm that puts the mip in v7 while v2 —
  the register the old arithmetic would have named — holds zero. The second half of the same fix:
  `shader_resource_mip_chain_plan` refused any resource whose SELECTED level is packed in the shared
  mip tail, which refused this resource's whole class rather than a corner (32x32 at four bytes per
  texel is smaller than one 64 KiB macroblock, so all six levels are in the tail); it is now admitted
  after proving the plan's in-block coordinates equal the ones `agc_shader_layout` published. #3134.
- **STILL OPEN, and this is what now holds fragment `0x30be800000`:** both halves above are
  compute-path facts, and this is a FRAGMENT stage. The graphics texture path uploads exactly one
  level (its chain is *generated* by a `vkCmdBlitImage` cascade, gated to `VK_FORMAT_R8G8B8A8_UNORM`)
  and has no native sampled decode for `R16G16_UINT` at all — `backend_color_format` falls through to
  `VK_FORMAT_R8G8B8A8_UNORM` for it, so an integer fetch on this resource would read converted texels
  even at level zero. The reject line now says which of the two it is: `[mimg-mip] … dyn_shape=1
  nsa=1 materialized_mips=1 compute=0`. The disassembly is unambiguous that the level is genuinely
  dynamic — `v_min_i32 / v_max_i32` clamp a log2-derived LOD into `v5`, and the x/y coordinates are
  shifted right by that same `v5` — so `Lod=0` remains the wrong answer. #3134, #2818.
