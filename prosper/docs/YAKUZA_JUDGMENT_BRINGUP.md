# *Yakuza Kiwami* (`PPSA31334`) and *Judgment* (`PPSA02739`) — Ryu Ga Gotoku bring-up

Two SEGA / Ryu Ga Gotoku Studio titles that share an engine family (PAR archives, CRIWARE
middleware) and, on the import tables, the same `libSceAmpr` surface. They are documented together
because the *prediction* that they share a blocker is the interesting part of this page — and it is
half wrong, which is exactly the kind of thing that is expensive to rediscover.

Trackers: [#2864](https://github.com/mattias800/prosper/issues/2864) (Kiwami),
[#2880](https://github.com/mattias800/prosper/issues/2880) (Judgment).

## Where they stand (2026-08-22)

| | *Yakuza Kiwami* | *Judgment* |
| --- | --- | --- |
| Rung | 0 | 0 |
| Boots | yes | yes |
| Faults | no | no |
| Guest draws executed | 0 (every draw skipped by the recompiler) | yes — 139 in one submit, then ~1/frame |
| Frames presented | 1920x1080, all pure black | 3840x2160, all pure black |
| Next blocker | the generic NGG merged vertex prologue ([#2922](https://github.com/mattias800/prosper/issues/2922)) | not localised ([#2923](https://github.com/mattias800/prosper/issues/2923)); one unmapped colour-target format is the only concrete lead |

Route for both, `tools/screenshot`, default launch, no non-default switches:

```bash
PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct \
    ./prosper/build-linux/screenshot <DUMP_ROOT>/<TITLE_ID>-app0 \
    --seconds 10 --count 6 --out <OUTDIR> --allow-guest-fault
```

## *Yakuza Kiwami*: APR gather/scatter, and then the recompiler

The 0.0 s `SIGSEGV addr=0x1d0000` was AMM (#2875, landed). The 0.1 s `SIGSEGV addr=0x1d4` behind it
was `sceAmprAprCommandBufferReadFileGatherScatter` (#2872, landed): the title reads **each archive
as exactly two segments** — a plain `ReadFile` for the unaligned tail and one `…GatherScatter` for
the aligned head — so an unimplemented gather/scatter dropped the entire body of every file. The
guest said so in its own words (`Failed!! Load Devil2 Shader Archive`) and then called a method on
the null object the failed load left behind.

The segments TILE each file exactly, which is what makes the implementation checkable rather than
merely plausible:

| file | plain `ReadFile` | `…GatherScatter` | file size |
| --- | --- | --- | --- |
| `sh_devil_ps5.par` | off `0x240000`, `0x37800` | off `0x0`, `0x240000` | `0x277800` |
| `sh_ptc_ps5.par` | off `0x200000`, `0x18000` | off `0x0`, `0x200000` | `0x218000` |
| `lexus_shader_ps5.par` | off `0x400000`, `0xd800` | off `0x0`, `0x400000` | `0x40d800` |
| `scenario2.bin_c` | off `0x40000`, `0xbae` | off `0x0`, `0x40000` | `0x40bae` |

With that landed the title loads its archives, its scenario data and its `db.nex` tables, reaches
the live renderer, and runs indefinitely at 1920x1080 — **with every draw skipped**.

### The current blocker: the generic NGG merged vertex prologue

`PROSPER_DBG=1` over a 30 s boot, 20 distinct shader programs reached, **20 rejected**:

| instructions | reject | stage |
| --- | --- | --- |
| 19 | `bf8a0000` = `s_barrier` (SOPP `0x0a`) at pc=8 | vertex |
| 1 | `f0a0010a,0004000e,00000c07` = MIMG `0x28` with an extra address word | fragment |

The 19 are one shape. Their prologue is the stock AMD NGG merged-stage wrapper, e.g.
`exec_vs_10100805400`:

```
pc=0  s_branch 3
pc=1  s_bfe_u32 s107, s3, 0x00040018     ; wave id
pc=3  s_mov_b32 exec_lo, -1
pc=4  s_bfe_u32 s0, s2, 0x00090016       ; vertex count
pc=6  s_bfe_u32 s1, s2, 0x0009000c       ; primitive count
pc=8  s_barrier                          <-- rejected here
pc=14 s_sendmsg 0x9                      ; GS_ALLOC_REQ
pc=16 v_mbcnt_lo_u32_b32 v1, -1, vcc_lo  ; lane index
pc=20 exp prim
```

`s_barrier` is rejected at `src/gpu/recompiler/rdna2_emit_alu.cpp` (`case 0x0a`) unless
`b.is_compute || b.ngg_private_lds`, and in the vertex shell `ngg_private_lds` is set only for a
byte-exact captured Astro Bot wrapper. That is deliberate and recorded in the code: every wave/peer
approximation there is an exception for that one wrapper, not a property of `GS_ALLOC_REQ`. Lowering
this family properly is the **NGG preamble** item already on `RECOMPILER_REMAINING.md`'s list, and
it is a frontier, not a missing opcode (#2922). Do not reach for a "drop the barrier" shortcut: these
programs pack vertices through LDS, so dropping it silently produces wrong geometry.

pc=3 used to be the reject instead, and that one *was* a missing lowering — see
`RECOMPILER_REMAINING.md`.

## *Judgment*: not the same wall, despite the same imports

Tracker #2880 predicted, from the import tables alone and correctly labelled as a prediction, that
#2875's AMM work would matter here too. Measured: **AMM works and is not the wall.** A default boot
reserves the window and takes the pool

```
[amm] virtual-address window [0x10000000000,0x11100000000) reserved
[amm] direct-memory pool [0x55200000,0x2d5200000) (10240 MiB, type 1, align 0x200000)
```

and then the title runs for **600 s without faulting**, never resolves a single APR file id, and
never calls `sceAmprAprCommandBufferReadFileGatherScatter` at all. So the two titles do **not**
currently share a blocker: Kiwami's frontier is behind Judgment's.

What Judgment does do, measured with `tools/hle_calls` over a 300-tick window on a live boot:
it submits real GPU work (`agc_dcb_draw_index_auto` 71, `agc_driver_submit_dcb` 70,
`agc_cb_dispatch` 70), compiles 119 shaders with **zero recompiler rejections**, and presents 4K
frames — 139 draws in one submit and then one draw per submit for hundreds of submits. Every
presented frame is pure black, and `tools/screenshot` reports `0%` of 25,375 published frames
carrying new content. It also polls `scePlayGoGetProgress` / `GetEta` / `GetLocus` continuously,
which prosper answers as fully installed.

The one concrete lead is a single unmapped colour-target format, logged once per boot:

```
[gpu] vk_color_format: unmapped CB surface format=0x1 number_type=0 comp_swap=3 -> Undefined
```

`COLOR_8` / `UNORM` / `comp_swap=3`. `vk_translate.cpp`'s `COLOR_8` row accepts `comp_swap=0` only.
**The meaning of `comp_swap=3` on a one-component surface is not derivable from anything in this
dump**, so it is recorded as a lead and not guessed — see the issue rather than inventing a row.

## Ruled out

One line per already-falsified hypothesis. Extend it; do not re-derive these.

- **"Judgment is blocked on the same libSceAmpr AMM functions as Yakuza Kiwami" (#2880's opening
  prediction).** Falsified 2026-08-22 by a default 600 s boot on master: AMM reserves its window and
  serves its pool, the title never faults, and it never reaches an APR read at all. The prediction
  was correctly labelled as one; it is recorded here so the next lane does not spend the AMM budget
  again.
- **"Judgment's black frames are a stalled or absent renderer."** Falsified 2026-08-22: it executes
  real draws (139 in one submit), compiles 119 shaders with zero recompiler rejections, and presents
  4K frames continuously. The defect is in what those draws produce, not in whether they happen.
- **"Yakuza Kiwami's wall is the renderer / the shader recompiler" (the reading before #2872).**
  Falsified 2026-08-21/22: the title printed its own `Failed!! Load … Shader Archive`, and the
  archives' bytes provably never arrived — the gather/scatter half of every two-segment read was
  discarded by an unimplemented NID. The recompiler only became the frontier *after* the bytes
  arrived.
- **"`sceAmprAprCommandBufferReadFileGatherScatter` needs a new descriptor-list construct."** The
  issue text guessed the destination might be a descriptor list. Falsified 2026-08-22 from the SDK's
  own inline wrapper at `eboot+0xcc4a70`: it is the plain `ReadFile` wrapper with the file id removed
  and every later argument shifted down one register — one `(dst, len, fileOffset)` segment, not a
  list.
