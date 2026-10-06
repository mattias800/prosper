---
kind: status
status: current
---

# MOUSE: P.I. For Hire (`PPSA26578`) — status

Unity (IL2CPP) with the Universal Render Pipeline's deferred path. Tracker [#4556].

**Rung 3 as of 2026-10-06** — the scripted route reaches first-person gameplay and the room
renders: walls, doors, furniture, lamps, the HUD over it. Route
`scripts/mouse-pi-PPSA26578/reach-gameplay.pad`, a default launch, no switches.

It was rung 2 until that date, with the HUD over a black world. Two prosper defects were behind
that picture and neither was one the run log named:

1. **The decal buffer started each frame from transparent black instead of its clear colour**, and
   this is what made the world black. Fixed by [#4621].
2. **Every shadow atlas was thrown away before the lighting pass sampled it.** This cost shadows
   and a great deal of time, but not light. Fixed by [#4621].

[#4556]: https://github.com/mattias800/prosper/issues/4556
[#4621]: https://github.com/mattias800/prosper/pull/4621

## How a frame is built

Read from one captured gameplay frame (1,350 draws, 35 dispatches, 38 fragment programs), which is
worth having before touching this title because the passes depend on each other in ways the draw
list does not show.

| draws | what | writes |
| --- | --- | --- |
| 0–634 | two shadow atlases, 4096x2048, **Z_16**, reversed-Z | depth only |
| 636–937 | depth prepass | depth, normals, an 8-bit mask |
| 939–942 | **decals**, blended into the decal buffer | `0x…5c10000`, RGBA8 |
| 943–1244 | G-buffer; samples the decal buffer | six targets, stencil `0x20` for lit pixels |
| 1245 | the sun, full-screen, stencil-tested, shadowed | scene colour |
| 1246–1287 | 21 local lights as stencil light volumes, two shadowed | scene colour |
| 1288–1310 | half-resolution fog, transparents, a second depth target | scene colour |
| 1319–1352 | tone map, upscale to 1920x1080, HUD | the frame |

Stencil bits `0x60` carry the material type and `0x10` marks a light volume, which is the
Universal Render Pipeline's deferred layout. The decal buffer is its "DBuffer": the decal draws
blend with the destination alpha as a running transmittance, and the G-buffer pass reads the
result, so **the decal buffer's alpha multiplies every surface's base colour**. Clear it to
alpha 0 and the world is black except where a decal lies. That is the picture this title had.

## What the two defects were

**The decal buffer.** Before the decal draws, a compute dispatch fills the buffer's 24,576-byte
DCC metadata plane with `0x40`. That is a fast clear, and `0x40` is the code for (0,0,0,1).
prosper materialized such a clear only for a span that *samples* the cleared target; a span that
*renders* to it started from the draw's own clear colour, which for this title is zeros. Read back after the decal draws, the buffer was `00,00,00,00`
wherever no decal lay, and the G-buffer's first target held content in 5.0% of its pixels.

**The shadow atlases.** prosper decides whether a guest write makes a retained depth image stale
by asking whether the write can overlap the image's guest plane, and it sized every depth plane at
four bytes a texel. A Z_16 plane is two. The doubled range ran past the end of the atlas's
mapping, the mapping table could not place it, and an unplaceable range is treated as overlapping
everything. Over the 470 s route the 19 atlases took 6,633,602 invalidations, 6,626,134 of them
from writes nowhere near, and none from a write inside an atlas.

## Measured

`prosper-app`, visible window, `--present-mode immediate`, the scripted route, 470 s, before on
main `f9e7817fd` and after on the head of [#4621].

| | before | after |
| --- | --- | --- |
| gameplay frame at flip 2000, mean luma of 255 | 4.8 | 12.9 |
| … pixels darker than 16 | 95.4% | 85.0% |
| depth samples refused as invalid, whole route | 159,886 | 6,217 |
| … of which shadow atlases | at least 151,710 | 0 |
| host copies, whole route | 105,515 MiB | 15,688 MiB |
| … per flip | 30.8 MiB | 4.5 MiB |
| flips | 3,421 | 3,496 |

Do not read a frame rate out of the last row. The after run shared the GPU with another lane for
its second half (`gpu-sync-wait` fired from 238 s on), so the two are not comparable on time. The
cost of the refused atlases alone was measured on one binary, with `PROSPER_DS_BRIDGE_IGNORE_VALID`
as the switch under test: host copies over the route 105,515 MiB against 8,927 MiB, and 3,421
flips against 4,153. The faster arm also had `PROSPER_DSLOG` on, which costs it time, so that
flip figure understates the difference.

## What is still wrong

- **Under 10 frames a second in gameplay, about 3 in the menus.** Not investigated beyond the
  above, and not re-measured on a quiet GPU since the fixes. Host copying is no longer the
  obvious cost: it fell by 85% and the flip count barely moved.
- **Before gameplay, one depth target is fast-cleared through HTILE and sampled with nothing
  drawn into it**, and prosper answers that sample from guest memory instead of from the clear
  value ([#4620]). The answer is right here only because both are zero. In gameplay the target is
  drawn into every frame and served.
- **110 draws are dropped in one burst before gameplay, in some runs** ([#4612]): a 16-dword
  fragment program that does not decode. It appeared in two of the last six runs of the route.
- Whether the gameplay picture is *right* is not known. No PS5 reference has been compared. The
  room is dark with two bright lamps, which may be the scene's look.

[#4612]: https://github.com/mattias800/prosper/issues/4612
[#4620]: https://github.com/mattias800/prosper/issues/4620

## Ruled out

- **"Something refused, dropped or skipped is what darkens the scene."** The run named three
  losses: fragment draws refused for a width-sensitive load, compute dispatches skipped for an
  unimplemented `ds_read_b32 gds`, and a truncated fragment program. [#4609] removed the first
  (970–1,098 draws to 0) and [#4615] the second (1,131–1,930 dispatches to 0), and the gameplay
  frame was as dark as before, compared numerically. The third is a burst before gameplay.
- **"The shadow passes are given the wrong extent."** The suspicion was that a depth-only pass
  with masked 1422x799 colour targets still bound would size its depth image from them.
  `PROSPER_DSLOG=1` on the captured frame prints `viewport-derived extent 4096x2048 … -> accept`
  and `new-entry … extent=4096x2048` for both atlases.
- **"The refused shadow atlases are what darkens the scene."** They were refused, 159,886 times,
  but serving them does not brighten anything. With `PROSPER_DS_BRIDGE_IGNORE_VALID=1`, which
  reported itself firing, the gameplay frames have a mean luma of 3.1–3.4 against 4.8–4.9
  without. The atlases are reversed-Z (cleared to 0, tested GEQUAL) and their guest bytes are
  zero, so the zeros read in their place meant "nothing occludes". Fixing them adds shadows.
- **"The lighting pass's stencil tests are failing."** The stencil state decodes as the pipeline's
  deferred layout should (lit materials write `0x20` under mask `0x60`; light volumes invert
  `0x10` on depth fail; lights test EQUAL against them). The lights were running and multiplying
  by a black base colour.
- **"The texture cache budget is starving the materials."** The runs use a 1,024 MiB texture
  budget to stand in for a discrete card. No texture-cache refusal appears in the log, and the
  G-buffer's first target is just as empty in a replay at the default 4,096 MiB.
- **"A second depth target is refused in gameplay, right after two draws have written it."**
  Published as [#4620] and wrong. It was read off a refusal count for the whole route plus the
  draw order of one gameplay frame, and the two were never put on one timeline. An ordered
  `PROSPER_DSLOG=1` run does that: the refusals of that target reach 5,115 by 176 s and are
  still 5,115 at 293 s, and the first frame with the two depth-writing draws is at 173 s. Every
  refusal is from a frame before gameplay, where nothing draws into the target after its clear.

[#4609]: https://github.com/mattias800/prosper/pull/4609
[#4615]: https://github.com/mattias800/prosper/pull/4615

## Instrument notes specific to this title

- **A replayed frame of this title is faithful up to the end of the lighting pass and not after
  it**, and nothing in its output says where the line is. `gpu_replay` discards every retained
  depth image on every guest write ([#4619]), so the fog pass reads zero depth and the replayed
  frame comes out nearly white (mean luma 171.7) where the live frame is nearly black (4.82).
  Before that pass the replay matches the live run's own targets.
- **A frame bundle's seeds are the live run's own intermediate targets**, one frame old, and
  `gpu_replay --dump-rtt-seed 0xADDR out.bmp` reads them without executing anything. That is what
  showed the G-buffer's first target was black while its normals were a complete room: no replay
  fidelity question arises, because nothing is replayed.
- **`--dump-rtt-seed` and `--output-target-after` write pictures clamped to 0..1.** A linear-depth
  or HDR target reads as flat black or flat white. Add `--native-texel X,Y` for values; that is
  how the decal buffer's alpha was read.
- **Until [#4622], no frame of this title could be replayed at all.** Its alpha-tested shadow
  casters end with a conditional branch past their own `s_endpgm`, and replay admission refused
  the frame for it: `logical-wave replay original decode unavailable stage=fs pc=53`.
- **A whole-route counter is not a statement about gameplay.** The route spends its first 170 s
  in logos, menus and an intro that run the same deferred pipeline, so a `[dsbridge] declined`
  total, a dropped-draw total or a host-copy total mostly describes those. Read the counter at two
  moments, or put it beside the pass it is about, before attaching it to a gameplay frame.
- **A slow start desynchronizes the route.** The stick moves that choose a difficulty are
  anchored to flips 840–1000 and the presses around them to wall-clock seconds. A run that shares
  the GPU with another lane reaches those flips a minute late; this one still arrived, but check
  `actual_f*.bmp` rather than assuming.

[#4619]: https://github.com/mattias800/prosper/issues/4619
[#4622]: https://github.com/mattias800/prosper/pull/4622

## What the next lane should do

Compare the gameplay frame with a PS5 reference before optimising anything: the room renders,
and whether its lighting is right is unknown. The frame rate is the larger problem and has no
diagnosis yet; start from the `[perf-alarm]` summary of a run on a quiet GPU.
